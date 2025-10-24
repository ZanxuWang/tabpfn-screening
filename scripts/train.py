"""
Training script for TabPFN drug screening.

This script trains TabPFN classifiers for multiple group sizes and performs
drug screening on the training data.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tabpfn_screening.data import (
    load_neurite_data,
    load_design_file,
    merge_design_with_features,
    normalize_columns,
    remove_nan_features,
)
from tabpfn_screening.quality_control import apply_mad_qc, save_qc_report
from tabpfn_screening.features import create_grouped_profiles
from tabpfn_screening.models.trainer import train_tabpfn_model, save_trained_model
from tabpfn_screening.screening import screen_drugs, rank_drugs
from tabpfn_screening.visualization import (
    plot_lda_projection,
    plot_roc_curve,
    plot_probability_density,
    plot_confusion_matrix,
    plot_drug_lda_projection,
    generate_drug_summary,
)
from tabpfn_screening.utils import slug

# Feature definitions
NEURITE_FEATURES = [
    "N_node", "Soma_surface", "N_stem", "Number_of_Nodes", "Soma_Surface", "Number_of_Stems",
    "Number_of_Bifurcations", "Number_of_Branches", "Number_of_Tips", "Overall_Width",
    "Overall_Height", "Average_Diameter", "Total_Length", "Total_Surface", "Total_Volume",
    "Max_Euclidean_Distance", "Max_Path_Distance", "Max_Branch_Order", "Average_Contraction",
    "Average_Fragmentation", "Average_PD_Ratio", "Avg_Bif_Angle_Local", "Avg_Bif_Angle_Remote",
    "Hausdorff_Dimension",
]

METADATA_COLS = ["SampleID", "Genotype", "Treatment", "Dosage", "DrugID", "Plate_full", "Plate_base", "Row", "Col", "Plate_Well", "WellID"]


def main():
    parser = argparse.ArgumentParser(description="Train TabPFN drug screening models")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to microsam_features directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--group-sizes", type=str, default="50,60,70,80,90,100", 
                       help="Comma-separated list of group sizes (default: 50,60,70,80,90,100)")
    parser.add_argument("--qc-threshold", type=float, default=2.0, 
                       help="MAD QC threshold (default: 2.0)")
    parser.add_argument("--min-neurons", type=int, default=0, 
                       help="Minimum neurons per well (default: 0)")
    parser.add_argument("--qc-per-plate", action="store_true", default=True,
                       help="Apply QC per plate (default: True)")
    parser.add_argument("--qc-global", dest="qc_per_plate", action="store_false",
                       help="Apply QC globally across all plates")
    parser.add_argument("--random-seed", type=int, default=42, 
                       help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device for TabPFN: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--n-top-drugs", type=int, default=10, 
                       help="Number of top/bottom drugs to visualize (default: 10)")
    
    args = parser.parse_args()
    
    # Parse paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse group sizes
    group_sizes = [int(x.strip()) for x in args.group_sizes.split(",")]
    
    print("=" * 60)
    print("TabPFN Drug Screening - Training Pipeline")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Group sizes: {group_sizes}")
    print(f"QC threshold (MAD): {args.qc_threshold}")
    print(f"Min neurons per well: {args.min_neurons}")
    print(f"Random seed: {args.random_seed}")
    print(f"Device: {args.device}")
    print()
    
    # ====================LOAD DATA ====================
    print("Loading training data...")
    
    # Define training plates
    training_plates = [
        "Xu_BA071Q1A_R_annotate_file.csv",
        "Xu_BA071Q1B_annotate_file.csv",
        "Xu_BA071Q2A_annotate_file.csv",
        "Xu_BA071Q2B_annotate_file.csv",
        "Xu_BA071Q3A_annotate_file.csv",
        "Xu_BA071Q3B_annotate_file.csv",
        "Xu_BA071Q4A_annotate_file.csv",
        "Xu_BA071Q4B_annotate_file.csv",
        "3-13-2025_QR19_QR23_annotate_file.csv",
    ]
    
    # Load design files (training uses multiple design files)
    training_design_files = [
        "6-26-2024_plate1_design.csv",
        "6-26-2024_plate2_design.csv",
        "6-26-2024_plate3_design.csv",
        "6-26-2024_plate4_design.csv",
    ]
    qr_design_file = "3-13-2025_design.csv"
    
    print(f"Loading design files: {training_design_files}")
    design_dfs = []
    for design_file in training_design_files:
        design_path = data_dir / "design" / design_file
        if design_path.exists():
            df = load_design_file(design_path)
            design_dfs.append(df)
        else:
            print(f"Warning: {design_file} not found, skipping...")
    
    # Load QR design file
    qr_design_path = data_dir / "design" / qr_design_file
    if qr_design_path.exists():
        print(f"Loading QR design file: {qr_design_file}")
        qr_design_df = load_design_file(qr_design_path)
    
    # Combine design files for main training plates
    design_df = pd.concat(design_dfs, ignore_index=True)
    print(f"Combined design data shape: {design_df.shape}")
    
    # Load and combine all training plates (mirroring notebook logic)
    combined_dfs = []
    for plate_file in training_plates:
        plate_path = data_dir / plate_file
        if not plate_path.exists():
            print(f"Warning: {plate_file} not found, skipping...")
            continue
        
        print(f"Loading: {plate_file}")
        df = load_neurite_data(plate_path, NEURITE_FEATURES)
        
        # Use the correct design file and merge strategy
        is_qr = "QR" in plate_file
        current_design_df = qr_design_df if is_qr else design_df
        
        merged = merge_design_with_features(df, current_design_df, is_qr_plate=is_qr)
        
        # Debug: Check if Genotype column exists after merge
        genotype_na_count = merged["Genotype"].isna().sum() if "Genotype" in merged.columns else len(merged)
        non_na = len(merged) - genotype_na_count
        print(f"  Merged: {non_na}/{len(merged)} rows with Genotype")
        
        combined_dfs.append(merged)
    
    # Combine all data
    combined_data = pd.concat(combined_dfs, ignore_index=True)
    combined_data = normalize_columns(combined_data)
    
    print(f"Combined training data shape: {combined_data.shape}")
    print(f"Control samples: {(combined_data['Genotype_norm'] == 'control').sum()}")
    print(f"Patient samples: {(combined_data['Genotype_norm'] == 'patient').sum()}")
    print()
    
    # ==================== QC ====================
    qc_method = "per-plate" if args.qc_per_plate else "global"
    print(f"Applying Quality Control (MAD method, {qc_method})...")
    qc_data, qc_report = apply_mad_qc(
        combined_data, 
        threshold=args.qc_threshold, 
        min_neurons=args.min_neurons,
        per_plate=args.qc_per_plate
    )
    
    qc_dir = output_dir / "quality_control"
    save_qc_report(qc_report, qc_dir)
    
    print(f"QC complete: {qc_report['wells_kept']} / {qc_report['total_wells_before']} wells kept")
    print(f"QC report saved to: {qc_dir}")
    print()
    
    # ==================== TRAIN MODELS ====================
    results_summary = []
    
    for group_size in group_sizes:
        print("=" * 60)
        print(f"Processing Group Size: {group_size}")
        print("=" * 60)
        
        # Create grouped profiles
        print(f"Creating grouped profiles (group_size={group_size})...")
        grouped_df = create_grouped_profiles(
            qc_data, 
            group_size, 
            NEURITE_FEATURES, 
            METADATA_COLS, 
            random_seed=args.random_seed
        )
        
        print(f"Grouped data shape: {grouped_df.shape}")
        
        # Normalize columns after grouping (to create Genotype_norm and DrugID_norm)
        grouped_df = normalize_columns(grouped_df)
        
        # Get aggregated features
        aggregated_features = [f"{f}_{stat}" for f in NEURITE_FEATURES for stat in ("mean", "median")]
        available_features = [f for f in aggregated_features if f in grouped_df.columns]
        print(f"Available features: {len(available_features)}")
        
        # Extract DMSO data (control vs patient)
        # Filter for DMSO samples (check if "dmso" is in the DrugID string, case-insensitive)
        dmso_mask = grouped_df["DrugID_norm"].astype(str).str.lower().str.contains("dmso", na=False)
        ctrl_dmso_mask = (grouped_df["Genotype_norm"] == "control") & dmso_mask
        pat_dmso_mask = (grouped_df["Genotype_norm"] == "patient") & dmso_mask
        
        print(f"DMSO samples - Control: {ctrl_dmso_mask.sum()}, Patient: {pat_dmso_mask.sum()}")
        
        dmso_mask = ctrl_dmso_mask | pat_dmso_mask
        dmso_df = grouped_df[dmso_mask].copy()
        
        X_dmso = dmso_df[available_features].values
        y_dmso = (dmso_df["Genotype_norm"] == "control").astype(int).values
        
        # Remove NaN
        dmso_df_clean, n_removed = remove_nan_features(dmso_df, available_features)
        if n_removed > 0:
            print(f"Removed {n_removed} rows with NaN values")
            X_dmso = dmso_df_clean[available_features].values
            y_dmso = (dmso_df_clean["Genotype_norm"] == "control").astype(int).values
        
        print(f"Training samples: {len(X_dmso)} (Control: {y_dmso.sum()}, Patient: {(y_dmso == 0).sum()})")
        
        # Train TabPFN
        print("Training TabPFN model...")
        model, train_results = train_tabpfn_model(
            X_dmso, 
            y_dmso, 
            available_features,
            test_size=0.2,
            random_state=args.random_seed,
            n_ensemble=8,
            device=args.device
        )
        
        print(f"Train AUC: {train_results['train_auc']:.4f}, Test AUC: {train_results['test_auc']:.4f}")
        print(f"Train Acc: {train_results['train_acc']:.4f}, Test Acc: {train_results['test_acc']:.4f}")
        
        # Save model
        save_trained_model(model, available_features, train_results, output_dir, group_size)
        
        # ==================== DRUG SCREENING ====================
        print("\nScreening drugs...")
        drug_results = screen_drugs(grouped_df, model, available_features)
        drug_ranking_df = rank_drugs(drug_results, group_size)
        
        # Save rankings
        ranking_path = output_dir / f"drug_rankings_group{group_size}.csv"
        drug_ranking_df.to_csv(ranking_path, index=False)
        print(f"Drug rankings saved to: {ranking_path}")
        
        if not drug_ranking_df.empty:
            print(f"\nTop 10 drugs (group_size={group_size}):")
            print(drug_ranking_df.head(10).to_string(index=False))
        
        # ==================== VISUALIZATIONS ====================
        print("\nGenerating visualizations...")
        
        # Get all DMSO predictions
        dmso_probs = model.predict_proba(X_dmso)[:, 1]
        dmso_preds = (dmso_probs > 0.5).astype(int)
        
        from sklearn.metrics import roc_auc_score
        dmso_auc = roc_auc_score(y_dmso, dmso_probs)
        
        # LDA projection
        lda = plot_lda_projection(
            X_dmso, y_dmso, dmso_probs,
            output_dir / f'classifier_projection_group{group_size}.png',
            'LDA Classifier Projection',
            group_size
        )
        
        # ROC curve
        plot_roc_curve(
            y_dmso, dmso_probs, dmso_auc,
            output_dir / f'roc_group{group_size}.png',
            'ROC Curve',
            group_size
        )
        
        # ==================== DRUG VISUALIZATIONS ====================
        drug_viz_dir = output_dir / f"drug_visualizations_group{group_size}"
        drug_viz_dir.mkdir(exist_ok=True)
        
        # Select top and bottom drugs
        top_drugs = list(drug_ranking_df.head(args.n_top_drugs)["DrugID"])
        bottom_drugs = list(drug_ranking_df.tail(args.n_top_drugs)["DrugID"])
        selected_drugs = list(dict.fromkeys(top_drugs + bottom_drugs))[:20]
        
        print(f"Generating visualizations for {len(selected_drugs)} drugs...")
        
        ctrl_probs = dmso_probs[y_dmso == 1]
        pat_probs = dmso_probs[y_dmso == 0]
        
        for drug_name in selected_drugs:
            drug_mask = (grouped_df["Genotype_norm"] == "patient") & (grouped_df["DrugID"] == drug_name)
            if not drug_mask.any():
                continue
            
            drug_df = grouped_df.loc[drug_mask].copy()
            X_drug = drug_df[available_features].values
            
            # Remove NaN
            drug_nan_mask = np.isnan(X_drug).any(axis=1)
            if drug_nan_mask.any():
                drug_df = drug_df[~drug_nan_mask].copy()
                X_drug = X_drug[~drug_nan_mask]
            
            if len(drug_df) == 0:
                continue
            
            drug_probs = model.predict_proba(X_drug)[:, 1]
            mean_prob = float(drug_probs.mean())
            safe_name = slug(f"{drug_name}_group{group_size}")
            
            # LDA projection
            plot_drug_lda_projection(
                lda, X_dmso, y_dmso, dmso_probs,
                X_drug, drug_probs, drug_name,
                drug_viz_dir / f"{safe_name}__lda_projection.png",
                group_size
            )
            
            # Probability densities
            probs_dict = [
                ("Control DMSO", ctrl_probs, "blue"),
                ("Patient DMSO", pat_probs, "red"),
                (drug_name, drug_probs, "green"),
            ]
            
            plot_probability_density(
                probs_dict, mean_prob,
                drug_viz_dir / f"{safe_name}__prob_density_log.png",
                "P(Control) density (log)", ylog=True
            )
            plot_probability_density(
                probs_dict, mean_prob,
                drug_viz_dir / f"{safe_name}__prob_density.png",
                "P(Control) density", ylog=False
            )
            
            # Confusion matrix
            y_true_drug = np.zeros(len(drug_probs), dtype=int)
            y_pred_drug = (drug_probs > 0.5).astype(int)
            plot_confusion_matrix(
                y_true_drug, y_pred_drug,
                drug_viz_dir / f"{safe_name}__confusion.png",
                f"Confusion: {drug_name}"
            )
            
            # Summary statistics
            drug_lda_1d = lda.transform(X_drug).ravel()
            ctrl_centroid_lda = np.mean(lda.transform(X_dmso[y_dmso == 1]).ravel())
            pat_centroid_lda = np.mean(lda.transform(X_dmso[y_dmso == 0]).ravel())
            
            generate_drug_summary(
                drug_name, drug_probs, drug_lda_1d,
                ctrl_centroid_lda, pat_centroid_lda,
                len(drug_df['Plate_Well'].unique()),
                group_size, "Training Data",
                drug_viz_dir / f"{safe_name}__summary.txt"
            )
        
        # Store results
        results_summary.append({
            'group_size': int(group_size),
            'n_groups': int(len(grouped_df)),
            'n_features': int(len(available_features)),
            'train_auc': float(train_results['train_auc']),
            'test_auc': float(train_results['test_auc']),
            'train_acc': float(train_results['train_acc']),
            'test_acc': float(train_results['test_acc']),
            'n_train': int(train_results['n_train']),
            'n_test': int(train_results['n_test']),
            'n_drugs_tested': int(len(drug_results)),
        })
        
        print(f"\nGroup size {group_size} complete!")
        print()
    
    # ==================== SUMMARY ====================
    print("=" * 60)
    print("Training Complete - Summary")
    print("=" * 60)
    
    results_df = pd.DataFrame(results_summary)
    summary_path = output_dir / "group_size_comparison.csv"
    results_df.to_csv(summary_path, index=False)
    
    print("\nGroup Size Comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    
    # Find optimal
    optimal_idx = results_df['test_auc'].idxmax()
    optimal_group_size = results_df.loc[optimal_idx, 'group_size']
    optimal_auc = results_df.loc[optimal_idx, 'test_auc']
    
    print(f"\nOptimal Configuration:")
    print(f"  Group size: {optimal_group_size}")
    print(f"  Test AUC: {optimal_auc:.4f}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Training pipeline complete!")


if __name__ == "__main__":
    main()

