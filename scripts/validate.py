"""
Validation script for TabPFN drug screening.

This script loads pre-trained TabPFN models and validates them on a new plate.
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
from tabpfn_screening.models.predictor import load_trained_model
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
    parser = argparse.ArgumentParser(description="Validate TabPFN drug screening models")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to microsam_features directory")
    parser.add_argument("--validation-plate", type=str, required=True, help="Validation plate filename")
    parser.add_argument("--design-file", type=str, required=True, help="Design file path (relative to data-dir)")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with trained models")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for validation results")
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
    parser.add_argument("--n-top-drugs", type=int, default=10, 
                       help="Number of top/bottom drugs to visualize (default: 10)")
    
    args = parser.parse_args()
    
    # Parse paths
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse group sizes
    group_sizes = [int(x.strip()) for x in args.group_sizes.split(",")]
    
    print("=" * 60)
    print("TabPFN Drug Screening - Validation Pipeline")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Validation plate: {args.validation_plate}")
    print(f"Design file: {args.design_file}")
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Group sizes: {group_sizes}")
    print(f"QC threshold (MAD): {args.qc_threshold}")
    print(f"Random seed: {args.random_seed}")
    print()
    
    # ==================== LOAD DATA ====================
    print("Loading validation data...")
    
    # Load design file
    design_path = data_dir / args.design_file
    print(f"Loading design file: {design_path}")
    design_df = load_design_file(design_path)
    
    # Load validation plate
    validation_path = data_dir / args.validation_plate
    print(f"Loading validation plate: {validation_path.name}")
    df = load_neurite_data(validation_path, NEURITE_FEATURES)
    
    # Determine if it's a QR plate
    is_qr = "QR" in args.validation_plate
    merged = merge_design_with_features(df, design_df, is_qr_plate=is_qr)
    merged = normalize_columns(merged)
    
    print(f"Validation data shape: {merged.shape}")
    print(f"Control samples: {(merged['Genotype_norm'] == 'control').sum()}")
    print(f"Patient samples: {(merged['Genotype_norm'] == 'patient').sum()}")
    print()
    
    # ==================== QC ====================
    qc_method = "per-plate" if args.qc_per_plate else "global"
    print(f"Applying Quality Control (MAD method, {qc_method})...")
    qc_data, qc_report = apply_mad_qc(
        merged, 
        threshold=args.qc_threshold, 
        min_neurons=args.min_neurons,
        per_plate=args.qc_per_plate
    )
    
    qc_dir = output_dir / "quality_control"
    save_qc_report(qc_report, qc_dir)
    
    print(f"QC complete: {qc_report['wells_kept']} / {qc_report['total_wells_before']} wells kept")
    print(f"QC report saved to: {qc_dir}")
    print()
    
    # ==================== VALIDATE MODELS ====================
    validation_results = []
    
    for group_size in group_sizes:
        print("=" * 60)
        print(f"Validating Group Size: {group_size}")
        print("=" * 60)
        
        try:
            # Load trained model
            print(f"Loading trained model for group_size={group_size}...")
            model, feature_names = load_trained_model(model_dir, group_size)
            print(f"Loaded model with {len(feature_names)} features")
            
            # Create grouped profiles
            print(f"Creating grouped profiles (group_size={group_size})...")
            grouped_df = create_grouped_profiles(
                qc_data, 
                group_size, 
                NEURITE_FEATURES, 
                METADATA_COLS, 
                random_seed=args.random_seed
            )
            
            print(f"Grouped validation data shape: {grouped_df.shape}")
            
            # Normalize columns after grouping (to create Genotype_norm and DrugID_norm)
            grouped_df = normalize_columns(grouped_df)
            
            # Check features
            missing_features = [f for f in feature_names if f not in grouped_df.columns]
            if missing_features:
                print(f"Warning: Missing {len(missing_features)} features, skipping...")
                continue
            
            # Extract DMSO data (filter for strings containing "dmso", case-insensitive)
            dmso_str_mask = grouped_df["DrugID_norm"].astype(str).str.lower().str.contains("dmso", na=False)
            ctrl_dmso_mask = (grouped_df["Genotype_norm"] == "control") & dmso_str_mask
            pat_dmso_mask = (grouped_df["Genotype_norm"] == "patient") & dmso_str_mask
            
            print(f"DMSO samples - Control: {ctrl_dmso_mask.sum()}, Patient: {pat_dmso_mask.sum()}")
            
            dmso_mask = ctrl_dmso_mask | pat_dmso_mask
            
            if not (ctrl_dmso_mask.any() and pat_dmso_mask.any()):
                print("Insufficient DMSO data, skipping...")
                continue
            
            dmso_df = grouped_df[dmso_mask].copy()
            X_dmso = dmso_df[feature_names].values
            y_dmso = (dmso_df["Genotype_norm"] == "control").astype(int).values
            
            # Remove NaN
            dmso_df_clean, n_removed = remove_nan_features(dmso_df, feature_names)
            if n_removed > 0:
                print(f"Removed {n_removed} rows with NaN values")
                X_dmso = dmso_df_clean[feature_names].values
                y_dmso = (dmso_df_clean["Genotype_norm"] == "control").astype(int).values
            
            print(f"Validation samples: {len(X_dmso)} (Control: {y_dmso.sum()}, Patient: {(y_dmso == 0).sum()})")
            
            # Predict
            dmso_probs = model.predict_proba(X_dmso)[:, 1]
            dmso_preds = (dmso_probs > 0.5).astype(int)
            
            from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
            dmso_auc = roc_auc_score(y_dmso, dmso_probs)
            dmso_acc = accuracy_score(y_dmso, dmso_preds)
            
            # Confusion matrix
            cm = confusion_matrix(y_dmso, dmso_preds, labels=[1, 0])
            tn, fp = cm[1, 1], cm[1, 0]
            fn, tp = cm[0, 1], cm[0, 0]
            
            print(f"Validation AUC: {dmso_auc:.4f}, Accuracy: {dmso_acc:.4f}")
            print(f"Confusion Matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
            
            # ==================== DRUG SCREENING ====================
            print("\nScreening drugs...")
            drug_results = screen_drugs(grouped_df, model, feature_names)
            drug_ranking_df = rank_drugs(drug_results, group_size)
            
            # Save rankings
            ranking_path = output_dir / f"drug_rankings_validation_group{group_size}.csv"
            drug_ranking_df.to_csv(ranking_path, index=False)
            print(f"Drug rankings saved to: {ranking_path}")
            
            if not drug_ranking_df.empty:
                print(f"\nTop 10 drugs (group_size={group_size}):")
                print(drug_ranking_df.head(10).to_string(index=False))
            
            # ==================== VISUALIZATIONS ====================
            print("\nGenerating visualizations...")
            
            # LDA projection
            lda = plot_lda_projection(
                X_dmso, y_dmso, dmso_probs,
                output_dir / f'classifier_projection_validation_group{group_size}.png',
                'LDA Classifier Projection (Validation)',
                group_size
            )
            
            # ROC curve
            plot_roc_curve(
                y_dmso, dmso_probs, dmso_auc,
                output_dir / f'roc_validation_group{group_size}.png',
                'ROC Curve (Validation)',
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
                X_drug = drug_df[feature_names].values
                
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
                    group_size, args.validation_plate,
                    drug_viz_dir / f"{safe_name}__summary.txt"
                )
            
            # Store results
            validation_results.append({
                'group_size': int(group_size),
                'dmso_auc': float(dmso_auc),
                'dmso_acc': float(dmso_acc),
                'n_dmso_samples': int(len(X_dmso)),
                'n_features': int(len(feature_names)),
                'n_drugs_tested': int(len(drug_results)),
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            })
            
            print(f"\nGroup size {group_size} validation complete!")
            print()
            
        except Exception as e:
            print(f"Error processing group size {group_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ==================== SUMMARY ====================
    print("=" * 60)
    print("Validation Complete - Summary")
    print("=" * 60)
    
    if len(validation_results) == 0:
        print("No validation results generated!")
        return
    
    results_df = pd.DataFrame(validation_results)
    summary_path = output_dir / "validation_summary.csv"
    results_df.to_csv(summary_path, index=False)
    
    print("\nValidation Results Summary:")
    print(results_df.to_string(index=False))
    print(f"\nSummary saved to: {summary_path}")
    
    # Find best performing
    best_idx = results_df['dmso_auc'].idxmax()
    best_group_size = results_df.loc[best_idx, 'group_size']
    best_auc = results_df.loc[best_idx, 'dmso_auc']
    
    print(f"\nBest Validation Performance:")
    print(f"  Group size: {best_group_size}")
    print(f"  Validation AUC: {best_auc:.4f}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Validation pipeline complete!")


if __name__ == "__main__":
    main()

