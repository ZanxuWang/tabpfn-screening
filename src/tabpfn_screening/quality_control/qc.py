"""Quality control functions using Median Absolute Deviation (MAD)."""

import numpy as np
import pandas as pd
from pathlib import Path


def apply_mad_qc(df, threshold=2.0, min_neurons=0, per_plate=True):
    """
    Apply Median Absolute Deviation (MAD) quality control to filter outlier wells.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with Plate_Well column (and Plate_full column if per_plate=True)
    threshold : float, optional
        MAD threshold for outlier detection (default: 2.0)
    min_neurons : int, optional
        Minimum neurons per well threshold (default: 0)
    per_plate : bool, optional
        Apply QC per plate separately (default: True)
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    dict
        QC report with statistics
    """
    # Determine plate column to use
    plate_col = None
    if per_plate:
        if "Plate_full" in df.columns:
            plate_col = "Plate_full"
        elif "Plate" in df.columns:
            plate_col = "Plate"
    
    if per_plate and plate_col is not None:
        # Apply QC per plate and combine
        print(f"  Applying per-plate QC using column: {plate_col}")
        print(f"  Found {df[plate_col].nunique()} unique plates: {sorted(df[plate_col].unique())}")
        
        filtered_dfs = []
        total_kept = 0
        total_filtered_min = 0
        total_filtered_mad = 0
        
        for plate_id in df[plate_col].unique():
            plate_df = df[df[plate_col] == plate_id].copy()
            plate_filtered, _ = _apply_mad_qc_single(plate_df, threshold, min_neurons)
            filtered_dfs.append(plate_filtered)
            
            # Track statistics
            well_counts_before = plate_df.groupby("Plate_Well").size()
            well_counts_after = plate_filtered.groupby("Plate_Well").size()
            total_kept += len(well_counts_after)
        
        # Combine filtered data
        df_filtered = pd.concat(filtered_dfs, ignore_index=True)
        
        # Generate combined report
        well_counts_before = df.groupby("Plate_Well").size()
        well_counts_after = df_filtered.groupby("Plate_Well").size()
        
        qc_report = {
            'method': 'MAD (per-plate)' if per_plate else 'MAD',
            'threshold': threshold,
            'min_neurons': min_neurons,
            'total_wells_before': len(well_counts_before),
            'wells_kept': len(well_counts_after),
            'wells_filtered_min': 0,  # Calculated per-plate
            'wells_filtered_mad': len(well_counts_before) - len(well_counts_after),
            'wells_filtered_total': len(well_counts_before) - len(well_counts_after),
            'filter_rate': (len(well_counts_before) - len(well_counts_after)) / len(well_counts_before) * 100,
            'mean_neurons_before': float(well_counts_before.mean()),
            'std_neurons_before': float(well_counts_before.std()),
            'mean_neurons_after': float(well_counts_after.mean()) if len(well_counts_after) > 0 else 0,
            'std_neurons_after': float(well_counts_after.std()) if len(well_counts_after) > 0 else 0,
            'median': float(well_counts_before.median()),
            'mad': 0,  # Not meaningful when combining per-plate results
        }
        
        return df_filtered, qc_report
    else:
        # Apply QC globally
        if per_plate:
            print(f"  WARNING: per_plate=True but no plate column found. Applying global QC instead.")
        return _apply_mad_qc_single(df, threshold, min_neurons)


def _apply_mad_qc_single(df, threshold=2.0, min_neurons=0):
    """
    Apply MAD QC to a single plate or combined data.
    
    Internal function used by apply_mad_qc.
    """
    # Count neurons per well
    well_counts = df.groupby("Plate_Well").size()
    
    # Calculate MAD
    median = np.median(well_counts)
    mad = np.median(np.abs(well_counts - median))
    
    # Filter wells
    kept_wells = []
    filtered_by_min = []
    filtered_by_mad = []
    
    for well_id, count in well_counts.items():
        # Check minimum threshold
        if count < min_neurons:
            filtered_by_min.append(well_id)
            continue
        
        # Check MAD threshold
        if mad > 0:
            mad_score = abs(count - median) / mad
            if mad_score > threshold:
                filtered_by_mad.append(well_id)
                continue
        
        kept_wells.append(well_id)
    
    # Filter DataFrame
    df_filtered = df[df["Plate_Well"].isin(kept_wells)].copy()
    
    # Generate report
    well_counts_after = df_filtered.groupby("Plate_Well").size()
    
    qc_report = {
        'method': 'MAD',
        'threshold': threshold,
        'min_neurons': min_neurons,
        'total_wells_before': len(well_counts),
        'wells_kept': len(kept_wells),
        'wells_filtered_min': len(filtered_by_min),
        'wells_filtered_mad': len(filtered_by_mad),
        'wells_filtered_total': len(filtered_by_min) + len(filtered_by_mad),
        'filter_rate': (len(filtered_by_min) + len(filtered_by_mad)) / len(well_counts) * 100,
        'mean_neurons_before': float(well_counts.mean()),
        'std_neurons_before': float(well_counts.std()),
        'mean_neurons_after': float(well_counts_after.mean()) if len(well_counts_after) > 0 else 0,
        'std_neurons_after': float(well_counts_after.std()) if len(well_counts_after) > 0 else 0,
        'median': float(median),
        'mad': float(mad),
    }
    
    return df_filtered, qc_report


def save_qc_report(qc_report, output_dir):
    """
    Save QC report to text file.
    
    Parameters
    ----------
    qc_report : dict
        QC report dictionary
    output_dir : str or Path
        Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "qc_report_mad.txt"
    
    report_text = f"""
Quality Control Report - MAD Method
=====================================

Method: {qc_report['method']}
Threshold: {qc_report['threshold']}
Minimum neurons threshold: {qc_report['min_neurons']}

Wells Statistics:
-----------------
Total wells before QC: {qc_report['total_wells_before']}
Wells kept: {qc_report['wells_kept']}
Wells filtered (total): {qc_report['wells_filtered_total']} ({qc_report['filter_rate']:.1f}%)
  - By minimum threshold: {qc_report['wells_filtered_min']}
  - By statistical QC (MAD): {qc_report['wells_filtered_mad']}

Neuron Count Statistics:
-------------------------
Mean neurons/well: {qc_report['mean_neurons_before']:.1f} -> {qc_report['mean_neurons_after']:.1f}
Std neurons/well: {qc_report['std_neurons_before']:.1f} -> {qc_report['std_neurons_after']:.1f}
Median: {qc_report['median']:.1f}
MAD: {qc_report['mad']:.1f}
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"QC report saved to: {report_path}")

