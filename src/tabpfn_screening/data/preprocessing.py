"""Data preprocessing and merging functions."""

import pandas as pd
import numpy as np


def merge_design_with_features(neurite_df, design_df, is_qr_plate=False):
    """
    Merge design metadata with neurite features.
    Mirrors the exact logic from drug_screen.ipynb notebook.
    
    Parameters
    ----------
    neurite_df : pd.DataFrame
        DataFrame with neurite features (must have Plate_base, WellID columns)
    design_df : pd.DataFrame
        DataFrame with design information (must have Plate_ID, Well_ID columns)
    is_qr_plate : bool
        If True, merge only on WellID (for QR19_QR23). If False, merge on both Plate_base and WellID.
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with features and metadata
    """
    if is_qr_plate:
        # For QR19_QR23, merge only on WellID
        merged = neurite_df.merge(
            design_df,
            how="left",
            left_on="WellID",
            right_on="Well_ID",
            suffixes=("", "_design"),
        )
        # Add Treatment as DMSO for consistency (as in notebook)
        if "Treatment" not in merged.columns:
            merged["Treatment"] = "0.05% DMSO"
    else:
        # For regular plates, merge on Plate_base and WellID
        merged = neurite_df.merge(
            design_df,
            how="left",
            left_on=["Plate_base", "WellID"],
            right_on=["Plate_ID", "Well_ID"],
            suffixes=("", "_design"),
        )
    
    # Handle different column names for drug identifier
    # Some design files use "Treatment", others use "DrugID"
    if "DrugID" not in merged.columns and "Treatment" in merged.columns:
        merged["DrugID"] = merged["Treatment"]
    
    return merged


def normalize_columns(df):
    """
    Normalize genotype and drug columns for consistent matching.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized columns added
    """
    df = df.copy()
    
    # Normalize genotype
    if "Genotype" in df.columns:
        df["Genotype_norm"] = df["Genotype"].astype(str).str.lower().str.strip()
    
    # Normalize drug ID
    if "DrugID" in df.columns:
        df["DrugID_norm"] = df["DrugID"].astype(str).str.upper().str.strip()
    
    return df


def remove_nan_features(df, feature_cols):
    """
    Remove rows with NaN values in feature columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_cols : list
        List of feature column names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with NaN rows removed
    int
        Number of rows removed
    """
    X = df[feature_cols].values
    nan_mask = np.isnan(X).any(axis=1)
    n_removed = nan_mask.sum()
    
    if n_removed > 0:
        df_clean = df[~nan_mask].copy()
        return df_clean, n_removed
    
    return df, 0

