"""Data loading functions for neurite features and design files."""

import re
import pandas as pd
from pathlib import Path


def load_neurite_data(feature_csv_path, neurite_features):
    """
    Load neurite morphology features from CSV file.
    Mirrors the exact logic from drug_screen.ipynb notebook.
    
    Parameters
    ----------
    feature_csv_path : str or Path
        Path to the feature CSV file
    neurite_features : list
        List of feature column names to load
        
    Returns
    -------
    pd.DataFrame
        DataFrame with neurite features, Plate_full, Plate_base, WellID, Plate_Well columns
    """
    feature_csv_path = Path(feature_csv_path)
    fname = feature_csv_path.name
    
    # Load data
    df = pd.read_csv(feature_csv_path, dtype=str)
    
    # Find well column (mirroring notebook logic)
    well_col = None
    for cand in ["WellID", "Well_Id", "Well_ID", "Well", "well", "Well Name"]:
        if cand in df.columns:
            well_col = cand
            break
    if well_col is None:
        raise KeyError(f"No well column in {fname}")
    
    # Extract plate token from filename (mirroring notebook logic)
    if "QR19" in fname or "QR23" in fname:
        # Special handling for QR19_QR23 plate
        plate_full = "QR19_QR23"
        plate_base = "QR19_QR23"
    else:
        # Extract BA###Q#A/B pattern (e.g., BA071Q1A)
        m = re.search(r'BA\d+Q\d+[AB]', fname)
        if m:
            plate_full = m.group(0)  # e.g., "BA071Q1A"
            plate_base = re.sub(r'[AB]$', '', plate_full)  # Strip A/B -> "BA071Q1"
        else:
            raise ValueError(f"Cannot extract plate identifier from {fname}")
    
    # Add plate and well identifiers
    df["Plate_full"] = plate_full
    df["Plate_base"] = plate_base
    df["WellID"] = df[well_col].astype(str).str.upper().str.strip()
    df["Plate_Well"] = df["Plate_full"] + "_" + df["WellID"]
    
    # Convert neurite features to numeric
    for col in neurite_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_design_file(design_csv_path):
    """
    Load experimental design file.
    Mirrors the exact logic from drug_screen.ipynb notebook.
    
    Parameters
    ----------
    design_csv_path : str or Path
        Path to the design CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with design information (plate, well, genotype, treatment, drug, etc.)
    """
    design_df = pd.read_csv(design_csv_path, dtype=str)
    
    # Standardize columns (strip whitespace only, preserve case for Plate_ID)
    for col in ["Plate_ID", "Well_ID", "Row", "Col", "SampleID", "Genotype", "Treatment", "Dosage", "DrugID"]:
        if col in design_df.columns:
            design_df[col] = design_df[col].astype(str).str.strip()
    
    # Normalize Well_ID to uppercase
    if "Well_ID" in design_df.columns:
        design_df["Well_ID"] = design_df["Well_ID"].str.upper()
    
    return design_df

