"""Drug screening and ranking functions."""

import numpy as np
import pandas as pd


def screen_drugs(grouped_df, model, feature_cols, dmso_label='DMSO'):
    """
    Screen drugs using a trained classifier.
    
    For each drug, predict P(Control) for patient samples treated with that drug.
    Higher P(Control) suggests the drug pushes patient neurons toward control-like phenotype.
    
    Parameters
    ----------
    grouped_df : pd.DataFrame
        Grouped profiles DataFrame with DrugID and Genotype_norm columns
    model : TabPFNClassifier
        Trained classifier model
    feature_cols : list
        List of feature column names to use
    dmso_label : str, optional
        Label for DMSO in DrugID column (default: 'DMSO')
        
    Returns
    -------
    list of dict
        List of drug results with DrugID, MeanProbControl, and n_samples
    """
    # Get all drugs (excluding DMSO)
    all_drugs = [d for d in grouped_df["DrugID"].unique() 
                 if str(d).upper() != dmso_label.upper()]
    
    drug_results = []
    
    for drug in all_drugs:
        # Get patient samples treated with this drug
        drug_mask = (
            (grouped_df["Genotype_norm"] == "patient") &
            (grouped_df["DrugID"] == drug)
        )
        
        if not drug_mask.any():
            continue
        
        drug_df = grouped_df.loc[drug_mask].copy()
        X_drug = drug_df[feature_cols].values
        
        # Check for NaN
        drug_nan_mask = np.isnan(X_drug).any(axis=1)
        if drug_nan_mask.any():
            drug_df = drug_df[~drug_nan_mask].copy()
            X_drug = X_drug[~drug_nan_mask]
        
        if len(drug_df) == 0:
            continue
        
        # Predict
        drug_probs = model.predict_proba(X_drug)[:, 1]
        mean_prob_control = float(drug_probs.mean())
        
        drug_results.append({
            "DrugID": drug,
            "MeanProbControl": mean_prob_control,
            "n_samples": int(len(drug_probs)),
        })
    
    return drug_results


def rank_drugs(drug_results, group_size, ascending=False):
    """
    Rank drugs by mean P(Control).
    
    Parameters
    ----------
    drug_results : list of dict
        Drug screening results
    group_size : int
        Group size used
    ascending : bool, optional
        Sort in ascending order (default: False, i.e., descending)
        
    Returns
    -------
    pd.DataFrame
        Ranked drugs DataFrame
    """
    df = pd.DataFrame(drug_results)
    
    if df.empty:
        return df
    
    # Add group size column
    df['group_size'] = int(group_size)
    
    # Sort by mean probability
    df = df.sort_values("MeanProbControl", ascending=ascending, ignore_index=True)
    
    return df

