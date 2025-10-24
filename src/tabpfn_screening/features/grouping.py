"""Neuron grouping and profile aggregation functions."""

import numpy as np
import pandas as pd


def create_grouped_profiles(df, group_size, neurite_features, metadata_cols, random_seed=42):
    """
    Create grouped neuron profiles by aggregating features within each well.
    
    For each well, neurons are randomly shuffled and grouped into batches of `group_size`.
    Features are aggregated (mean and median) for each group.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with individual neuron features
    group_size : int
        Number of neurons per group
    neurite_features : list
        List of neurite feature column names
    metadata_cols : list
        List of metadata column names to preserve
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with grouped profiles (one row per group)
    """
    np.random.seed(random_seed)
    
    def group_and_aggregate(well_data):
        """Group neurons within a single well and aggregate features."""
        n_neurons = len(well_data)
        
        # If fewer neurons than group_size, keep as single group
        if n_neurons < group_size:
            groups = [well_data]
        else:
            # Shuffle and create groups
            shuffled_indices = np.random.permutation(n_neurons)
            n_complete_groups = n_neurons // group_size
            groups = []
            
            # Complete groups
            for i in range(n_complete_groups):
                start_idx = i * group_size
                end_idx = start_idx + group_size
                group_indices = shuffled_indices[start_idx:end_idx]
                groups.append(well_data.iloc[group_indices])
            
            # Remaining neurons (if any)
            remaining = n_neurons % group_size
            if remaining > 0:
                remaining_indices = shuffled_indices[n_complete_groups * group_size:]
                groups.append(well_data.iloc[remaining_indices])
        
        # Aggregate each group
        aggregated_groups = []
        for i, group in enumerate(groups):
            agg_data = {}
            
            # Aggregate features
            feat_cols = [c for c in neurite_features if c in group.columns]
            for feat in feat_cols:
                agg_data[f"{feat}_mean"] = group[feat].mean()
                agg_data[f"{feat}_median"] = group[feat].median()
            
            # Keep metadata (take first value)
            for meta in metadata_cols:
                if meta in group.columns:
                    s = group[meta].dropna()
                    agg_data[meta] = s.iloc[0] if len(s) > 0 else np.nan
            
            # Add group info
            agg_data["Group_ID"] = i
            agg_data["Group_Size"] = len(group)
            
            aggregated_groups.append(agg_data)
        
        return pd.DataFrame(aggregated_groups)
    
    # Process each well
    grouped_dfs = []
    for well_id in df["Plate_Well"].unique():
        well_data = df[df["Plate_Well"] == well_id]
        well_grouped = group_and_aggregate(well_data)
        grouped_dfs.append(well_grouped)
    
    return pd.concat(grouped_dfs, ignore_index=True)

