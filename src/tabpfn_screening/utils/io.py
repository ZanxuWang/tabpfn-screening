"""Input/Output utility functions."""

import json
import joblib
from pathlib import Path


def save_json(data, filepath):
    """
    Save data to JSON file.
    
    Parameters
    ----------
    data : dict
        Data to save
    filepath : str or Path
        Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load data from JSON file.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    dict
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_feature_list(features, filepath):
    """
    Save feature list to JSON file.
    
    Parameters
    ----------
    features : list
        List of feature names
    filepath : str or Path
        Output file path
    """
    save_json({'features': features}, filepath)


def load_feature_list(filepath):
    """
    Load feature list from JSON file.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    list
        List of feature names
    """
    data = load_json(filepath)
    return data['features']


def save_model(model, filepath):
    """
    Save model using joblib.
    
    Parameters
    ----------
    model : object
        Model to save
    filepath : str or Path
        Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load model using joblib.
    
    Parameters
    ----------
    filepath : str or Path
        Input file path
        
    Returns
    -------
    object
        Loaded model
    """
    return joblib.load(filepath)

