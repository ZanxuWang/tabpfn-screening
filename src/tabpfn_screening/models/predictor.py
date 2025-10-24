"""Model loading and prediction functions."""

from pathlib import Path
from ..utils.io import load_model, load_feature_list


def load_trained_model(model_dir, group_size):
    """
    Load a trained TabPFN model and its feature list.
    
    Parameters
    ----------
    model_dir : str or Path
        Directory containing saved models
    group_size : int
        Group size of the model to load
        
    Returns
    -------
    model : TabPFNClassifier
        Loaded model
    feature_names : list
        List of feature names used by the model
    """
    model_dir = Path(model_dir)
    
    # Load model
    model_path = model_dir / f"tabpfn_model_group{group_size}.joblib"
    model = load_model(model_path)
    
    # Load features
    features_path = model_dir / f"selected_features_group{group_size}.json"
    feature_names = load_feature_list(features_path)
    
    return model, feature_names


def predict_with_model(model, X):
    """
    Make predictions with a trained model.
    
    Parameters
    ----------
    model : TabPFNClassifier
        Trained model
    X : np.ndarray
        Feature matrix (n_samples, n_features)
        
    Returns
    -------
    probs : np.ndarray
        Predicted probabilities for control class (n_samples,)
    preds : np.ndarray
        Binary predictions (1 = control, 0 = patient)
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)
    return probs, preds

