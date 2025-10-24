"""TabPFN model training functions."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from tabpfn import TabPFNClassifier
from ..utils.io import save_model, save_feature_list


def train_tabpfn_model(
    X, 
    y, 
    feature_names,
    test_size=0.2,
    random_state=42,
    n_ensemble=8,
    device='cpu'
):
    """
    Train a TabPFN classifier for control vs patient classification.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Binary labels (1 = control, 0 = patient)
    feature_names : list
        List of feature names
    test_size : float, optional
        Proportion of data for testing (default: 0.2)
    random_state : int, optional
        Random seed (default: 42)
    n_ensemble : int, optional
        Number of ensemble models (default: 8)
    device : str, optional
        Device to use ('cpu' or 'cuda', default: 'cpu')
        
    Returns
    -------
    model : TabPFNClassifier
        Trained model
    results : dict
        Training results with metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize TabPFN (mirroring notebook - just use device)
    model = TabPFNClassifier(device=device)
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate on train set
    train_probs = model.predict_proba(X_train)[:, 1]
    train_preds = (train_probs > 0.5).astype(int)
    train_auc = roc_auc_score(y_train, train_probs)
    train_acc = accuracy_score(y_train, train_preds)
    
    # Evaluate on test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    test_auc = roc_auc_score(y_test, test_probs)
    test_acc = accuracy_score(y_test, test_preds)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds, labels=[1, 0])
    tn, fp = cm[1, 1], cm[1, 0]
    fn, tp = cm[0, 1], cm[0, 0]
    
    # Results
    results = {
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_auc': float(train_auc),
        'train_acc': float(train_acc),
        'test_auc': float(test_auc),
        'test_acc': float(test_acc),
        'confusion_matrix': {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }
    }
    
    return model, results


def save_trained_model(model, feature_names, results, output_dir, group_size):
    """
    Save trained model, features, and results.
    
    Parameters
    ----------
    model : TabPFNClassifier
        Trained model
    feature_names : list
        List of feature names
    results : dict
        Training results
    output_dir : Path
        Output directory
    group_size : int
        Group size used for training
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"tabpfn_model_group{group_size}.joblib"
    save_model(model, model_path)
    
    # Save feature list
    features_path = output_dir / f"selected_features_group{group_size}.json"
    save_feature_list(feature_names, features_path)
    
    # Save results
    from ..utils.io import save_json
    results_path = output_dir / f"training_results_group{group_size}.json"
    save_json(results, results_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Features saved to: {features_path}")
    print(f"Results saved to: {results_path}")

