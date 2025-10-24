"""Plotting functions for visualizations."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, confusion_matrix
from pathlib import Path


def plot_lda_projection(X, y, probs, output_path, title, group_size):
    """
    Plot LDA projection with predicted probabilities.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        True labels (1 = control, 0 = patient)
    probs : np.ndarray
        Predicted probabilities
    output_path : str or Path
        Output file path
    title : str
        Plot title
    group_size : int
        Group size
    """
    # Fit LDA
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    lda_1d = lda.transform(X).ravel()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ctrl_mask = y == 1
    pat_mask = y == 0
    
    ax.scatter(lda_1d[ctrl_mask], probs[ctrl_mask], 
              c='blue', alpha=0.5, s=30, label='Control DMSO')
    ax.scatter(lda_1d[pat_mask], probs[pat_mask], 
              c='red', alpha=0.5, s=30, label='Patient DMSO')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision boundary')
    ax.set_xlabel('LDA Component 1', fontsize=12)
    ax.set_ylabel('P(Control)', fontsize=12)
    ax.set_title(f'{title} - Group Size {group_size}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return lda


def plot_roc_curve(y_true, y_prob, auc_score, output_path, title, group_size):
    """
    Plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    auc_score : float
        AUC score
    output_path : str or Path
        Output file path
    title : str
        Plot title
    group_size : int
        Group size
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'TabPFN (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} - Group Size {group_size}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_density(probs_dict, mean_prob, output_path, title, ylog=False):
    """
    Plot probability density for multiple groups.
    
    Parameters
    ----------
    probs_dict : list of tuples
        List of (label, probabilities, color) tuples
    mean_prob : float
        Mean probability to mark with vertical line
    output_path : str or Path
        Output file path
    title : str
        Plot title
    ylog : bool, optional
        Use log scale for y-axis (default: False)
    """
    x = np.linspace(0, 1, 200)
    plt.figure(figsize=(7, 5))
    
    for label, arr, color in probs_dict:
        if len(arr) > 1:
            kde = gaussian_kde(arr)
            plt.plot(x, kde(x), label=label, color=color)
    
    if mean_prob is not None:
        plt.axvline(mean_prob, linestyle="--", alpha=0.8, 
                   label=f"Drug mean = {mean_prob:.3f}", color="black")
    
    if ylog:
        plt.yscale("log")
        plt.ylabel("Density (log)")
    else:
        plt.ylabel("Density")
    
    plt.xlabel("P(Control)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path, title):
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    output_path : str or Path
        Output file path
    title : str
        Plot title
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    
    plt.figure(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Pred: Control', 'Pred: Patient'],
                yticklabels=['True: Control', 'True: Patient'])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_drug_lda_projection(
    lda, 
    X_baseline, 
    y_baseline, 
    probs_baseline,
    X_drug, 
    drug_probs, 
    drug_name,
    output_path, 
    group_size
):
    """
    Plot LDA projection with drug samples overlaid on baseline DMSO data.
    
    Parameters
    ----------
    lda : LinearDiscriminantAnalysis
        Fitted LDA model
    X_baseline : np.ndarray
        Baseline (DMSO) feature matrix
    y_baseline : np.ndarray
        Baseline labels
    probs_baseline : np.ndarray
        Baseline predicted probabilities
    X_drug : np.ndarray
        Drug-treated feature matrix
    drug_probs : np.ndarray
        Drug predicted probabilities
    drug_name : str
        Name of the drug
    output_path : str or Path
        Output file path
    group_size : int
        Group size
    """
    # Transform data
    lda_1d = lda.transform(X_baseline).ravel()
    drug_lda_1d = lda.transform(X_drug).ravel()
    
    mean_prob = float(drug_probs.mean())
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ctrl_mask = y_baseline == 1
    pat_mask = y_baseline == 0
    
    # Plot baseline data
    ax.scatter(lda_1d[ctrl_mask], probs_baseline[ctrl_mask], 
              c='blue', alpha=0.4, s=35, label='Control DMSO', edgecolors='none')
    ax.scatter(lda_1d[pat_mask], probs_baseline[pat_mask], 
              c='red', alpha=0.4, s=35, label='Patient DMSO', edgecolors='none')
    
    # Highlight drug samples
    ax.scatter(drug_lda_1d, drug_probs, 
              c='green', s=70, alpha=0.9, edgecolors='darkgreen', 
              linewidth=1.5, label=f'{drug_name} (n={len(drug_probs)})', marker='D')
    
    # Add decision boundary
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.6, 
              label='Decision boundary (P=0.5)')
    
    # Add mean probability line for the drug
    ax.axhline(y=mean_prob, color='green', linestyle=':', linewidth=2, alpha=0.8, 
              label=f'Drug mean P(Control) = {mean_prob:.3f}')
    
    ax.set_xlabel('LDA Component 1', fontsize=13)
    ax.set_ylabel('P(Control)', fontsize=13)
    
    closer = "Yes" if mean_prob > 0.5 else "No"
    ax.set_title(f'LDA Classifier Projection: {drug_name}\n' + 
                f'Mean P(Control) = {mean_prob:.3f} (Closer to Control = {closer})', 
                fontsize=14, weight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

