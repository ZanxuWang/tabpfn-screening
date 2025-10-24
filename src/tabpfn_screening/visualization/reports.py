"""Report generation functions."""

import numpy as np
from pathlib import Path


def generate_drug_summary(
    drug_name,
    drug_probs,
    drug_lda_1d,
    ctrl_centroid_lda,
    pat_centroid_lda,
    n_wells,
    group_size,
    plate_name,
    output_path
):
    """
    Generate drug summary statistics report.
    
    Parameters
    ----------
    drug_name : str
        Drug name
    drug_probs : np.ndarray
        Predicted probabilities for drug
    drug_lda_1d : np.ndarray
        LDA projection values for drug
    ctrl_centroid_lda : float
        Control centroid in LDA space
    pat_centroid_lda : float
        Patient centroid in LDA space
    n_wells : int
        Number of wells
    group_size : int
        Group size
    plate_name : str
        Plate name
    output_path : str or Path
        Output file path
    """
    mean_prob = float(drug_probs.mean())
    drug_mean_lda = np.mean(drug_lda_1d)
    
    dist_to_ctrl = abs(drug_mean_lda - ctrl_centroid_lda)
    dist_to_pat = abs(drug_mean_lda - pat_centroid_lda)
    closer_to = "Control" if dist_to_ctrl < dist_to_pat else "Patient"
    
    stats_text = (
        f"Drug: {drug_name}\n"
        f"Group Size: {group_size}\n"
        f"Plate: {plate_name}\n"
        f"\n"
        f"Classifier Predictions:\n"
        f"----------------------\n"
        f"Mean P(Control): {mean_prob:.3f}\n"
        f"Std P(Control): {np.std(drug_probs):.3f}\n"
        f"Min P(Control): {np.min(drug_probs):.3f}\n"
        f"Max P(Control): {np.max(drug_probs):.3f}\n"
        f"Samples classified as Control: {(drug_probs > 0.5).sum()} ({(drug_probs > 0.5).mean()*100:.1f}%)\n"
        f"Samples classified as Patient: {(drug_probs <= 0.5).sum()} ({(drug_probs <= 0.5).mean()*100:.1f}%)\n"
        f"Total samples: {len(drug_probs)}\n"
        f"Wells: {n_wells}\n"
        f"\n"
        f"LDA Projection Analysis:\n"
        f"-----------------------\n"
        f"Drug mean LDA: {drug_mean_lda:.3f}\n"
        f"Drug std LDA: {np.std(drug_lda_1d):.3f}\n"
        f"Control centroid LDA: {ctrl_centroid_lda:.3f}\n"
        f"Patient centroid LDA: {pat_centroid_lda:.3f}\n"
        f"Distance to Control: {dist_to_ctrl:.3f}\n"
        f"Distance to Patient: {dist_to_pat:.3f}\n"
        f"Closer to: {closer_to}\n"
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(stats_text)

