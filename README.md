# TabPFN Drug Screening Pipeline

A modular pipeline for high-throughput drug screening using TabPFN (Tabular Prior-Fitted Networks) with neurite morphology features.

## Overview

This pipeline trains TabPFN classifiers to distinguish between control and patient neuron phenotypes, then screens drugs to identify compounds that shift patient phenotypes toward control-like morphology.

### Key Features

- **Automated Quality Control**: MAD-based filtering of outlier wells
- **Grouped Neuron Profiles**: Aggregate features from multiple neurons per well
- **TabPFN Classification**: State-of-the-art tabular classifier (no hyperparameter tuning needed)
- **Drug Screening**: Rank drugs by predicted control-like phenotype restoration
- **Rich Visualizations**: LDA projections, ROC curves, probability densities, confusion matrices
- **Modular Design**: Easy to extend and customize

## Installation

### 1. Clone/Download Repository

```bash
cd D:\
# Repository should be at D:\tabpfn-screening
```

### 2. Create Environment

```bash
# Using conda (recommended)
conda create -n tabpfn-screening python=3.10
conda activate tabpfn-screening

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
cd D:\tabpfn-screening
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Data Setup

Place your data in the `data/microsam_features/` directory:

```
data/
└── microsam_features/
    ├── design/
    │   ├── 3-13-2025_design.csv
    │   ├── 10-16-2024_design_BX.csv
    │   └── ...
    ├── Xu_BA071Q1A_R_annotate_file.csv
    ├── Xu_BA071Q1B_annotate_file.csv
    ├── Xu_BA071_CP_1_annotate_file.csv
    ├── 3-13-2025_QR19_QR23_annotate_file.csv
    └── ...
```

## Usage

### Training Pipeline

Train TabPFN models on multiple group sizes:

```bash
python scripts/train.py \
    --data-dir data/microsam_features \
    --output-dir results/training \
    --group-sizes 50,60,70,80,90,100 \
    --qc-threshold 2.0 \
    --min-neurons 0 \
    --random-seed 42 \
    --device cuda \
    --n-top-drugs 10
```

**Arguments:**
- `--data-dir`: Path to microsam_features directory
- `--output-dir`: Where to save models and results
- `--group-sizes`: Comma-separated list of neuron group sizes to test
- `--qc-threshold`: MAD threshold for quality control (default: 2.0)
- `--min-neurons`: Minimum neurons per well (default: 0)
- `--random-seed`: Random seed for reproducibility (default: 42)
- `--device`: 'cpu' or 'cuda' for GPU acceleration (default: cpu)
- `--n-top-drugs`: Number of top/bottom drugs to visualize (default: 10)

**Outputs:**
- `tabpfn_model_group{size}.joblib` - Trained models
- `selected_features_group{size}.json` - Feature lists
- `drug_rankings_group{size}.csv` - Ranked drug lists
- `classifier_projection_group{size}.png` - LDA projections
- `roc_group{size}.png` - ROC curves
- `drug_visualizations_group{size}/` - Per-drug visualizations
- `group_size_comparison.csv` - Summary of all group sizes

### Validation Pipeline

Validate trained models on a new plate:

```bash
python scripts/validate.py \
    --data-dir data/microsam_features \
    --validation-plate Xu_BA071_CP_1_annotate_file.csv \
    --design-file design/10-16-2024_design_BX.csv \
    --model-dir results/training \
    --output-dir results/validation \
    --group-sizes 50,60,70,80,90,100 \
    --qc-threshold 2.0 \
    --random-seed 42 \
    --n-top-drugs 10
```

**Arguments:**
- `--data-dir`: Path to microsam_features directory
- `--validation-plate`: Validation plate filename
- `--design-file`: Design file path (relative to data-dir)
- `--model-dir`: Directory with trained models
- `--output-dir`: Where to save validation results
- `--group-sizes`: Group sizes to validate (must match trained models)
- `--qc-threshold`: MAD threshold for QC
- `--random-seed`: Random seed
- `--n-top-drugs`: Number of drugs to visualize

**Outputs:**
- `drug_rankings_validation_group{size}.csv` - Drug rankings
- `classifier_projection_validation_group{size}.png` - LDA projections
- `roc_validation_group{size}.png` - ROC curves
- `drug_visualizations_group{size}/` - Per-drug visualizations
- `validation_summary.csv` - Summary of validation performance

## Project Structure

```
tabpfn-screening/
├── data/                          # Data directory (add your files here)
│   └── microsam_features/
├── results/                       # Output directory (auto-created)
├── src/
│   └── tabpfn_screening/          # Main package
│       ├── data/                  # Data loading and preprocessing
│       ├── quality_control/       # QC filtering
│       ├── features/              # Feature engineering (grouping)
│       ├── models/                # Model training and prediction
│       ├── screening/             # Drug screening logic
│       ├── visualization/         # Plotting and reports
│       └── utils/                 # Helper functions
├── scripts/
│   ├── train.py                   # Training CLI
│   └── validate.py                # Validation CLI
├── notebooks/                     # Original Jupyter notebooks
│   ├── drug_screen.ipynb
│   └── drug_screen_validate.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

## Pipeline Overview

### 1. Data Loading
- Load neurite morphology features from CSV files
- Merge with experimental design metadata
- Normalize genotype and drug labels

### 2. Quality Control
- Count neurons per well
- Apply MAD-based outlier detection
- Filter wells with abnormal neuron counts
- Generate QC reports

### 3. Feature Grouping
- Group neurons within each well (e.g., groups of 50)
- Aggregate features (mean and median)
- Create training samples from grouped profiles

### 4. Model Training
- Train TabPFN classifier (Control vs Patient on DMSO)
- Evaluate with train/test split
- Save model, features, and performance metrics

### 5. Drug Screening
- Predict P(Control) for patient samples + each drug
- Rank drugs by mean P(Control)
- Higher score = more control-like = potential rescue

### 6. Visualization
- LDA projections showing drug effects
- ROC curves for model performance
- Probability density distributions
- Confusion matrices
- Per-drug summary statistics

## Output Interpretation

### Drug Rankings

Drugs are ranked by `MeanProbControl`:
- **High scores (>0.5)**: Drug shifts patient neurons toward control phenotype
- **Low scores (<0.5)**: Drug maintains or worsens patient phenotype

### LDA Projections

- **X-axis**: Linear discriminant (control vs patient separation)
- **Y-axis**: Predicted P(Control) from TabPFN
- **Green diamonds**: Drug-treated patient neurons
- **Blue dots**: Control DMSO neurons
- **Red dots**: Patient DMSO neurons

Drugs clustering near control DMSO are promising rescue candidates.

## Tips

1. **Group Size Selection**: Test multiple sizes (50-100). Smaller = more samples but noisier; larger = fewer samples but more stable.

2. **QC Threshold**: If too many wells filtered, increase `--qc-threshold` to 3.0 or higher.

3. **Validation**: Use a different plate from training to assess generalization.

4. **Device**: If you have a GPU, use `--device cuda` for faster training (requires CUDA-enabled PyTorch).

5. **Top Drugs**: Visualize top 10-20 drugs to see clear patterns. Too many visualizations can be overwhelming.

## Troubleshooting

### "Insufficient DMSO data"
- QC filtered out all DMSO wells
- Solution: Relax `--qc-threshold` or use `--min-neurons 0`

### "Missing features"
- Validation plate doesn't have same features as training
- Solution: Ensure same feature extraction pipeline

### Out of Memory
- Reduce group sizes or use fewer neurons
- Use `--device cpu` if GPU memory is limited



