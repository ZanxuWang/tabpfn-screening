# Repository Structure

## Complete File Tree

```
D:\tabpfn-screening/
│
├── README.md                       # Main documentation
├── STRUCTURE.md                    # This file
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation script
├── .gitignore                      # Git ignore rules
│
├── data/                           # ← PUT YOUR DATA HERE
│   └── microsam_features/          # Your neurite features and design files
│       ├── design/
│       │   └── *.csv
│       └── *.csv
│
├── results/                        # Output directory (auto-created)
│   ├── training/                   # Training outputs
│   └── validation/                 # Validation outputs
│
├── notebooks/                      # Original Jupyter notebooks
│   ├── drug_screen.ipynb           # Training notebook
│   └── drug_screen_validate.ipynb  # Validation notebook
│
├── scripts/                        # CLI entry points
│   ├── train.py                    # Training pipeline
│   └── validate.py                 # Validation pipeline
│
└── src/
    └── tabpfn_screening/           # Main package
        ├── __init__.py
        │
        ├── data/                   # Data loading and preprocessing
        │   ├── __init__.py
        │   ├── loader.py           # Load CSV files, extract plate numbers
        │   └── preprocessing.py    # Merge design with features, normalize
        │
        ├── quality_control/        # QC filtering
        │   ├── __init__.py
        │   └── qc.py              # MAD-based outlier detection
        │
        ├── features/               # Feature engineering
        │   ├── __init__.py
        │   └── grouping.py        # Group neurons and aggregate features
        │
        ├── models/                 # Model training and prediction
        │   ├── __init__.py
        │   ├── trainer.py         # Train TabPFN models
        │   └── predictor.py       # Load models and predict
        │
        ├── screening/              # Drug screening
        │   ├── __init__.py
        │   └── drug_screening.py  # Screen drugs and rank by P(Control)
        │
        ├── visualization/          # Plotting and reports
        │   ├── __init__.py
        │   ├── plots.py           # All plotting functions
        │   └── reports.py         # Text report generation
        │
        └── utils/                  # Helper utilities
            ├── __init__.py
            ├── helpers.py         # Slug function
            └── io.py              # File I/O, model save/load
```

## Module Descriptions

### `data/`
- **loader.py**: Loads neurite CSV files and design files, extracts plate numbers
- **preprocessing.py**: Merges design metadata with features, normalizes columns, handles NaN

### `quality_control/`
- **qc.py**: Applies MAD (Median Absolute Deviation) filtering to remove outlier wells

### `features/`
- **grouping.py**: Groups neurons within wells and aggregates features (mean/median)

### `models/`
- **trainer.py**: Trains TabPFN classifier, evaluates performance, saves models
- **predictor.py**: Loads trained models and makes predictions

### `screening/`
- **drug_screening.py**: Screens drugs and ranks by mean P(Control)

### `visualization/`
- **plots.py**: LDA projection, ROC curves, probability densities, confusion matrices, drug visualizations
- **reports.py**: Generates text summary reports for each drug

### `utils/`
- **helpers.py**: Utility functions (e.g., slug for filenames)
- **io.py**: Save/load JSON, models, feature lists

## Scripts

### `train.py`
Main training pipeline:
1. Load training data (multiple plates)
2. Apply QC
3. Create grouped profiles for each group size
4. Train TabPFN models
5. Screen drugs
6. Generate visualizations
7. Save models and results

### `validate.py`
Validation pipeline:
1. Load validation data (single plate)
2. Apply QC
3. Create grouped profiles
4. Load trained models
5. Predict on validation data
6. Screen drugs
7. Generate visualizations
8. Compare performance

## Key Files Generated

### Training Outputs
- `tabpfn_model_group{size}.joblib` - Trained TabPFN models
- `selected_features_group{size}.json` - Feature lists used by each model
- `training_results_group{size}.json` - Training metrics (AUC, accuracy, etc.)
- `drug_rankings_group{size}.csv` - Ranked drugs by P(Control)
- `classifier_projection_group{size}.png` - LDA visualization
- `roc_group{size}.png` - ROC curve
- `group_size_comparison.csv` - Summary across all group sizes
- `drug_visualizations_group{size}/` - Per-drug visualizations
  - `{drug}__lda_projection.png`
  - `{drug}__prob_density_log.png`
  - `{drug}__prob_density.png`
  - `{drug}__confusion.png`
  - `{drug}__summary.txt`

### Validation Outputs
- `drug_rankings_validation_group{size}.csv` - Drug rankings on validation data
- `classifier_projection_validation_group{size}.png` - LDA projection
- `roc_validation_group{size}.png` - ROC curve
- `validation_summary.csv` - Performance summary
- `drug_visualizations_group{size}/` - Per-drug visualizations (same format as training)

## Usage Flow

```
┌─────────────────┐
│  Load Data      │ ← data/loader.py, preprocessing.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Quality Control│ ← quality_control/qc.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Group Neurons  │ ← features/grouping.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train TabPFN   │ ← models/trainer.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Screen Drugs   │ ← screening/drug_screening.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualize      │ ← visualization/plots.py, reports.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Results   │ ← utils/io.py
└─────────────────┘
```



