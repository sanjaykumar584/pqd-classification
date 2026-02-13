# 3. How the Project Works - Complete Flow

## High-Level Overview

```
TRAINING PHASE (run once)                    PREDICTION PHASE (run anytime)
=========================                    ============================

  17,000 raw signals                           New raw signal
        |                                            |
        v                                            v
  Extract 36 features                          Extract 36 features
  from each signal                             (same process)
        |                                            |
        v                                            v
  Train Random Forest                          Load saved model
  classifier                                         |
        |                                            v
        v                                      Predict class
  Save trained model                           + Normal/Abnormal
  (.pkl file)                                        |
                                                     v
                                               Output result
```

## Detailed Flow: Training Phase

This is what happens when you run notebooks 01 through 04.

### Step 1: Data Loading (Notebook 01)

```
dataset/XPQRS/Flicker.csv           ─┐
dataset/XPQRS/Harmonics.csv          │
dataset/XPQRS/Sag.csv                ├──> load_xpqrs() ──> signals (17000, 100)
dataset/XPQRS/Pure_Sinusoidal.csv    │                      labels  (17000,)
...17 CSV files                      ─┘
```

- Reads all 17 CSV files alphabetically
- Stacks them into one big numpy array: 17,000 rows x 100 columns
- Creates matching labels array: 17,000 class name strings

**File:** `src/data_loader.py` -> `load_xpqrs()`

### Step 2: Signal Visualization (Notebook 02)

```
signals ──> Plot waveform gallery (all 17 classes)
        ──> Plot disturbance vs pure sinusoidal comparison
        ──> Plot amplitude distributions
        ──> Plot FFT frequency spectra
```

- No data transformation, purely visual exploration
- Saves plots to `results/figures/`
- Helps understand what each disturbance looks like

**File:** `src/visualization.py`

### Step 3: Feature Extraction (Notebook 03)

```
signals (17000, 100)
        |
        v
  For each of 17,000 signals:
        |
        ├──> extract_time_features()    ──> 14 numbers
        ├──> extract_fft_features()     ──> 10 numbers
        └──> extract_wavelet_features() ──> 12 numbers
        |
        v
  Feature matrix (17000, 36) + labels
        |
        v
  Save to results/tables/xpqrs_features.csv
```

- Converts raw waveforms into meaningful numerical features
- Also generates PCA visualization and feature importance preview
- Saves the feature CSV so notebook 04 can load it directly

**File:** `src/feature_extractor.py` -> `extract_features_batch()`

### Step 4: Model Training (Notebook 04)

```
xpqrs_features.csv (17000, 36)
        |
        v
  Split: 80% train (13600) / 20% test (3400)
        |
        v
  Build Pipeline:
        |
        ├──> StandardScaler (normalize features to mean=0, std=1)
        └──> RandomForestClassifier (100 trees)
        |
        v
  5-Fold Cross-Validation on training set
        |
        v
  Train on full training set
        |
        v
  Evaluate on test set ──> Accuracy, F1, Precision, Recall
        |
        v
  Save pipeline to results/models/xpqrs_random_forest.pkl
  Save results to results/tables/xpqrs_model_results.csv
  Save confusion matrix to results/figures/
```

**File:** Notebook `04_model_training_evaluation.ipynb`

### Step 5: Results Analysis (Notebook 05)

```
  Load saved models + results CSVs
        |
        ├──> Cross-dataset comparison (XPQRS vs PQ Disturbances)
        ├──> Per-class F1 score heatmap
        ├──> Feature importance analysis by domain
        ├──> t-SNE visualization of feature space
        ├──> Misclassification analysis (which classes get confused)
        └──> ROC curves
```

**File:** Notebook `05_results_comparison.ipynb`

## Detailed Flow: Prediction Phase

This is what happens when you use `predict_signal()`.

```
  New signal: np.array of 100 values
        |
        v
  ┌─ VALIDATE ──────────────────────────────┐
  │  Check shape == (100,)                   │
  │  Convert to float64                      │
  └──────────────────────────────────────────┘
        |
        v
  ┌─ EXTRACT FEATURES ──────────────────────┐
  │  extract_all_features(signal) -> dict    │
  │    14 time + 10 FFT + 12 wavelet = 36   │
  └──────────────────────────────────────────┘
        |
        v
  ┌─ FORMAT ─────────────────────────────────┐
  │  dict -> numpy array shape (1, 36)       │
  │  in correct column order                 │
  │  Replace any inf/nan with 0              │
  └──────────────────────────────────────────┘
        |
        v
  ┌─ LOAD MODEL (once, then cached) ────────┐
  │  joblib.load('xpqrs_random_forest.pkl')  │
  │  Pipeline: StandardScaler + RandomForest │
  └──────────────────────────────────────────┘
        |
        v
  ┌─ PREDICT ────────────────────────────────┐
  │  Scaler normalizes the 36 features       │
  │  RandomForest predicts class (0-16)      │
  │  RandomForest gives probabilities        │
  └──────────────────────────────────────────┘
        |
        v
  ┌─ MAP OUTPUT ─────────────────────────────┐
  │  Numeric label -> class name string      │
  │  Check if Pure_Sinusoidal -> "Normal"    │
  │  Otherwise -> "Abnormal"                 │
  └──────────────────────────────────────────┘
        |
        v
  Result: {
      status: "Normal" or "Abnormal",
      disturbance_type: "Sag",
      confidence: 0.98,
      all_probabilities: {class: prob, ...}
  }
```

**File:** `src/predictor.py` -> `predict_signal()`

## File Map

```
final-year-project/
│
├── src/                          # Python modules
│   ├── data_loader.py            # Loads CSV/Excel datasets
│   ├── feature_extractor.py      # Extracts 36 features from signals
│   ├── visualization.py          # Plotting utilities
│   └── predictor.py              # Prediction on new signals
│
├── notebooks/                    # Jupyter notebooks (run in order)
│   ├── 01_data_loading_exploration.ipynb
│   ├── 02_signal_visualization.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training_evaluation.ipynb
│   └── 05_results_comparison.ipynb
│
├── dataset/                      # Input data
│   ├── XPQRS/                    # 17 CSV files (raw signals)
│   └── PQ Disturbances Dataset/  # 13 Excel files (pre-extracted features)
│
├── results/                      # All outputs
│   ├── figures/                  # 26 PNG plots
│   ├── models/                   # Trained model .pkl files
│   └── tables/                   # CSV result tables
│
├── docs/                         # Documentation (you are here)
├── requirements.txt              # Python dependencies
└── venv/                         # Virtual environment
```
