# Power Quality Disturbance (PQD) Classification

A machine learning pipeline for automated detection and classification of power quality disturbances from raw waveform signals. The system extracts multi-domain features (time, frequency, wavelet) and trains several classifiers — with **Gradient Boosting** as the primary deployed model — achieving **~91.1% accuracy** across 17 disturbance classes.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Disturbance Classes](#disturbance-classes)
5. [Feature Extraction](#feature-extraction)
6. [Model Performance](#model-performance)
7. [Installation](#installation)
8. [Quickstart](#quickstart)
9. [Python API](#python-api)
10. [Notebooks Guide](#notebooks-guide)
11. [Dataset](#dataset)
12. [Results & Artifacts](#results--artifacts)
13. [Documentation](#documentation)

---

## Overview

Power quality disturbances — voltage sags, swells, harmonics, transients, flicker, interruptions, and their combinations — cause equipment damage and financial losses in electrical systems. Manual identification is slow and error-prone.

This project automates disturbance classification with a fully reproducible ML pipeline:

```
Raw waveform signal (100 samples, one 20 ms cycle at 5 kHz)
        │
        ▼
  Feature extraction (36 features across 3 domains)
        │
        ▼
  Gradient Boosting classifier (trained on 17,000 labeled signals)
        │
        ▼
  Predicted class + Normal/Abnormal status + confidence score
```

---

## Features

- **17-class disturbance classification** including compound disturbances (e.g., Harmonics with Sag)
- **36-feature multi-domain extraction**: time-domain, FFT frequency-domain, and DWT wavelet-domain
- **Six ML models** trained and benchmarked side-by-side
- **Trained model artifacts** saved as `.pkl` pipelines (StandardScaler + classifier) ready for inference
- **Interactive Jupyter notebooks** guiding the full workflow from raw data to live demo
- **Rich visualizations**: waveform galleries, confusion matrices, ROC curves, PCA/t-SNE, feature importance

---

## Project Structure

```
pqd-classification/
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── XPQRS/                        # Primary dataset (17 CSV files, 17,000 signals)
│   │   ├── Pure_Sinusoidal.csv
│   │   ├── Sag.csv
│   │   ├── Swell.csv
│   │   └── ...                       # (17 files total)
│   ├── PQ Disturbances Dataset/      # Secondary dataset (pre-extracted wavelet features)
│   ├── Power_Quality_Data.csv
│   ├── Power_Quality_Dataset.csv
│   └── PQD_Dataset.mat
│
├── src/
│   ├── data_loader.py                # Load XPQRS CSVs → NumPy arrays
│   ├── feature_extractor.py          # Extract 36 features per signal
│   ├── predictor.py                  # Load model, predict single / batch signals
│   └── visualization.py             # Plotting utilities
│
├── notebooks/
│   ├── 01_data_loading_exploration.ipynb
│   ├── 02_signal_visualization.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_model_training_evaluation.ipynb
│   ├── 05_results_comparison.ipynb
│   └── 06_live_demo.ipynb
│
├── results/
│   ├── models/                       # Trained .pkl pipelines
│   │   ├── xpqrs_gradient_boosting.pkl  # Deployed model
│   │   ├── xpqrs_random_forest.pkl
│   │   └── ...
│   ├── figures/                      # Generated plots (PNG)
│   └── tables/                       # CSV result tables
│
└── docs/                             # Detailed written documentation
    ├── 01_input_signal_explained.md
    ├── 02_feature_extraction.md
    ├── 03_project_flow.md
    ├── 04_model_training.md
    ├── 05_prediction_and_output.md
    └── 06_live_demo_explained.md
```

---

## Disturbance Classes

The model classifies signals into **17 classes** — 1 normal and 16 disturbance types:

| # | Class | Type |
|---|-------|------|
| 1 | `Pure_Sinusoidal` | Normal reference |
| 2 | `Sag` | Voltage sag (short-term undervoltage) |
| 3 | `Swell` | Voltage swell (short-term overvoltage) |
| 4 | `Interruption` | Complete loss of voltage |
| 5 | `Transient` | Impulsive transient spike |
| 6 | `Oscillatory_Transient` | Oscillatory transient burst |
| 7 | `Harmonics` | Harmonic distortion |
| 8 | `Notch` | Voltage notching |
| 9 | `Flicker` | Voltage flicker |
| 10 | `Harmonics_with_Sag` | Compound: harmonics + sag |
| 11 | `Harmonics_with_Swell` | Compound: harmonics + swell |
| 12 | `Sag_with_Harmonics` | Compound: sag + harmonics |
| 13 | `Swell_with_Harmonics` | Compound: swell + harmonics |
| 14 | `Flicker_with_Sag` | Compound: flicker + sag |
| 15 | `Flicker_with_Swell` | Compound: flicker + swell |
| 16 | `Sag_with_Oscillatory_Transient` | Compound: sag + oscillatory transient |
| 17 | `Swell_with_Oscillatory_Transient` | Compound: swell + oscillatory transient |

Each signal is also tagged as **Normal** (`Pure_Sinusoidal`) or **Abnormal** (any other class).

---

## Feature Extraction

Each raw waveform (100 samples) is transformed into a **36-element feature vector** across three domains:

### Time-Domain Features (14)

| Feature | Description |
|---------|-------------|
| `mean` | Signal mean |
| `std` | Standard deviation |
| `rms` | Root mean square |
| `peak` | Maximum absolute amplitude |
| `crest_factor` | Peak / RMS |
| `skewness` | Third statistical moment |
| `kurtosis` | Fourth statistical moment |
| `zero_crossing_rate` | Rate of sign changes |
| `peak_to_peak` | Max − Min amplitude |
| `form_factor` | RMS / Mean absolute |
| `energy` | Sum of squared samples |
| `waveform_length` | Sum of absolute differences |
| `iqr` | Interquartile range (Q75 − Q25) |
| `entropy` | Shannon entropy of amplitude histogram |

### Frequency-Domain / FFT Features (10)

Computed from the magnitude spectrum via `np.fft.rfft`:

| Feature | Description |
|---------|-------------|
| `fft_mean` | Mean of FFT magnitudes |
| `fft_std` | Std of FFT magnitudes |
| `fft_max` | Peak FFT magnitude |
| `fft_dominant_freq` | Frequency bin of peak magnitude |
| `fft_spectral_centroid` | Weighted mean frequency |
| `fft_spectral_spread` | Weighted std of frequency |
| `fft_thd` | Total harmonic distortion (harmonics 2–5) |
| `fft_fundamental_amp` | Amplitude at 50 Hz |
| `fft_harmonic_ratio` | Harmonics energy / total energy |
| `fft_energy` | Total spectral energy |

### Wavelet-Domain / DWT Features (12)

Computed using **db4 wavelet**, 3-level decomposition via PyWavelets:

| Feature | Description |
|---------|-------------|
| `wavelet_cA3_energy` | Approximation (level 3) energy |
| `wavelet_cD1_energy` | Detail level 1 energy |
| `wavelet_cD2_energy` | Detail level 2 energy |
| `wavelet_cD3_energy` | Detail level 3 energy |
| `wavelet_cA3_std` | Approximation std |
| `wavelet_cD1_std` | Detail 1 std |
| `wavelet_cD2_std` | Detail 2 std |
| `wavelet_cD3_std` | Detail 3 std |
| `wavelet_cA3_mean_abs` | Approximation mean absolute |
| `wavelet_cD1_mean_abs` | Detail 1 mean absolute |
| `wavelet_cD2_mean_abs` | Detail 2 mean absolute |
| `wavelet_energy_ratio` | cD1 energy / total wavelet energy |

---

## Model Performance

All models trained on the XPQRS dataset (17,000 signals), evaluated on a held-out 20% test split with 5-fold cross-validation:

| Model | CV Accuracy | Test Accuracy | Test F1 (Macro) |
|-------|-------------|---------------|-----------------|
| **Gradient Boosting** | 90.65 ± 0.63% | **91.12%** | **0.9107** |
| **Random Forest** *(deployed)* | 89.72 ± 0.43% | **90.62%** | **0.9056** |
| Decision Tree | 85.45 ± 0.97% | 86.50% | 0.8646 |
| Logistic Regression | 84.17 ± 0.56% | 85.24% | 0.8489 |
| SVM | 81.90 ± 0.46% | 83.35% | 0.8298 |
| KNN | 79.21 ± 0.77% | 80.06% | 0.7955 |

Gradient Boosting is deployed as the primary model because it achieves the highest test accuracy (91.12%) and F1-score (0.9107) on the XPQRS dataset.

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd pqd-classification

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Signal arrays and math |
| `pandas` | ≥2.0 | Data loading and tables |
| `scipy` | ≥1.10 | Statistics (skew, kurtosis) |
| `scikit-learn` | ≥1.3 | ML models and pipelines |
| `PyWavelets` | ≥1.4 | DWT feature extraction |
| `matplotlib` | ≥3.7 | Plotting |
| `seaborn` | ≥0.12 | Statistical visualizations |
| `joblib` | ≥1.3 | Model serialization |
| `jupyter` | ≥1.0 | Interactive notebooks |
| `openpyxl` | ≥3.1 | Reading Excel datasets |

---

## Quickstart

Run the full pipeline end-to-end by executing the notebooks in order:

```bash
jupyter notebook
```

Then open and run notebooks **01 → 06** sequentially (see [Notebooks Guide](#notebooks-guide)).

To use the trained model directly:

```python
import numpy as np
import sys
sys.path.insert(0, 'src')

from predictor import predict_signal

# Example: load a signal from the XPQRS dataset
import pandas as pd
signal = pd.read_csv('dataset/XPQRS/Sag.csv', header=None).values[0]

result = predict_signal(signal)
print(result['status'])            # "Abnormal"
print(result['disturbance_type']) # "Sag"
print(f"{result['confidence']:.1%}")  # e.g. "94.0%"
```

---

## Python API

### `src/data_loader.py`

```python
from data_loader import load_xpqrs, load_xpqrs_as_dataframe

# Returns NumPy arrays
signals, labels = load_xpqrs('dataset/XPQRS')
# signals.shape → (17000, 100)
# labels.shape  → (17000,)

# Returns a DataFrame with columns s_0 … s_99 + 'label'
df = load_xpqrs_as_dataframe('dataset/XPQRS')
```

### `src/feature_extractor.py`

```python
from feature_extractor import extract_all_features, extract_features_batch, ALL_FEATURE_NAMES

# Single signal → 36-element dict
signal = signals[0]                    # shape (100,)
features = extract_all_features(signal)
print(len(features))                   # 36

# Batch → DataFrame with 36 feature columns
feature_df = extract_features_batch(signals)
# feature_df.shape → (17000, 36)

print(ALL_FEATURE_NAMES)               # list of 36 feature names
```

### `src/predictor.py`

```python
from predictor import predict_signal, predict_batch, load_model

# Single prediction
result = predict_signal(signal)
# result = {
#   'status': 'Abnormal',
#   'disturbance_type': 'Harmonics',
#   'confidence': 0.87,
#   'all_probabilities': {'Flicker': 0.01, 'Harmonics': 0.87, ...}
# }

# Batch prediction
results = predict_batch(signals[:50])
# results → list of 50 dicts

# Load a different model
load_model('results/models/xpqrs_gradient_boosting.pkl')
result = predict_signal(signal)
```

### `src/visualization.py`

```python
from visualization import (
    plot_signal,
    plot_waveform_gallery,
    plot_confusion_matrix,
    plot_feature_importance,
)

plot_signal(signal, label='Sag', sampling_rate=5000)
plot_waveform_gallery(signals, labels, n_per_class=3)
```

---

## Notebooks Guide

| Notebook | Description |
|----------|-------------|
| `01_data_loading_exploration.ipynb` | Load all 17 CSV files, inspect shapes, class distribution, basic statistics |
| `02_signal_visualization.ipynb` | Waveform gallery, disturbance vs. reference comparison, FFT spectrum plots |
| `03_feature_extraction.ipynb` | Extract 36 features, correlation matrix, feature distributions by domain, PCA/t-SNE |
| `04_model_training_evaluation.ipynb` | Train and evaluate all 6 models, confusion matrices, ROC curves, cross-validation |
| `05_results_comparison.ipynb` | Side-by-side model comparison, feature importance, final summary table |
| `06_live_demo.ipynb` | Interactive prediction demo — paste or generate a signal and classify in real time |

Run them in order (01 → 04) to reproduce all results. Notebooks 05 and 06 can be run independently once models are trained.

---

## Dataset

### XPQRS Dataset (Primary)

Located in `dataset/XPQRS/`. Contains **17,000 synthetic waveform signals** — 1,000 per disturbance class.

| Property | Value |
|----------|-------|
| Signal length | 100 samples |
| Sampling rate | 5,000 Hz |
| Fundamental frequency | 50 Hz |
| Duration per signal | 20 ms (1 cycle) |
| Number of classes | 17 |
| Signals per class | 1,000 |
| Total signals | 17,000 |
| Format | CSV (one signal per row, no header) |

### PQ Disturbances Dataset (Secondary)

Located in `dataset/PQ Disturbances Dataset/`. Contains pre-extracted wavelet features in Excel format across 13 disturbance types. Used in the `pq_disturbances_*` models.

---

## Results & Artifacts

### Trained Models (`results/models/`)

| File | Description |
|------|-------------|
| `xpqrs_gradient_boosting.pkl` | **Deployed model** — StandardScaler + Gradient Boosting |
| `xpqrs_random_forest.pkl` | Random Forest model (for comparison) |
| `xpqrs_decision_tree.pkl` | Interpretable baseline |
| `xpqrs_logistic_regression.pkl` | Linear baseline |
| `xpqrs_svm.pkl` | SVM baseline |
| `xpqrs_knn.pkl` | KNN baseline |

All `.pkl` files are scikit-learn `Pipeline` objects loadable with `joblib.load()`.

### Result Tables (`results/tables/`)

| File | Description |
|------|-------------|
| `xpqrs_model_results.csv` | CV accuracy, test accuracy, F1, precision, recall for all models |
| `xpqrs_features.csv` | Extracted feature matrix (17,000 × 36) |
| `pq_model_results.csv` | Results on the secondary PQ Disturbances dataset |
| `pq_features.csv` | Feature matrix for the secondary dataset |
| `final_comparison.csv` | Cross-dataset comparison summary |

### Figures (`results/figures/`)

| File | Description |
|------|-------------|
| `xpqrs_waveform_gallery.png` | All 17 class waveforms |
| `xpqrs_cm_random_forest.png` | Random Forest confusion matrix |
| `xpqrs_cm_gradient_boosting.png` | Gradient Boosting confusion matrix |
| `xpqrs_model_accuracy.png` | Model accuracy comparison bar chart |
| `xpqrs_f1_heatmap.png` | Per-class F1 scores heatmap |
| `feature_importance_rf.png` | Top feature importances (Random Forest) |
| `pca_xpqrs.png` | PCA 2D projection of feature space |
| `tsne_xpqrs.png` | t-SNE 2D projection |
| `roc_curves.png` | One-vs-rest ROC curves for all classes |
| `fft_spectrum_comparison.png` | FFT spectra across disturbance types |

---

## Documentation

Detailed explanations of each project component are in the `docs/` folder:

| File | Topic |
|------|-------|
| [01_input_signal_explained.md](docs/01_input_signal_explained.md) | What a raw PQD signal is and how it is structured |
| [02_feature_extraction.md](docs/02_feature_extraction.md) | In-depth walkthrough of all 36 features |
| [03_project_flow.md](docs/03_project_flow.md) | End-to-end pipeline: training and prediction phases |
| [04_model_training.md](docs/04_model_training.md) | Model selection, hyperparameters, and evaluation strategy |
| [05_prediction_and_output.md](docs/05_prediction_and_output.md) | Understanding prediction output fields |
| [06_live_demo_explained.md](docs/06_live_demo_explained.md) | How the interactive demo notebook works |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Keep new features focused — one PR per concern.
3. Add or update the relevant notebook if you change the pipeline.
4. Ensure `requirements.txt` stays minimal and pinned to minimum versions.

---

## License

This project is for research and educational purposes. Dataset credits go to the original XPQRS dataset authors.
