# PPT Content: How the Model Works (3 Slides)

---

## Slide 1: System Design

**Objectives:**
- To prevent damage to sensitive equipment (computers, medical devices, industrial machines) caused by power disturbances
- To replace slow, error-prone manual detection with an automated system that classifies disturbances in milliseconds
- To enable real-time power quality monitoring that can scale to thousands of signals
- To accurately identify the specific type of disturbance so engineers can fix the root cause (e.g., faulty capacitor, overloaded transformer)
- To reduce production downtime and financial losses in industries caused by undetected power quality issues

**Training Phase (done once):**
- Load XPQRS dataset (17,000 signals, 17 classes, 100 samples each)
- Extract 36 features from each signal (time + FFT + wavelet)
- Split 80% train / 20% test
- Normalize features using StandardScaler
- Train Random Forest classifier (100 decision trees)
- Validate with 5-fold cross-validation
- Save trained model to disk (.pkl)

**Prediction Phase (runs anytime):**
- Take new signal (100 voltage values)
- Extract 36 features
- Load saved model and predict
- Output: Normal/Abnormal + disturbance type + confidence score

**Tools Used:** Python, NumPy, Scikit-learn, SciPy, PyWavelets, Matplotlib, Jupyter Notebook

**Project Structure:**

| Component | Purpose |
|---|---|
| `src/feature_extractor.py` | Extracts 36 features from raw signals |
| `src/predictor.py` | Loads model and predicts on new signals |
| `src/data_loader.py` | Loads XPQRS dataset (17 CSV files) |
| `notebooks/` | Step-by-step pipeline (data → features → train → evaluate → demo) |
| `results/models/` | Saved trained model (.pkl) |

---

## Slide 2: How It Works

**Input:** Raw electrical signal (100 voltage samples, 20 ms, one cycle at 50 Hz)

```
Raw Signal (100 values) → Feature Extraction (36 features) → Random Forest (100 trees vote) → Output
```

**Feature Extraction — 36 features from 3 domains:**

| Domain | Features | What It Captures |
|---|---|---|
| Time (14) | RMS, Peak, Kurtosis, Energy, etc. | Signal shape — detects sags, swells, interruptions |
| FFT (10) | THD, Harmonic magnitudes, Dominant freq | Frequency content — detects harmonics |
| Wavelet (12) | Sub-band energies (cA3, cD3, cD2, cD1) | Time-frequency events — detects transients, notches |

**Classification:** Random Forest (100 decision trees) — each tree votes, majority wins. Confidence = % of trees that agreed.

**17 Classes:** Pure Sinusoidal, Sag, Swell, Interruption, Transient, Oscillatory Transient, Harmonics, Flicker, Notch + 8 compound types

---

## Slide 2: Results and Output

| Metric | Value |
|---|---|
| **Test Accuracy** | **90.62%** |
| **F1 Score** | **90.56%** |
| **Cross-Validation** | **89.72% +/- 0.43%** |

**Output for each signal:**

```
{
    status:           "Normal" or "Abnormal",
    disturbance_type: "Sag",
    confidence:       98%
}
```

**Dataset:** 17,000 signals (1,000 per class), 80/20 train-test split, 5-fold cross-validation

**Key Advantage:** Multi-domain features (time + FFT + wavelet) capture disturbances that any single domain would miss
