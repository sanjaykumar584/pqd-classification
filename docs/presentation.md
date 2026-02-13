# Power Quality Disturbance Detection and Classification Using Machine Learning

---

## 1. What Problem Are We Solving?

### The Problem

Electricity from the power grid is supposed to be a clean, smooth wave at 50 Hz. But in the real world, things go wrong — motors start, lightning strikes, equipment fails, heavy loads switch on and off. These events cause **disturbances** in the electrical signal.

These disturbances can:
- Damage sensitive equipment (computers, medical devices, industrial machines)
- Cause data loss and system crashes
- Reduce the lifespan of electrical components
- Lead to production downtime in factories

### The Current Approach

Today, power engineers manually analyze electrical signals using oscilloscopes or basic threshold-based monitoring systems. This is:
- **Slow** — a human has to look at the waveform and decide what's wrong
- **Error-prone** — some disturbances look very similar to the human eye
- **Not scalable** — can't monitor thousands of signals in real-time

### Our Solution

We built a **machine learning system** that can automatically:
1. **Detect** whether an electrical signal is normal or abnormal
2. **Identify** the exact type of disturbance (out of 17 known types)
3. **Label** it with a confidence score

It does this in **milliseconds**, with **90.6% accuracy** across 17 disturbance types.

---

## 2. What Are Power Quality Disturbances?

A normal power signal is a smooth sine wave. A disturbance is any deviation from this:

```
Normal (Pure Sinusoidal):        Voltage Sag:               Harmonics:
                                 (voltage drops)            (wave gets distorted)
   /\      /\                      /\      /\                 /\    /\
  /  \    /  \                    / \    /  \                /  --\/  --\
 /    \  /    \                  /   ----    \              /            \
------\/------\/---         ----/            \---       ---/              \---
                                  (dip here)               (extra wiggles)
```

### The 17 Types We Detect

**9 Single Disturbances:**

| Type | What Happens | Real-World Cause |
|---|---|---|
| Pure Sinusoidal | Clean, normal signal | No problem |
| Sag | Voltage drops temporarily | Motor starting, heavy load |
| Swell | Voltage rises temporarily | Load disconnection |
| Interruption | Voltage drops to near zero | Circuit breaker trip, fault |
| Transient | Sharp voltage spike | Lightning, switching |
| Oscillatory Transient | Ringing oscillation | Capacitor bank switching |
| Harmonics | Waveform gets distorted | Computers, LED lights, VFDs |
| Flicker | Voltage fluctuates rapidly | Arc furnaces, welding |
| Notch | Small cuts in the waveform | Power converter switching |

**8 Compound Disturbances (two problems at once):**

| Type | Combination |
|---|---|
| Harmonics with Sag | Distorted waveform + voltage dip |
| Harmonics with Swell | Distorted waveform + voltage rise |
| Flicker with Sag | Voltage fluctuation + voltage dip |
| Flicker with Swell | Voltage fluctuation + voltage rise |
| Sag with Harmonics | Voltage dip + distorted waveform |
| Swell with Harmonics | Voltage rise + distorted waveform |
| Sag with Oscillatory Transient | Voltage dip + ringing |
| Swell with Oscillatory Transient | Voltage rise + ringing |

---

## 3. The Dataset

We use the **XPQRS dataset** — a collection of 17,000 simulated power signals.

| Property | Value |
|---|---|
| Total signals | 17,000 |
| Signals per class | 1,000 (perfectly balanced) |
| Number of classes | 17 |
| Samples per signal | 100 |
| Sampling rate | 5,000 Hz |
| Signal duration | 20 ms (one complete cycle at 50 Hz) |
| Amplitude range | Approximately -2.5 to +2.5 (normalized) |

Each signal is a **numpy array of 100 numbers** representing voltage measurements taken over one cycle (20 milliseconds) of the power waveform.

### Why 100 Samples?

- The power grid operates at **50 Hz** (50 cycles per second)
- One cycle = 1/50 = **0.02 seconds = 20 ms**
- At 5,000 samples/second: 5000 x 0.02 = **100 samples per cycle**
- One cycle captures enough information to see most disturbance patterns

---

## 4. How It Works — The Complete Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                         TRAINING (done once)                        │
│                                                                      │
│   Raw Signal         Feature             Trained                     │
│   (100 numbers)  --> Extraction     -->   Random Forest              │
│   x 17,000           (36 features)        Classifier                 │
│                                           (saved to disk)            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      PREDICTION (runs anytime)                       │
│                                                                      │
│   New Signal         Feature         Load Model        Result        │
│   (100 numbers) -->  Extraction  --> & Predict    -->  Normal /      │
│                      (36 features)                     Abnormal +    │
│                                                        Type + Score  │
└──────────────────────────────────────────────────────────────────────┘
```

### Step 1: Feature Extraction

We can't feed raw signal values directly to a classifier — the model needs **meaningful characteristics** that describe what the signal looks like. We extract **36 features** from three different perspectives:

#### Time-Domain Features (14 features)

These describe the **shape and statistics** of the signal directly.

| Feature | What It Tells Us | Example Use |
|---|---|---|
| RMS (Root Mean Square) | Effective signal strength | Sags have low RMS, swells have high RMS |
| Peak | Maximum absolute voltage | Transients cause unusually high peaks |
| Crest Factor | Peak / RMS ratio | High for signals with sharp spikes |
| Kurtosis | How "spiky" the signal is | Transients have very high kurtosis |
| Zero Crossing Rate | How often signal crosses zero | Harmonics alter the crossing pattern |
| Energy | Total power in the signal | Interruptions have near-zero energy |
| Entropy | How unpredictable the signal is | Noisy/disturbed signals have higher entropy |
| + 7 more | std, mean, skewness, peak-to-peak, form factor, waveform length, IQR | |

#### Frequency-Domain Features (10 features via FFT)

The **Fast Fourier Transform** converts the signal into its frequency components — it answers "what frequencies are present and how strong are they?"

| Feature | What It Tells Us | Example Use |
|---|---|---|
| Fundamental Magnitude | Strength of the 50 Hz component | Weaker during sags and interruptions |
| 3rd/5th/7th Harmonic | Strength at 150/250/350 Hz | Non-zero when harmonics are present |
| THD (Total Harmonic Distortion) | Ratio of harmonic energy to fundamental | Key indicator — near 0 for clean signal, high for harmonics |
| Spectral Centroid | "Center of gravity" of frequency content | Shifts when high-frequency content is added |
| Dominant Frequency | Strongest frequency component | Should be 50 Hz for normal signals |
| High-Frequency Energy Ratio | Energy above 500 Hz / total energy | High for transients and notches |
| + 2 more | spectral spread, spectral energy | |

#### Wavelet-Domain Features (12 features via DWT)

The **Discrete Wavelet Transform** splits the signal into frequency bands while preserving time information — it can tell us *when* and *at what frequency* something happened.

We use the **Daubechies-4 wavelet** with 3 decomposition levels:

```
Signal (0 - 2500 Hz)
├── cA3: 0 - 312.5 Hz      → Captures: fundamental, sag, swell, flicker
├── cD3: 312.5 - 625 Hz    → Captures: higher harmonics
├── cD2: 625 - 1250 Hz     → Captures: oscillatory transients
└── cD1: 1250 - 2500 Hz    → Captures: impulsive transients, notches
```

For each band, we compute: **energy**, **standard deviation**, and **entropy** = 4 bands x 3 stats = 12 features.

#### Why Three Domains?

Each domain catches things the others miss:

| Disturbance | Best Detected By | Why |
|---|---|---|
| Sag / Swell | Time-domain (RMS, peak) | Direct amplitude change |
| Harmonics | FFT (THD, harmonic magnitudes) | Extra frequency components |
| Transients | Wavelet (cD1 energy) | Short-duration, high-frequency events |
| Flicker | Time + Wavelet | Slow amplitude modulation |

Using all three together gives **higher accuracy** than any single domain alone.

### Step 2: Model Training (Random Forest)

After feature extraction, each signal becomes a row of **36 numbers**. The Random Forest classifier learns to map these 36 numbers to one of the 17 classes.

#### What Is Random Forest?

A Random Forest is an **ensemble of 100 decision trees** that vote together:

```
Signal's 36 features
     |
     ├── Tree 1:  "I think this is Sag"
     ├── Tree 2:  "I think this is Sag"
     ├── Tree 3:  "I think this is Flicker_with_Sag"
     ├── ...
     └── Tree 100: "I think this is Sag"

     MAJORITY VOTE → "Sag" (confidence: 98/100 = 0.98)
```

Each tree is slightly different because it:
- Trains on a random subset of the data
- Considers a random subset of features at each decision point

This randomness prevents **overfitting** (memorizing the training data instead of learning patterns) and makes the predictions more robust.

#### How One Tree Decides

Each tree is a series of yes/no questions about the features:

```
Is peak > 1.2?
├── YES → Is THD > 0.15?
│         ├── YES → Harmonics
│         └── NO  → Swell
└── NO  → Is RMS < 0.3?
          ├── YES → Interruption
          └── NO  → Is fundamental_mag > 0.45?
                    ├── YES → Pure Sinusoidal
                    └── NO  → Sag
```

#### Training Process

```
17,000 signals (36 features each)
         |
         ├── 80% (13,600) → Training set (model learns from these)
         └── 20% (3,400)  → Test set (model is evaluated, never seen during training)
```

1. **StandardScaler** normalizes all 36 features to the same scale (mean=0, std=1)
2. **5-fold cross-validation** checks the model is stable across different data splits
3. **Final training** on the full training set
4. **Evaluation** on the held-out test set

### Step 3: Prediction

When a new signal arrives:

```
New signal (100 values)
     ↓
Extract 36 features (time + FFT + wavelet)
     ↓
Normalize features (using saved scaler)
     ↓
100 trees vote → predicted class + confidence
     ↓
Output:
  • Status:     "Normal" or "Abnormal"
  • Type:       "Sag" (specific disturbance name)
  • Confidence: 0.98 (98% sure)
```

---

## 5. Results

### Accuracy on XPQRS Dataset (17 classes, 17,000 signals)

| Metric | Value | What It Means |
|---|---|---|
| **Test Accuracy** | **90.62%** | 9 out of 10 signals classified correctly |
| **Test F1 Score** | **90.56%** | Balanced performance across all 17 classes |
| **Test Precision** | **90.67%** | When it says "Sag", it's right ~91% of the time |
| **Test Recall** | **90.62%** | It catches ~91% of actual Sag signals |
| **Cross-Validation** | **89.72% +/- 0.43%** | Consistent across 5 different data splits |

### What the Model Gets Right

- **Pure Sinusoidal (Normal)** — near 100% accuracy
- **Interruption** — near 100% (very distinct: voltage drops to zero)
- **Harmonics** — near 100% (clear frequency pattern at 150/250/350 Hz)
- **Transient** — ~99% (distinctive sharp spikes)
- **Sag, Swell** — ~95%+ (clear amplitude changes)

### What the Model Finds Harder

Some **compound disturbances** share very similar features:
- **Harmonics_with_Sag** vs **Sag_with_Harmonics** — both have harmonics + sag, just differ in which is dominant
- **Flicker_with_Sag** vs **Sag** — flicker can be subtle when combined with a sag

This is expected — even human experts can find these pairs difficult to distinguish.

### Confidence Scores

| Confidence Level | Meaning |
|---|---|
| 0.90 - 1.00 | Very confident — prediction is almost certainly correct |
| 0.70 - 0.90 | Fairly confident — likely correct but some uncertainty |
| 0.50 - 0.70 | Uncertain — model is unsure between two or more classes |
| Below 0.50 | Low confidence — signal may not match any learned pattern well |

---

## 6. How to Use the System

### Running a Prediction

```python
import sys
sys.path.insert(0, 'src')
from predictor import predict_signal
from data_loader import load_xpqrs
import numpy as np

# Load a signal
signals, labels = load_xpqrs('dataset/XPQRS/')
signal = signals[10500]  # a Sag signal

# Predict
result = predict_signal(signal)

print(result['status'])           # "Abnormal"
print(result['disturbance_type']) # "Sag"
print(result['confidence'])       # 0.98
```

### Output Explained

```python
result = {
    'status': 'Abnormal',           # Is there a problem? Yes
    'disturbance_type': 'Sag',      # What kind? Voltage sag
    'confidence': 0.98,             # How sure? 98%
    'all_probabilities': {          # Probability for every class
        'Pure_Sinusoidal': 0.00,
        'Sag': 0.98,
        'Sag_with_Harmonics': 0.01,
        'Flicker_with_Sag': 0.01,
        ...
    }
}
```

---

## 7. Project Structure

```
final-year-project/
├── src/                              # Core Python modules
│   ├── data_loader.py                # Loads datasets
│   ├── feature_extractor.py          # Extracts 36 features from signals
│   ├── visualization.py              # Plotting utilities
│   └── predictor.py                  # Prediction on new signals
│
├── notebooks/                        # Step-by-step pipeline
│   ├── 01_data_loading_exploration   # Load and explore datasets
│   ├── 02_signal_visualization       # Visualize waveforms
│   ├── 03_feature_extraction         # Extract 36 features
│   ├── 04_model_training_evaluation  # Train Random Forest, evaluate
│   └── 05_results_comparison         # Detailed analysis and plots
│
├── dataset/XPQRS/                    # 17 CSV files, 17000 signals
├── results/
│   ├── figures/                      # All generated plots
│   ├── models/                       # Trained model (.pkl)
│   └── tables/                       # Result CSVs
│
├── docs/                             # Documentation
└── requirements.txt                  # Python dependencies
```

## 8. Tools and Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3** | Programming language |
| **NumPy** | Numerical computation and array handling |
| **Pandas** | Data loading and manipulation |
| **Scikit-learn** | Machine learning (Random Forest, StandardScaler, evaluation metrics) |
| **SciPy** | Statistical functions (skewness, kurtosis) |
| **PyWavelets** | Discrete Wavelet Transform for wavelet features |
| **Matplotlib + Seaborn** | Plots and visualizations |
| **Jupyter Notebook** | Interactive development environment |
| **Joblib** | Saving and loading trained models |

---

## 9. Summary

| Aspect | Detail |
|---|---|
| **Problem** | Automatically detect and classify power quality disturbances |
| **Input** | Raw electrical signal (100 voltage samples, one 50 Hz cycle) |
| **Feature Extraction** | 36 features from 3 domains (time, frequency, wavelet) |
| **Classifier** | Random Forest (100 decision trees) |
| **Output** | Normal/Abnormal + disturbance type + confidence score |
| **Accuracy** | 90.62% across 17 disturbance types |
| **Dataset** | XPQRS — 17,000 signals, 17 classes, balanced |
| **Key Advantage** | Multi-domain feature extraction captures disturbances that single-domain approaches miss |
