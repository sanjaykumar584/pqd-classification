# 2. How the Input Is Processed (Feature Extraction)

## Why Can't We Feed Raw Signals Directly to the Model?

A raw signal is just 100 numbers — the model has no idea what they mean. Two signals from the same disturbance class can look very different due to random variation, phase shifts, or noise.

**Feature extraction** solves this by converting each 100-point signal into **36 meaningful numbers** that describe the signal's characteristics. These 36 features capture patterns that are consistent within the same class and different across classes.

```
Raw Signal (100 numbers)              Features (36 numbers)
[0.0, 0.31, 0.58, ...]    --->       [rms=0.73, thd=0.02, cA3_energy=98.9, ...]
  (hard to classify)                    (easy to classify)
```

## The Three Feature Domains

Features are extracted from three different perspectives ("domains") of the signal:

### Domain 1: Time-Domain Features (14 features)

These look at the signal's **shape and statistics** directly — no transformation needed.

| Feature | Formula / Meaning | Why It Helps |
|---|---|---|
| **mean** | Average of all 100 values | Normal signals have mean near 0; DC offset indicates problems |
| **std** | Standard deviation | Measures how spread out values are |
| **rms** | Root Mean Square = sqrt(mean(signal^2)) | Effective signal strength; sags have low RMS, swells have high RMS |
| **peak** | Maximum absolute value | Transients cause high peaks |
| **crest_factor** | peak / rms | High for signals with sharp spikes (transients) |
| **skewness** | Asymmetry of the distribution | Symmetrical signals have skewness near 0 |
| **kurtosis** | "Peakedness" of the distribution | High kurtosis = sharp spikes; low = flat signal |
| **zero_crossing_rate** | How often the signal crosses zero | Harmonics change the crossing pattern |
| **peak_to_peak** | max - min | Total voltage swing |
| **form_factor** | rms / mean(abs(signal)) | Shape indicator; pure sine = 1.11 |
| **energy** | Sum of squared values | Total signal power; interruptions have very low energy |
| **waveform_length** | Sum of absolute differences between consecutive points | Rough/noisy signals have high waveform length |
| **iqr** | 75th percentile - 25th percentile | Robust spread measure |
| **entropy** | Shannon entropy of histogram | Measures randomness/unpredictability of the signal |

**Example:** A voltage sag has lower `rms` and `energy` than a normal signal because the voltage drops. A transient has a high `peak` and `crest_factor` because of the sharp spike.

### Domain 2: Frequency-Domain Features (10 features via FFT)

The **Fast Fourier Transform (FFT)** converts the signal from time to frequency — it shows what frequencies are present and how strong they are.

```
Time Domain:  "The signal looks like THIS shape"
     |
     v  (FFT)
Frequency Domain:  "The signal contains 50 Hz (strong) + 150 Hz (weak) + 250 Hz (weak)"
```

| Feature | What It Measures | Why It Helps |
|---|---|---|
| **fundamental_mag** | Strength of the 50 Hz component | Core power signal; weaker during sags/interruptions |
| **harmonic_3rd** | Strength at 150 Hz (3rd harmonic) | Non-zero for harmonic distortion |
| **harmonic_5th** | Strength at 250 Hz (5th harmonic) | Non-zero for harmonic distortion |
| **harmonic_7th** | Strength at 350 Hz (7th harmonic) | Non-zero for harmonic distortion |
| **thd** | Total Harmonic Distortion = harmonic_energy / fundamental | Key indicator of harmonic distortion; near 0 for clean signal |
| **spectral_centroid** | "Center of mass" of frequency spectrum | Higher when high-frequency content is present |
| **spectral_spread** | How spread out the frequency content is | Narrow for pure sine, wide for complex signals |
| **spectral_energy** | Total energy in frequency domain | Same as time-domain energy but computed from FFT |
| **dominant_freq** | Frequency with highest magnitude | Should be 50 Hz for normal; different for anomalies |
| **hf_energy_ratio** | Fraction of energy above 500 Hz | High for transients, notches (high-frequency events) |

**Example:** A signal with harmonics will have high `thd`, non-zero `harmonic_3rd`/`harmonic_5th`, and higher `spectral_centroid`. A pure sinusoidal will have `thd` near 0 and `dominant_freq` = 50 Hz.

### Domain 3: Wavelet-Domain Features (12 features via DWT)

The **Discrete Wavelet Transform (DWT)** splits the signal into frequency bands using the **Daubechies-4 (db4)** wavelet at 3 decomposition levels. Unlike FFT, wavelets capture both frequency AND time information.

```
                    Signal (0-2500 Hz)
                    /                \
              cA1 (0-1250 Hz)    cD1 (1250-2500 Hz)  ← Level 1
              /            \
        cA2 (0-625 Hz)   cD2 (625-1250 Hz)           ← Level 2
        /          \
  cA3 (0-312.5 Hz) cD3 (312.5-625 Hz)                ← Level 3
```

Each band captures different phenomena:

| Sub-band | Frequency Range | What It Captures |
|---|---|---|
| **cA3** | 0 - 312.5 Hz | Fundamental (50 Hz), sag, swell, flicker, low harmonics |
| **cD3** | 312.5 - 625 Hz | Higher harmonics |
| **cD2** | 625 - 1250 Hz | Oscillatory transients |
| **cD1** | 1250 - 2500 Hz | Impulsive transients, notches |

For each sub-band, three statistics are computed:

| Statistic | What It Measures |
|---|---|
| **energy** | Total power in that frequency band (sum of squared coefficients) |
| **std** | Variation within the band |
| **entropy** | Complexity/disorder of the coefficients |

This gives 4 bands x 3 statistics = **12 wavelet features**.

**Example:** A transient spike will have high `cD1_energy` (high-frequency energy), while a voltage sag will mainly affect `cA3_energy` (low-frequency energy drops).

## The Complete Feature Vector

After extraction, each signal becomes a row of 36 numbers:

```
[mean, std, rms, peak, crest_factor, skewness, kurtosis,
 zero_crossing_rate, peak_to_peak, form_factor, energy,
 waveform_length, iqr, entropy,                              ← 14 time features
 fundamental_mag, harmonic_3rd, harmonic_5th, harmonic_7th,
 thd, spectral_centroid, spectral_spread, spectral_energy,
 dominant_freq, hf_energy_ratio,                              ← 10 FFT features
 cA3_energy, cA3_std, cA3_entropy,
 cD3_energy, cD3_std, cD3_entropy,
 cD2_energy, cD2_std, cD2_entropy,
 cD1_energy, cD1_std, cD1_entropy]                            ← 12 wavelet features
```

## Code Reference

All feature extraction happens in `src/feature_extractor.py`:

```python
from feature_extractor import extract_all_features

signal = signals[0]  # shape (100,)
features = extract_all_features(signal)

# features = {
#     'mean': -0.0026,
#     'std': 0.7386,
#     'rms': 0.7386,
#     'peak': 1.0749,
#     ...
#     'cD1_entropy': 7.76e-06
# }
# Total: 36 key-value pairs
```

## Why Three Domains?

Each domain sees things the others miss:

| Disturbance | Best Domain | Why |
|---|---|---|
| Sag / Swell | Time-domain | Amplitude changes are directly visible in rms, peak |
| Harmonics | Frequency (FFT) | Harmonic frequencies show up as peaks at 150/250/350 Hz |
| Transient | Wavelet | Short-duration spike needs both time and frequency info |
| Flicker | Time + Wavelet | Slow amplitude modulation appears in low-frequency wavelet bands |
| Notch | Wavelet | Brief cuts are captured by high-frequency wavelet coefficients |

Using all three together gives the model the most complete picture, leading to higher accuracy than any single domain alone.
