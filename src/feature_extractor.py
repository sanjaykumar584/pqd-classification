"""
Feature extraction for Power Quality Disturbance signals.

Extracts three categories of features from raw waveform signals:
  1. Time-domain / Statistical features (14)
  2. Frequency-domain / FFT features (10)
  3. Wavelet-domain / DWT features (12)
  Total: 36 features per signal
"""

import numpy as np
from scipy.stats import skew, kurtosis
import pywt


# ── Signal parameters ────────────────────────────────────────────────────────

SAMPLING_RATE = 5000   # Hz
SIGNAL_LENGTH = 100    # samples


# ── Time-domain features ─────────────────────────────────────────────────────

def extract_time_features(signal):
    """Extract 14 time-domain / statistical features from a signal.

    Parameters
    ----------
    signal : np.ndarray, shape (100,)

    Returns
    -------
    dict : feature_name -> value
    """
    f = {}
    f['mean'] = np.mean(signal)
    f['std'] = np.std(signal)
    f['rms'] = np.sqrt(np.mean(signal ** 2))
    f['peak'] = np.max(np.abs(signal))
    f['crest_factor'] = f['peak'] / (f['rms'] + 1e-10)
    f['skewness'] = float(skew(signal))
    f['kurtosis'] = float(kurtosis(signal))
    f['zero_crossing_rate'] = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
    f['peak_to_peak'] = np.max(signal) - np.min(signal)
    mean_abs = np.mean(np.abs(signal))
    f['form_factor'] = f['rms'] / (mean_abs + 1e-10)
    f['energy'] = np.sum(signal ** 2)
    f['waveform_length'] = np.sum(np.abs(np.diff(signal)))
    f['iqr'] = float(np.percentile(signal, 75) - np.percentile(signal, 25))

    # Shannon entropy on histogram bins
    hist, _ = np.histogram(signal, bins=20, density=True)
    hist = hist[hist > 0]
    bin_width = (np.max(signal) - np.min(signal)) / 20
    if bin_width > 0:
        probs = hist * bin_width
        probs = probs[probs > 0]
        f['entropy'] = float(-np.sum(probs * np.log2(probs + 1e-10)))
    else:
        f['entropy'] = 0.0

    return f


# ── Frequency-domain features (FFT) ─────────────────────────────────────────

def extract_fft_features(signal, fs=SAMPLING_RATE):
    """Extract 10 frequency-domain features using FFT.

    With 100 samples at 5 kHz: frequency resolution = 50 Hz.
    Bin 0 = DC, Bin 1 = 50 Hz (fundamental), Bin 3 = 150 Hz (3rd harmonic), etc.

    Parameters
    ----------
    signal : np.ndarray, shape (100,)
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    dict : feature_name -> value
    """
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    f = {}
    f['fundamental_mag'] = fft_mag[1]               # 50 Hz
    f['harmonic_3rd'] = fft_mag[3] if len(fft_mag) > 3 else 0.0   # 150 Hz
    f['harmonic_5th'] = fft_mag[5] if len(fft_mag) > 5 else 0.0   # 250 Hz
    f['harmonic_7th'] = fft_mag[7] if len(fft_mag) > 7 else 0.0   # 350 Hz

    # Total Harmonic Distortion
    fundamental = fft_mag[1] + 1e-10
    harmonic_energy = np.sqrt(np.sum(fft_mag[2:] ** 2))
    f['thd'] = harmonic_energy / fundamental

    # Spectral centroid and spread
    mag_sum = np.sum(fft_mag) + 1e-10
    f['spectral_centroid'] = np.sum(freqs * fft_mag) / mag_sum
    f['spectral_spread'] = np.sqrt(
        np.sum(((freqs - f['spectral_centroid']) ** 2) * fft_mag) / mag_sum
    )

    # Spectral energy
    f['spectral_energy'] = np.sum(fft_mag ** 2)

    # Dominant frequency (excluding DC)
    f['dominant_freq'] = freqs[np.argmax(fft_mag[1:]) + 1]

    # High-frequency energy ratio (above 500 Hz)
    total_energy = np.sum(fft_mag ** 2) + 1e-10
    hf_mask = freqs > 500
    f['hf_energy_ratio'] = np.sum(fft_mag[hf_mask] ** 2) / total_energy

    return f


# ── Wavelet-domain features (DWT) ───────────────────────────────────────────

def extract_wavelet_features(signal, wavelet='db4', level=3):
    """Extract 12 wavelet features using Discrete Wavelet Transform.

    3-level DWT with Daubechies-4 wavelet produces 4 coefficient sets:
      cA3: 0–312.5 Hz    (fundamental, sag, swell, flicker, low harmonics)
      cD3: 312.5–625 Hz  (higher harmonics)
      cD2: 625–1250 Hz   (oscillatory transients)
      cD1: 1250–2500 Hz  (impulsive transients, notches)

    For each: energy, standard deviation, entropy → 4 x 3 = 12 features.

    Note: level=3 is the maximum safe level for 100 samples with db4
    (pywt.dwt_max_level(100, 'db4') == 3).

    Parameters
    ----------
    signal : np.ndarray, shape (100,)
    wavelet : str
        Wavelet family.
    level : int
        Decomposition levels.

    Returns
    -------
    dict : feature_name -> value
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs = [cA3, cD3, cD2, cD1]
    level_names = ['cA3', 'cD3', 'cD2', 'cD1']

    f = {}
    for name, c in zip(level_names, coeffs):
        c = np.array(c, dtype=np.float64)
        f[f'{name}_energy'] = np.sum(c ** 2)
        f[f'{name}_std'] = np.std(c)
        # Log-energy entropy
        c_sq = c ** 2 + 1e-10
        f[f'{name}_entropy'] = float(-np.sum(c_sq * np.log(c_sq)))

    return f


# ── Combined feature extraction ─────────────────────────────────────────────

def extract_all_features(signal):
    """Extract all 36 features from a single signal.

    Parameters
    ----------
    signal : np.ndarray, shape (100,)

    Returns
    -------
    dict : feature_name -> value (36 entries)
    """
    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_fft_features(signal))
    features.update(extract_wavelet_features(signal))
    return features


def extract_features_batch(signals, labels=None, verbose=True):
    """Extract features from multiple signals.

    Parameters
    ----------
    signals : np.ndarray, shape (n_samples, 100)
    labels : np.ndarray or None, shape (n_samples,)
    verbose : bool
        Print progress every 1000 signals.

    Returns
    -------
    pd.DataFrame
        Columns = feature names (+ 'label' if labels provided).
    """
    import pandas as pd

    n = signals.shape[0]
    all_features = []

    for i in range(n):
        feats = extract_all_features(signals[i])
        all_features.append(feats)
        if verbose and (i + 1) % 1000 == 0:
            print(f'  Extracted features for {i + 1}/{n} signals')

    df = pd.DataFrame(all_features)

    if labels is not None:
        df['label'] = labels

    if verbose:
        print(f'  Done. Feature matrix shape: {df.shape}')

    return df


# ── Feature name helpers ─────────────────────────────────────────────────────

TIME_FEATURE_NAMES = [
    'mean', 'std', 'rms', 'peak', 'crest_factor', 'skewness', 'kurtosis',
    'zero_crossing_rate', 'peak_to_peak', 'form_factor', 'energy',
    'waveform_length', 'iqr', 'entropy'
]

FFT_FEATURE_NAMES = [
    'fundamental_mag', 'harmonic_3rd', 'harmonic_5th', 'harmonic_7th',
    'thd', 'spectral_centroid', 'spectral_spread', 'spectral_energy',
    'dominant_freq', 'hf_energy_ratio'
]

WAVELET_FEATURE_NAMES = [
    f'{name}_{stat}'
    for name in ['cA3', 'cD3', 'cD2', 'cD1']
    for stat in ['energy', 'std', 'entropy']
]

ALL_FEATURE_NAMES = TIME_FEATURE_NAMES + FFT_FEATURE_NAMES + WAVELET_FEATURE_NAMES


def get_feature_domain(feature_name):
    """Return which domain a feature belongs to: 'time', 'fft', or 'wavelet'."""
    if feature_name in TIME_FEATURE_NAMES:
        return 'time'
    elif feature_name in FFT_FEATURE_NAMES:
        return 'fft'
    elif feature_name in WAVELET_FEATURE_NAMES:
        return 'wavelet'
    return 'unknown'
