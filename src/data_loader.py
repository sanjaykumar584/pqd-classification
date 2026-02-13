"""
Data loading utilities for Power Quality Disturbance Classification.

Handles loading from:
  - XPQRS dataset (17 CSV files, raw waveform signals)
  - PQ Disturbances Dataset (13 Excel files, pre-extracted wavelet features)
"""

import os
import numpy as np
import pandas as pd


# ── XPQRS Dataset ────────────────────────────────────────────────────────────

XPQRS_CLASSES = [
    'Pure_Sinusoidal', 'Sag', 'Swell', 'Interruption', 'Transient',
    'Oscillatory_Transient', 'Harmonics', 'Harmonics_with_Sag',
    'Harmonics_with_Swell', 'Flicker', 'Flicker_with_Sag',
    'Flicker_with_Swell', 'Sag_with_Oscillatory_Transient',
    'Swell_with_Oscillatory_Transient', 'Sag_with_Harmonics',
    'Swell_with_Harmonics', 'Notch'
]

# Signal parameters
SAMPLING_RATE = 5000      # Hz
FUNDAMENTAL_FREQ = 50     # Hz
SIGNAL_LENGTH = 100       # samples
SIGNAL_DURATION = 0.02    # seconds (20 ms)


def load_xpqrs(data_dir):
    """Load all 17 XPQRS CSV files into a single DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to the XPQRS directory containing the CSV files.

    Returns
    -------
    signals : np.ndarray, shape (17000, 100)
        Raw waveform signals.
    labels : np.ndarray, shape (17000,)
        String class labels.
    """
    all_signals = []
    all_labels = []

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for csv_file in sorted(csv_files):
        class_name = csv_file.replace('.csv', '')
        df = pd.read_csv(os.path.join(data_dir, csv_file), header=None)
        all_signals.append(df.values.astype(np.float64))
        all_labels.extend([class_name] * len(df))

    signals = np.vstack(all_signals)
    labels = np.array(all_labels)
    return signals, labels


def load_xpqrs_as_dataframe(data_dir):
    """Load XPQRS data as a pandas DataFrame with a 'label' column.

    Parameters
    ----------
    data_dir : str
        Path to the XPQRS directory containing the CSV files.

    Returns
    -------
    pd.DataFrame
        Columns 0-99 are signal samples, column 'label' is the class name.
    """
    signals, labels = load_xpqrs(data_dir)
    df = pd.DataFrame(signals, columns=[f's_{i}' for i in range(signals.shape[1])])
    df['label'] = labels
    return df


# ── PQ Disturbances Dataset ─────────────────────────────────────────────────

PQ_FILE_LABEL_MAP = {
    'Fundamental Signal Dataset.xlsx': 'Fundamental',
    'Sag Dataset.xlsx': 'Sag',
    'Swell Dataset.xlsx': 'Swell',
    'Interruption Dataset.xlsx': 'Interruption',
    'Transients Dataset.xlsx': 'Transient',
    'Harmonics Dataset.xlsx': 'Harmonics',
    'Flicker Dataset.xlsx': 'Flicker',
    'Sag+Harmonics Dataset.xlsx': 'Sag+Harmonics',
    'Swell+Harmonics Dataset.xlsx': 'Swell+Harmonics',
    'Sag+Flicker Dataset.xlsx': 'Sag+Flicker',
    'Swell+Flicker Dataset.xlsx': 'Swell+Flicker',
    'Interruption+Harmonics Dataset.xlsx': 'Interruption+Harmonics',
    'Harmonics+Flicker Dataset.xlsx': 'Harmonics+Flicker',
}

# Feature columns present in each individual Excel file
PQ_WAVELET_LEVELS = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'A8']
PQ_STAT_TYPES = ['Mean', 'Std', 'Rms', 'Range', 'Energy', 'Kurtosis', 'Skewness', 'Entropy']
PQ_FEATURE_NAMES = [f'{stat}-{level}' for stat in PQ_STAT_TYPES for level in PQ_WAVELET_LEVELS]


def load_pq_disturbances(data_dir):
    """Load all individual PQ Disturbances Excel files.

    Each file contains pre-extracted wavelet features (8 statistical measures
    across 9 wavelet sub-bands = 72 features per sample).

    Parameters
    ----------
    data_dir : str
        Path to the 'PQ Disturbances Dataset' directory.

    Returns
    -------
    features : np.ndarray, shape (n_samples, 72)
        Pre-extracted wavelet features.
    labels : np.ndarray, shape (n_samples,)
        String class labels.
    feature_names : list of str
        Names of the 72 feature columns.
    """
    frames = []
    for filename, label in PQ_FILE_LABEL_MAP.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f'Warning: {filepath} not found, skipping.')
            continue
        df = pd.read_excel(filepath, engine='openpyxl')
        # Drop 'Sample' column if present
        if 'Sample' in df.columns:
            df = df.drop(columns=['Sample'])
        df['label'] = label
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in combined.columns if c != 'label']
    features = combined[feature_cols].values.astype(np.float64)
    labels = combined['label'].values
    return features, labels, feature_cols


def load_pq_disturbances_as_dataframe(data_dir):
    """Load PQ Disturbances data as a pandas DataFrame with a 'label' column.

    Parameters
    ----------
    data_dir : str
        Path to the 'PQ Disturbances Dataset' directory.

    Returns
    -------
    pd.DataFrame
        72 feature columns + 'label' column.
    """
    features, labels, feature_cols = load_pq_disturbances(data_dir)
    df = pd.DataFrame(features, columns=feature_cols)
    df['label'] = labels
    return df


# ── Time axis helper ─────────────────────────────────────────────────────────

def get_time_axis():
    """Return the time axis (in seconds) for XPQRS signals."""
    return np.linspace(0, SIGNAL_DURATION, SIGNAL_LENGTH, endpoint=False)


def get_time_axis_ms():
    """Return the time axis (in milliseconds) for XPQRS signals."""
    return get_time_axis() * 1000
