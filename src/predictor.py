"""
Prediction module for Power Quality Disturbance Classification.

Loads a trained Random Forest model and predicts disturbance type
from raw waveform signals.

Usage:
    from predictor import predict_signal

    signal = np.array([...])  # 100 samples, one cycle at 5 kHz
    result = predict_signal(signal)
    print(result['status'])           # "Normal" or "Abnormal"
    print(result['disturbance_type']) # e.g., "Harmonics"
    print(result['confidence'])       # e.g., 0.87
"""

import os
import numpy as np
import joblib

from feature_extractor import extract_all_features, ALL_FEATURE_NAMES
from data_loader import XPQRS_CLASSES

# Class names in the order LabelEncoder assigned them (sorted alphabetically)
CLASS_NAMES = sorted(XPQRS_CLASSES)

NORMAL_CLASS = 'Pure_Sinusoidal'

# Default model path (relative to this file's directory)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'models', 'xpqrs_random_forest.pkl'
)

# Module-level cache so the model is loaded only once
_pipeline = None


def load_model(model_path=None):
    """Load the trained pipeline from disk.

    Parameters
    ----------
    model_path : str or None
        Path to the .pkl file. Defaults to results/models/xpqrs_random_forest.pkl.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline (StandardScaler + RandomForestClassifier).
    """
    global _pipeline
    path = model_path or _DEFAULT_MODEL_PATH
    _pipeline = joblib.load(path)
    return _pipeline


def predict_signal(signal, model_path=None):
    """Predict the disturbance type for a single raw waveform signal.

    Parameters
    ----------
    signal : np.ndarray, shape (100,)
        Raw waveform signal (one cycle, 5 kHz sampling rate).
    model_path : str or None
        Path to model .pkl file. Uses default if None.

    Returns
    -------
    dict with keys:
        status : str
            "Normal" or "Abnormal".
        disturbance_type : str
            Predicted class name (e.g., "Sag", "Harmonics_with_Swell").
        confidence : float
            Probability of the predicted class (0–1).
        all_probabilities : dict
            {class_name: probability} for all 17 classes.
    """
    global _pipeline
    if _pipeline is None:
        load_model(model_path)

    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape != (100,):
        raise ValueError(f"Expected signal shape (100,), got {signal.shape}")

    # Extract 36 features
    features_dict = extract_all_features(signal)

    # Build feature vector in the correct column order
    feature_vector = np.array([[features_dict[name] for name in ALL_FEATURE_NAMES]])

    # Replace any inf/nan
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict
    predicted_label = int(_pipeline.predict(feature_vector)[0])
    probabilities = _pipeline.predict_proba(feature_vector)[0]

    class_name = CLASS_NAMES[predicted_label]
    confidence = float(probabilities[predicted_label])

    return {
        'status': 'Normal' if class_name == NORMAL_CLASS else 'Abnormal',
        'disturbance_type': class_name,
        'confidence': confidence,
        'all_probabilities': {
            CLASS_NAMES[i]: float(p) for i, p in enumerate(probabilities)
        },
    }


def predict_batch(signals, model_path=None):
    """Predict disturbance types for multiple signals.

    Parameters
    ----------
    signals : np.ndarray, shape (n, 100)
        Array of raw waveform signals.
    model_path : str or None
        Path to model .pkl file. Uses default if None.

    Returns
    -------
    list of dict
        One result dict per signal (same format as predict_signal).
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] != 100:
        raise ValueError(f"Expected shape (n, 100), got {signals.shape}")

    return [predict_signal(signals[i], model_path) for i in range(len(signals))]
