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

# Practical interpretation map for each predicted disturbance class.
# This is rule-based domain guidance, not a second ML model.
DISTURBANCE_GUIDANCE = {
    'Pure_Sinusoidal': {
        'likely_cause': 'Power supply is stable and close to ideal sinusoidal behavior.',
        'recommended_fix': 'No corrective action needed. Continue routine monitoring.'
    },
    'Sag': {
        'likely_cause': 'Temporary voltage drop, often from motor starting, feeder faults, or weak grid conditions.',
        'recommended_fix': 'Use voltage regulators or DVR/UPS, stagger large motor starts, and check upstream faults.'
    },
    'Swell': {
        'likely_cause': 'Temporary voltage rise, often due to sudden load reduction, capacitor switching, or neutral issues.',
        'recommended_fix': 'Use overvoltage protection, verify capacitor bank control, and inspect neutral/ground connections.'
    },
    'Interruption': {
        'likely_cause': 'Near-total voltage loss, usually caused by breaker trips, protection operation, or line faults.',
        'recommended_fix': 'Check protection logs and fault location, improve feeder reliability, and add UPS/backup supply.'
    },
    'Transient': {
        'likely_cause': 'Fast high-amplitude spike from lightning, switching events, or fault clearing.',
        'recommended_fix': 'Install surge protection devices, improve grounding, and review switching practices.'
    },
    'Oscillatory_Transient': {
        'likely_cause': 'Damped oscillation after capacitor or line switching and resonance in the network.',
        'recommended_fix': 'Tune switching strategy, add damping/filtering, and evaluate resonance conditions.'
    },
    'Harmonics': {
        'likely_cause': 'Nonlinear loads such as VFDs, rectifiers, and SMPS are distorting the waveform.',
        'recommended_fix': 'Install harmonic filters, use line reactors, and redistribute or isolate nonlinear loads.'
    },
    'Harmonics_with_Sag': {
        'likely_cause': 'Nonlinear-load distortion plus a simultaneous voltage dip from system stress or fault.',
        'recommended_fix': 'Apply both harmonic mitigation and sag mitigation: filters plus voltage support devices.'
    },
    'Harmonics_with_Swell': {
        'likely_cause': 'Harmonic distortion with concurrent overvoltage event from switching or load change.',
        'recommended_fix': 'Use harmonic filters and overvoltage protection; verify capacitor and regulator coordination.'
    },
    'Flicker': {
        'likely_cause': 'Low-frequency voltage fluctuation from rapidly varying loads like welders or arc furnaces.',
        'recommended_fix': 'Use STATCOM/SVC or dynamic voltage support and separate highly fluctuating loads.'
    },
    'Flicker_with_Sag': {
        'likely_cause': 'Rapid load fluctuation with an additional voltage dip event.',
        'recommended_fix': 'Address fluctuating load behavior and add sag compensation devices or feeder reinforcement.'
    },
    'Flicker_with_Swell': {
        'likely_cause': 'Rapid load fluctuation combined with an overvoltage event.',
        'recommended_fix': 'Stabilize fluctuating loads and apply overvoltage control/protection.'
    },
    'Sag_with_Oscillatory_Transient': {
        'likely_cause': 'Voltage dip followed by ringing oscillation, typically due to fault and switching interaction.',
        'recommended_fix': 'Mitigate sag and transient together: voltage support, damping, and switching/fault review.'
    },
    'Swell_with_Oscillatory_Transient': {
        'likely_cause': 'Voltage rise with oscillatory ringing due to switching and network resonance effects.',
        'recommended_fix': 'Use overvoltage protection, damping/filtering, and update switching coordination.'
    },
    'Sag_with_Harmonics': {
        'likely_cause': 'Voltage dip occurs in a network already distorted by nonlinear loads.',
        'recommended_fix': 'Use harmonic filters and sag mitigation together; inspect feeder loading and fault susceptibility.'
    },
    'Swell_with_Harmonics': {
        'likely_cause': 'Overvoltage event happens with harmonic distortion from nonlinear devices.',
        'recommended_fix': 'Apply harmonic filtering and overvoltage protection; verify regulator/capacitor operation.'
    },
    'Notch': {
        'likely_cause': 'Commutation notches from power-electronic converters and rectifier switching.',
        'recommended_fix': 'Add line reactors/transformer isolation, improve converter front-end design, and use filters.'
    },
}

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
        likely_cause : str
            Most probable real-world cause for the predicted disturbance.
        recommended_fix : str
            Suggested first corrective action for the predicted disturbance.
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
    guidance = DISTURBANCE_GUIDANCE.get(
        class_name,
        {
            'likely_cause': 'Cause is uncertain for this class in current rule set.',
            'recommended_fix': 'Collect more waveform history and perform detailed power quality audit.',
        },
    )

    return {
        'status': 'Normal' if class_name == NORMAL_CLASS else 'Abnormal',
        'disturbance_type': class_name,
        'confidence': confidence,
        'likely_cause': guidance['likely_cause'],
        'recommended_fix': guidance['recommended_fix'],
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
