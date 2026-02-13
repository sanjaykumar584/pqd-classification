# 5. How to Give Input and What the Output Means

## Giving Input

### Option 1: Single Signal

```python
import sys
sys.path.insert(0, 'src')
import numpy as np
from predictor import predict_signal

# Your signal: a numpy array of exactly 100 values
signal = np.array([0.0, 0.314, 0.587, 0.809, ...])  # 100 values

result = predict_signal(signal)
```

### Option 2: Load from the Dataset

```python
from data_loader import load_xpqrs

signals, labels = load_xpqrs('dataset/XPQRS/')

# Pick any signal by index (0 to 16999)
result = predict_signal(signals[5000])

# Or pick by class name
sag_signals = signals[labels == 'Sag']
result = predict_signal(sag_signals[0])
```

### Option 3: Multiple Signals at Once

```python
from predictor import predict_batch

# Pass multiple signals as a 2D array (n, 100)
batch = signals[0:50]  # first 50 signals
results = predict_batch(batch)

for i, r in enumerate(results):
    print(f"Signal {i}: {r['status']} - {r['disturbance_type']} ({r['confidence']:.2f})")
```

### Option 4: Test One Signal from Each Class

```python
from predictor import predict_signal
from data_loader import load_xpqrs
import numpy as np

signals, labels = load_xpqrs('dataset/XPQRS/')

for cls in np.unique(labels):
    idx = (labels == cls).nonzero()[0][0]
    r = predict_signal(signals[idx])
    print(f"True: {cls:35s} -> {r['status']:8s} | {r['disturbance_type']:35s} (conf: {r['confidence']:.2f})")
```

### Input Requirements

| Requirement | Detail |
|---|---|
| **Type** | numpy array (`np.ndarray`) |
| **Shape** | Exactly `(100,)` for single signal, `(n, 100)` for batch |
| **Data type** | Numeric (float64 preferred) |
| **Values** | Voltage samples, typically in range -2.5 to +2.5 |
| **Sampling rate** | 5,000 Hz (100 samples = 1 cycle at 50 Hz) |

If the shape is wrong, you get a clear error:
```
ValueError: Expected signal shape (100,), got (50,)
```

## Understanding the Output

`predict_signal()` returns a dictionary with 4 keys:

```python
{
    'status': 'Abnormal',
    'disturbance_type': 'Sag',
    'confidence': 0.98,
    'all_probabilities': {
        'Flicker': 0.0,
        'Flicker_with_Sag': 0.01,
        'Flicker_with_Swell': 0.0,
        'Harmonics': 0.0,
        'Harmonics_with_Sag': 0.0,
        'Harmonics_with_Swell': 0.0,
        'Interruption': 0.0,
        'Notch': 0.0,
        'Oscillatory_Transient': 0.0,
        'Pure_Sinusoidal': 0.0,
        'Sag': 0.98,
        'Sag_with_Harmonics': 0.01,
        'Sag_with_Oscillatory_Transient': 0.0,
        'Swell': 0.0,
        'Swell_with_Harmonics': 0.0,
        'Swell_with_Oscillatory_Transient': 0.0,
        'Transient': 0.0
    }
}
```

### Key 1: `status`

| Value | Meaning |
|---|---|
| `"Normal"` | The signal is a clean sinusoidal wave — no disturbance detected |
| `"Abnormal"` | A disturbance was detected in the signal |

The rule is simple: if the predicted class is `Pure_Sinusoidal`, status is `Normal`. Everything else is `Abnormal`.

### Key 2: `disturbance_type`

The specific type of disturbance identified. One of these 17 values:

| Value | What It Means |
|---|---|
| `Pure_Sinusoidal` | Normal, clean power signal |
| `Sag` | Voltage dropped below normal level |
| `Swell` | Voltage rose above normal level |
| `Interruption` | Voltage dropped to near zero (power loss) |
| `Transient` | Brief, sharp voltage spike |
| `Oscillatory_Transient` | Ringing oscillation superimposed on signal |
| `Harmonics` | Waveform distorted by non-linear loads |
| `Flicker` | Rapid voltage fluctuations |
| `Notch` | Small V-shaped cuts in the waveform |
| `Harmonics_with_Sag` | Harmonics + voltage dip |
| `Harmonics_with_Swell` | Harmonics + voltage rise |
| `Flicker_with_Sag` | Flicker + voltage dip |
| `Flicker_with_Swell` | Flicker + voltage rise |
| `Sag_with_Harmonics` | Voltage dip + harmonics |
| `Swell_with_Harmonics` | Voltage rise + harmonics |
| `Sag_with_Oscillatory_Transient` | Voltage dip + ringing |
| `Swell_with_Oscillatory_Transient` | Voltage rise + ringing |

### Key 3: `confidence`

A number between 0.0 and 1.0 representing how sure the model is about its prediction.

| Confidence | Interpretation |
|---|---|
| **0.90 - 1.00** | Very confident. The prediction is almost certainly correct. |
| **0.70 - 0.90** | Fairly confident. Likely correct but some uncertainty. |
| **0.50 - 0.70** | Uncertain. The model is unsure between two or more classes. |
| **Below 0.50** | Low confidence. The signal may not match any learned pattern well. |

**How it works:** The Random Forest has 100 decision trees. Each tree votes for a class. The confidence is the fraction of trees that agree. If 98 out of 100 trees say "Sag", the confidence is 0.98.

### Key 4: `all_probabilities`

A dictionary showing the probability assigned to every class. All 17 values sum to 1.0.

This is useful for:
- **Seeing the second-most-likely class** when confidence is low
- **Understanding confusion** — if a Sag signal shows 0.60 for Sag and 0.30 for Flicker_with_Sag, the model sees both patterns

Example of reading it:
```python
result = predict_signal(signal)
probs = result['all_probabilities']

# Sort by probability (highest first)
sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
for cls, prob in sorted_probs[:5]:
    print(f"  {cls}: {prob:.2%}")

# Output:
#   Sag: 98.00%
#   Sag_with_Harmonics: 1.00%
#   Flicker_with_Sag: 1.00%
#   Pure_Sinusoidal: 0.00%
#   ...
```

## Complete Working Example

```python
import sys
sys.path.insert(0, 'src')
import numpy as np
from predictor import predict_signal
from data_loader import load_xpqrs

# Load dataset
signals, labels = load_xpqrs('dataset/XPQRS/')

# Pick a signal (index 10500 is a Sag signal)
signal = signals[10500]
true_label = labels[10500]

# Predict
result = predict_signal(signal)

# Display results
print(f"True label:       {true_label}")
print(f"Status:           {result['status']}")
print(f"Disturbance type: {result['disturbance_type']}")
print(f"Confidence:       {result['confidence']:.2%}")
print()
print("Top 3 probabilities:")
sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
for cls, prob in sorted_probs[:3]:
    print(f"  {cls}: {prob:.2%}")
```

Expected output:
```
True label:       Sag
Status:           Abnormal
Disturbance type: Sag
Confidence:       98.00%

Top 3 probabilities:
  Sag: 98.00%
  Sag_with_Harmonics: 1.00%
  Flicker_with_Sag: 1.00%
```

## Model Accuracy Summary

The model correctly classifies **90.62%** of signals across all 17 disturbance types. Performance is highest for distinct disturbances (Interruption, Harmonics, Pure Sinusoidal) and slightly lower for compound disturbances that share similar characteristics.
