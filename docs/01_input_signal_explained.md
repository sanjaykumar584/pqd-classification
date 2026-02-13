# 1. The Input Signal - What It Is and What It Means

## What is a Power Quality Signal?

Electricity from the power grid is supposed to be a clean **sinusoidal wave** (a smooth, repeating S-curve) at **50 Hz** (in India). This is what a "normal" power signal looks like:

```
Voltage
  +1 |    ****
     |  **    **
     | *        *
  0  |*----------*----------*---> Time
     |            *        *
     |             **    **
  -1 |               ****
     |<--- 20 ms (one cycle) --->|
```

When something goes wrong in the grid (a fault, a motor starting, lightning, etc.), the signal gets **distorted**. These distortions are called **Power Quality Disturbances (PQDs)**.

## What Does One Input Signal Look Like?

Each input to our system is a **numpy array of exactly 100 numbers**:

```python
signal = [0.0, 0.314, 0.587, 0.809, 0.951, ..., -0.309]  # 100 values
```

These 100 numbers represent **voltage measurements taken over 20 milliseconds** (one complete cycle of the 50 Hz wave).

### Signal Parameters

| Parameter | Value | What It Means |
|---|---|---|
| **Sampling Rate** | 5,000 Hz | 5,000 measurements per second |
| **Signal Length** | 100 samples | Number of data points in one signal |
| **Duration** | 20 ms (0.02 seconds) | Total time captured |
| **Fundamental Frequency** | 50 Hz | The base frequency of the power grid |
| **Amplitude Range** | Approximately -2.5 to +2.5 | Normalized voltage (not actual volts) |

### Why 100 Samples?

At 5,000 Hz sampling rate with a 50 Hz power signal:
- One complete cycle = 1/50 = 0.02 seconds = 20 ms
- Samples in one cycle = 5,000 x 0.02 = **100 samples**

So each signal captures exactly **one full cycle** of the electrical waveform.

## The 17 Types of Disturbances

The system can identify these 17 patterns:

### Simple Disturbances

| Class | What Happens | Real-World Cause |
|---|---|---|
| **Pure_Sinusoidal** | Clean, normal signal | Everything is working fine |
| **Sag** | Voltage drops (dip) for part of the cycle | Heavy load starting (e.g., motor, AC) |
| **Swell** | Voltage rises above normal | Sudden load removal, capacitor switching |
| **Interruption** | Voltage drops to near zero | Power outage, breaker trip |
| **Transient** | Sharp, brief voltage spike | Lightning strike, switching surge |
| **Oscillatory_Transient** | Ringing/oscillation added to signal | Capacitor bank switching |
| **Harmonics** | Waveform becomes distorted/non-sinusoidal | Non-linear loads (computers, LEDs, VFDs) |
| **Flicker** | Voltage fluctuates rapidly | Arc furnaces, welding machines |
| **Notch** | Small V-shaped cuts in the waveform | Power electronic converter switching |

### Compound Disturbances (two problems at once)

| Class | What It Combines |
|---|---|
| **Harmonics_with_Sag** | Harmonic distortion + voltage dip |
| **Harmonics_with_Swell** | Harmonic distortion + voltage rise |
| **Flicker_with_Sag** | Voltage fluctuation + voltage dip |
| **Flicker_with_Swell** | Voltage fluctuation + voltage rise |
| **Sag_with_Harmonics** | Voltage dip + harmonic distortion |
| **Swell_with_Harmonics** | Voltage rise + harmonic distortion |
| **Sag_with_Oscillatory_Transient** | Voltage dip + ringing oscillation |
| **Swell_with_Oscillatory_Transient** | Voltage rise + ringing oscillation |

## The Dataset (XPQRS)

The XPQRS dataset contains **17,000 signals total**:
- **1,000 signals per class** (perfectly balanced)
- Stored as **17 CSV files** (one per class)
- Each CSV has 1,000 rows (signals) and 100 columns (sample points)
- Signals are computer-generated (simulated) to represent realistic disturbances

### How the Data Is Stored

```
dataset/XPQRS/
    Flicker.csv                          # 1000 flicker signals
    Harmonics.csv                        # 1000 harmonic signals
    Pure_Sinusoidal.csv                  # 1000 normal signals
    Sag.csv                              # 1000 sag signals
    ...                                  # 17 files total
```

Each CSV row is one signal:
```
0.0, 0.314, 0.587, 0.809, ..., -0.309   # 100 comma-separated values
```

### Loading the Data in Python

```python
from data_loader import load_xpqrs

signals, labels = load_xpqrs('dataset/XPQRS/')

# signals.shape = (17000, 100)  -- 17000 signals, each with 100 samples
# labels.shape  = (17000,)      -- class name for each signal

print(signals[0])    # First signal: array of 100 float values
print(labels[0])     # Its label: e.g., "Flicker"
```
