# 6. How the Live Demo Notebook Works

**File:** `notebooks/06_live_demo.ipynb`

This notebook is meant to be run during a presentation. You run each cell one by one, and it shows the model predicting disturbances in real time with visual output.

---

## How to Open and Run

```bash
cd /Users/sanjay/Desktop/sanjay/final-year-project
source venv/bin/activate
jupyter notebook notebooks/06_live_demo.ipynb
```

Then press **Shift+Enter** on each cell to run it.

---

## Cell-by-Cell Breakdown

### Setup Cell (Cell 1)

**What it does:** Loads everything needed for the demo.

```python
from data_loader import load_xpqrs, get_time_axis_ms
from predictor import predict_signal, predict_batch
signals, labels = load_xpqrs('../dataset/XPQRS/')
```

- Loads all 17,000 signals from the dataset into memory
- Loads the trained Gradient Boosting model (happens automatically on first prediction)
- Creates a time axis (0 to 20 ms) for plotting

**What you see:** `Loaded 17000 signals across 17 classes`

---

### Helper Function: `demo_predict()` (Cell 3)

**What it does:** This is a reusable function that takes any signal and produces a visual output with two panels:

```
┌─────────────────────────────┬──────────────────┐
│                             │                  │
│   Waveform Plot             │   PREDICTION     │
│   (signal drawn as a        │                  │
│    green or red line)       │   Normal         │
│                             │   or             │
│   X-axis: Time (ms)         │   Abnormal       │
│   Y-axis: Amplitude         │                  │
│                             │   Disturbance    │
│                             │   type name      │
│                             │                  │
│                             │   Confidence: 98%│
│                             │                  │
│                             │   True label     │
│                             │   Correct/Wrong  │
└─────────────────────────────┴──────────────────┘
```

- **Green** background and line = Normal (clean signal)
- **Red** background and line = Abnormal (disturbance detected)
- Shows the true label and whether the prediction was correct

**How it works inside:**
1. Calls `predict_signal(signal)` — this extracts 36 features and runs the Gradient Boosting classifier
2. Checks if the result is Normal or Abnormal
3. Picks green or red color based on that
4. Draws the waveform on the left
5. Writes the prediction text on the right

---

### Demo 1: Single Signal Prediction (Cells 5, 7, 9, 11, 13, 15)

**What it does:** Shows 6 individual predictions, one per cell:

| Cell | Signal Type | What You See |
|---|---|---|
| Cell 5 | **Pure Sinusoidal** | Green panel — "Normal", smooth sine wave |
| Cell 7 | **Sag** | Red panel — "Abnormal", wave dips in the middle |
| Cell 9 | **Harmonics** | Red panel — "Abnormal", wave looks distorted/wiggly |
| Cell 11 | **Interruption** | Red panel — "Abnormal", wave drops to near zero |
| Cell 13 | **Transient** | Red panel — "Abnormal", wave has a sharp spike |
| Cell 15 | **Flicker** | Red panel — "Abnormal", wave amplitude fluctuates |

**How each cell picks a signal:**
```python
idx = (labels == 'Sag').nonzero()[0][10]
```
This finds all signals labelled "Sag" in the dataset and picks the 10th one.

**Good for presenting:** Run each cell one at a time. Explain what the disturbance is, then run the cell to show the model got it right.

---

### Demo 2: Predict All 17 Classes (Cell 17)

**What it does:** Loops through all 17 disturbance types, picks one signal from each, predicts it, and prints a table:

```
True Label                   Predicted                    Status     Confidence  Result
========================================================================================
Flicker                      Flicker                      Abnormal   100%        OK
Harmonics                    Harmonics                    Abnormal   100%        OK
Interruption                 Interruption                 Abnormal   100%        OK
Pure Sinusoidal              Pure Sinusoidal              Normal     100%        OK
Sag                          Sag                          Abnormal   98%         OK
...
========================================================================================
Correct: 17/17
```

**How it works:**
1. Gets all unique class names and sorts them alphabetically
2. For each class, picks the first signal (index 0)
3. Runs `predict_signal()` on it
4. Compares prediction to true label
5. Prints OK or MISS

**Good for presenting:** Shows the model works across all 17 types in one go. The "17/17 Correct" at the bottom is the key takeaway.

---

### Demo 3: Normal vs Abnormal Side-by-Side (Cell 19)

**What it does:** Creates a 2x3 grid of plots. Each plot overlays:
- **Blue line** = Normal signal (clean sine wave)
- **Red line** = Disturbance signal

```
┌──────────────┬──────────────┬──────────────┐
│  Sag — 98%   │ Swell — 100% │Interrupt—100%│
│  blue + red  │  blue + red  │  blue + red  │
├──────────────┼──────────────┼──────────────┤
│Transient—99% │Harmonics—100%│ Flicker—100% │
│  blue + red  │  blue + red  │  blue + red  │
└──────────────┴──────────────┴──────────────┘
```

**How it works:**
1. Gets one Pure Sinusoidal signal as the reference (blue)
2. For each of 6 disturbance types, gets one signal (red)
3. Plots both on the same axes so you can see the difference
4. Titles show the disturbance type and confidence

**Good for presenting:** Audience can visually see how each disturbance changes the waveform compared to normal.

---

### Demo 4: Confidence Breakdown (Cells 22, 23, 24)

**What it does:** For each signal, shows two panels:
- **Left:** The waveform plot
- **Right:** A horizontal bar chart of all 17 class probabilities

```
┌──────────────────┬────────────────────────────────┐
│                  │ Pure Sinusoidal  ████████ 100%  │
│   Signal Plot    │ Sag              ░ 0%           │
│                  │ Swell            ░ 0%           │
│                  │ Harmonics        ░ 0%           │
│                  │ ...              ░ 0%           │
└──────────────────┴────────────────────────────────┘
```

- The **top bar** is colored (green for Normal, red for Abnormal)
- All other bars are grey
- Percentage labels show how much probability each class got

**How it works:**
1. Runs `predict_signal()` which returns `all_probabilities` — a dict with probability for each of the 17 classes
2. Sorts them highest to lowest
3. Draws horizontal bars

Three examples are shown:
- Cell 22: Normal signal (green, 100% Pure Sinusoidal)
- Cell 23: Sag signal (red, 98% Sag)
- Cell 24: Harmonics signal (red, 100% Harmonics)

**Good for presenting:** Shows the audience that the model doesn't just guess — it assigns probabilities to every class and picks the highest one.

---

### Demo 5: Random Signal (Cell 26)

**What it does:** Picks a completely random signal from the dataset and predicts it.

```python
idx = np.random.randint(0, len(signals))  # random number between 0 and 16999
```

**Every time you run this cell, it picks a different signal.** The output is the same visual as Demo 1 — waveform plot + prediction panel.

**Good for presenting:** Interactive — you can keep pressing Shift+Enter to show the model predicting different random signals. Audience can see it works on any signal, not just cherry-picked ones.

---

### Demo 6: Batch Monitoring Simulation (Cell 28)

**What it does:** Simulates a power quality monitoring system. Picks 20 random signals and predicts all of them at once, showing a dashboard-style table:

```
#    True Label              Predicted               Status     Conf
--------------------------------------------------------------------
1    Flicker                 Flicker                 [!!]       100%
2    Swell with Harmonics    Swell with Harmonics    [!!]       53%
3    Pure Sinusoidal         Pure Sinusoidal         [OK]       100%
...
--------------------------------------------------------------------
Summary: 4 Normal, 16 Abnormal out of 20 signals
```

- `[OK]` = Normal signal (no problem)
- `[!!]` = Abnormal signal (disturbance detected)
- `<-- misclassified` appears if the prediction was wrong

**How it works:**
1. Randomly picks 20 signal indices
2. Calls `predict_batch()` which predicts all 20 in one call
3. Prints each result in a formatted table
4. Counts Normal vs Abnormal at the bottom

**Good for presenting:** Shows real-world use case — monitoring multiple signals at once and flagging the abnormal ones.

---

## Suggested Presentation Order

1. **Demo 1** — Show 2-3 individual predictions (Normal, Sag, Harmonics) to introduce the concept
2. **Demo 3** — Show the side-by-side comparison so audience sees what disturbances look like
3. **Demo 4** — Show the confidence bar chart for one signal to explain how the model decides
4. **Demo 2** — Run the all-17-classes table to prove it works on everything (17/17)
5. **Demo 5** — Let the audience see random predictions (run 3-4 times)
6. **Demo 6** — End with the batch monitoring simulation to show practical use
