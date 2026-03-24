# 8. Hyperparameter Tuning (Easy Guide)

## What Is Hyperparameter Tuning?

A machine learning model has two kinds of values:

1. **Learned parameters**
- These are learned from data automatically (for example, split rules inside trees).

2. **Hyperparameters**
- These are settings we choose before training (for example, how many trees to build).

**Hyperparameter tuning** means trying many setting combinations and keeping the one that gives the best validation score.

---

## Why We Tune in This Project

In this project, we use **Random Forest** for Power Quality Disturbance classification.

Tuning helps us find a stronger model by adjusting:
- model complexity
- tree behavior
- feature usage per split
- class weighting strategy

Without tuning, we use a default baseline model. With tuning, we search for better settings.

---

## Where Tuning Happens

Tuning is implemented in:
- `notebooks/04_model_training_evaluation.ipynb`

Key function:
- `train_and_evaluate(df, dataset_name, tune_random_forest=False, n_iter=120)`

When this is used:
```python
xpqrs_results, xpqrs_le, xpqrs_classes = train_and_evaluate(
    xpqrs_df, 'xpqrs', tune_random_forest=True, n_iter=120
)
```

it performs RandomizedSearchCV tuning.

---

## Current Tuning Configuration (Your Project)

### Search method
- **Algorithm**: `RandomizedSearchCV`
- **Iterations**: `n_iter=120`
- **Scoring metric for tuning**: `accuracy`
- **Cross-validation**: 5-fold stratified (`StratifiedKFold`, `random_state=42`)

### Hyperparameter search space

```python
param_dist = {
    'clf__n_estimators': [100, 150, 200, 300, 400, 500, 700, 900, 1100],
    'clf__max_depth': [8, 10, 15, 20, 30, 40, 50, None],
    'clf__min_samples_split': [2, 3, 4, 6, 8, 10, 15, 20],
    'clf__min_samples_leaf': [1, 2, 3, 4, 6, 8],
    'clf__max_features': ['sqrt', 'log2', None],
    'clf__class_weight': [None, 'balanced', 'balanced_subsample'],
}
```

---

## Current Best Hyperparameters (Latest Saved)

From:
- `results/tables/random_forest_best_params.json`

Current best values:
```json
{
  "n_estimators": 200,
  "min_samples_split": 2,
  "min_samples_leaf": 1,
  "max_features": "log2",
  "max_depth": null,
  "class_weight": null
}
```

Important meaning:
- `null` in JSON means `None` in Python.
- So currently:
  - `max_depth=None` (no depth limit)
  - `class_weight=None` (no class reweighting)

---

## Decision Tree Anatomy (Node, Split, Leaf)

Before tuning Random Forest, it helps to understand one decision tree.

### What is a node?
- A **node** is a decision point in the tree.
- At a node, the model asks a question like:
  - `THD <= 0.12 ?`
  - `RMS <= 0.65 ?`

### What is a split?
- A **split** is the action of dividing data into two branches based on that node question.
- Left branch = samples that satisfy the condition.
- Right branch = samples that do not.

### What is a leaf?
- A **leaf** is the final end of a branch (no more splitting).
- The leaf stores the class distribution of samples that reached it.
- For classification, prediction is based on the majority class in that leaf.

### What is depth?
- **Depth** is how far a node is from the root.
- Root node depth is 0.
- Deeper trees ask more questions and create smaller, more specific leaves.

### Why this matters for tuning
- Hyperparameters control:
  - How many trees exist (`n_estimators`)
  - How deep trees can grow (`max_depth`)
  - When nodes are allowed to split (`min_samples_split`)
  - How small leaves are allowed to be (`min_samples_leaf`)
  - How much feature randomness is injected (`max_features`)

---

## Hyperparameters Explained in Simple Terms

### 1) `n_estimators`
- **What it controls**: Number of trees in the forest.
- **Simple idea**: More trees = more opinions before final vote.
- **Effect**:
  - Too low: unstable predictions.
  - Higher: usually more stable and accurate, but slower.
- **Current best**: `200`.

How tweaking works:
- If you increase it:
  - Variance usually decreases.
  - Predictions become more stable.
  - Training and inference time increase.
- If you decrease it:
  - Model trains faster.
  - But results can fluctuate more between runs.
- Practical note:
  - Gains often saturate after a certain point; doubling trees does not always double quality.

### 2) `max_depth`
- **What it controls**: Maximum depth of each tree.
- **Simple idea**: How many questions a tree can ask before deciding.
- **Effect**:
  - Small depth: simpler model, less overfitting, may underfit.
  - Large/None: more detail, can overfit.
- **Current best**: `None` (unlimited depth).

How tweaking works:
- If you increase it (or set `None`):
  - Trees can form very specific rules.
  - Training accuracy may rise.
  - Overfitting risk increases.
- If you decrease it:
  - Trees stay shallow and general.
  - Better regularization.
  - Underfitting risk increases.
- Practical note:
  - For noisy features, limiting depth can improve test performance.

### 3) `min_samples_split`
- **What it controls**: Minimum samples required to split a node.
- **Simple idea**: A node needs this many examples before creating child branches.
- **Effect**:
  - Low values: trees split more, become complex.
  - High values: trees are more conservative.
- **Current best**: `2`.

How tweaking works:
- If you increase it:
  - Fewer splits happen.
  - Trees become smoother and less complex.
  - Overfitting decreases, but underfitting may increase.
- If you decrease it:
  - More splitting is allowed.
  - Trees capture fine patterns and also noise.
- Practical note:
  - This parameter regularizes internal nodes (decision points).

### 4) `min_samples_leaf`
- **What it controls**: Minimum samples allowed in a leaf node.
- **Simple idea**: Final decision buckets cannot be too tiny.
- **Effect**:
  - Low values: flexible but can memorize noise.
  - Higher values: smoother, more general decisions.
- **Current best**: `1`.

How tweaking works:
- If you increase it:
  - Very small leaves are blocked.
  - Probability estimates are smoother.
  - Model becomes more robust to noise.
- If you decrease it:
  - Leaves can become tiny and highly specific.
  - Model may memorize rare training cases.
- Practical note:
  - This parameter directly controls leaf granularity.

### 5) `max_features`
- **What it controls**: How many features are considered at each split.
- **Simple idea**: Each split only sees part of the feature list.
- **Effect**:
  - Smaller subset: more diversity between trees.
  - Larger subset: stronger individual trees, less diversity.
- **Current best**: `log2`.

How tweaking works:
- If you use smaller subsets (`sqrt`, `log2`):
  - Trees become less correlated.
  - Ensemble diversity increases.
  - Generalization often improves.
- If you use larger subset (`None`):
  - Each split can search all features.
  - Trees become stronger individually but more similar to each other.
  - Overfitting chance can rise.
- Practical note:
  - This is one of the most important Random Forest knobs.

### 6) `class_weight`
- **What it controls**: Class importance during training.
- **Simple idea**: Tell the model to care more about minority classes.
- **Effect**:
  - `None`: treat classes equally.
  - `balanced`: give rare classes more weight globally.
  - `balanced_subsample`: similar balancing but per bootstrap sample.
- **Current best**: `None`.

How tweaking works:
- If you use `balanced`:
  - Minority classes get stronger influence in split decisions.
  - Macro metrics (like macro-F1) may improve.
  - Overall accuracy can go up or down depending on data.
- If you use `balanced_subsample`:
  - Similar idea, but weights are recalculated for each bootstrap sample.
  - Can be more stable in some imbalanced settings.
- If you keep `None`:
  - Best when dataset is already balanced.

---

## Parameter Interactions (Important)

Hyperparameters do not act independently. They influence each other.

1. `max_depth` with `min_samples_leaf`
- Deep trees plus tiny leaves can overfit.
- If you keep large depth, consider larger `min_samples_leaf`.

2. `max_depth` with `min_samples_split`
- Large depth and very low split threshold allow aggressive branching.
- Increase `min_samples_split` to regularize deep trees.

3. `n_estimators` with `max_features`
- Smaller `max_features` increases tree diversity.
- More trees (`n_estimators`) then helps average out variance.

4. `class_weight` with metric choice
- If you tune for accuracy, class weighting may be less favored.
- If minority recall is important, consider tuning on `f1_macro` instead.

---

## How To Read Tuning Outputs

### `random_forest_tuning_trials.csv`
- Every row is one tested hyperparameter combination.
- Useful columns to inspect:
  - parameter columns (begin with `param_`)
  - `mean_test_score` or rank columns
  - split-wise test columns for stability

### `random_forest_best_params.json`
- Best combination selected by `RandomizedSearchCV`.
- This is what gets used to build the final tuned classifier.

### `random_forest_baseline_vs_tuned.csv`
- Direct baseline vs tuned metric comparison.
- Use this to decide if tuning gave practical gains.

---

## How Results Are Saved

After tuning, these files are created:

1. `results/tables/random_forest_tuning_trials.csv`
- Every tried combination + CV scores.

2. `results/tables/random_forest_best_params.json`
- Best hyperparameter set.

3. `results/models/xpqrs_random_forest_tuned.pkl`
- Final tuned model pipeline.

4. `results/tables/random_forest_baseline_vs_tuned.csv`
- Side-by-side baseline vs tuned comparison.

---

## How To Change Hyperparameters Safely

1. Open `notebooks/04_model_training_evaluation.ipynb`.
2. In the training pipeline cell, edit `param_dist` values.
3. Keep value ranges realistic (not too narrow, not too extreme).
4. Increase or decrease `n_iter` based on runtime budget.
5. Re-run from the training cell onward.
6. Check:
- `random_forest_best_params.json`
- `random_forest_baseline_vs_tuned.csv`

---

## Practical Tips

1. If tuned model is not improving:
- Increase `n_iter` (e.g., 120 -> 180).
- Narrow around promising values from `random_forest_tuning_trials.csv`.

2. If training is too slow:
- Reduce very large `n_estimators` values.
- Use smaller search ranges temporarily.

3. If overfitting appears:
- Try smaller `max_depth`.
- Increase `min_samples_leaf` and `min_samples_split`.

4. If minority classes perform poorly:
- Prioritize `class_weight='balanced'` options.

---

## Quick Summary

- Tuning in this project uses **RandomizedSearchCV** with **120 trials**.
- It currently optimizes **accuracy** with 5-fold stratified CV.
- Latest best setup is:
  - `n_estimators=200`
  - `max_depth=None`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `max_features='log2'`
  - `class_weight=None`
- Main tuning outputs are saved in `results/tables/` and `results/models/` for analysis and deployment.