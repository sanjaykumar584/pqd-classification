# 7. Recent Project Changes (Last 4 Commits)

This document summarizes the **latest 4 commits** and what they changed in the project.

## Commit 1: `449939b` — Add model comparison notebook and update results

### What was added/updated
- Added `notebooks/08_all_model_comparison.ipynb`
- Updated `notebooks/05_results_comparison.ipynb`
- Added comparison plots:
  - `results/figures/all_models_accuracy_comparison.png`
  - `results/figures/all_models_f1_comparison.png`
- Updated consolidated table:
  - `results/tables/final_comparison.csv`

### Why it matters
- Introduces a dedicated notebook for full model-to-model comparison.
- Improves review-readiness with visual accuracy/F1 comparisons.

---

## Commit 2: `9844e47` — Add tuning analysis figures and summary table for Random Forest model

### What was added/updated
- Added `notebooks/07_hyperparameter_tuning_analysis.ipynb`
- Updated `notebooks/06_live_demo.ipynb`
- Added tuning analysis figures:
  - `results/figures/tuning_trial_progression.png`
  - `results/figures/tuning_top_trials.png`
  - `results/figures/tuning_per_param_distribution.png`
  - `results/figures/tuning_cv_stability.png`
  - `results/figures/tuning_baseline_vs_tuned_analysis.png`
- Added curated tuning table:
  - `results/tables/random_forest_tuning_summary.csv`

### Why it matters
- Makes hyperparameter search behavior transparent.
- Provides reviewer-friendly diagnostics for trial quality and stability.

---

## Commit 3: `dbe61ca` — Add disturbance guidance with likely causes and recommended fixes for each class

### What was added/updated
- Updated inference logic in `src/predictor.py`
- Updated demo notebook: `notebooks/06_live_demo.ipynb`

### Functional change
- Prediction output now includes practical domain guidance per disturbance class:
  - likely cause
  - recommended fix

### Why it matters
- Converts raw model output into actionable power-quality interpretation.
- Makes live demo outputs easier for non-ML audiences to understand.

---

## Commit 4: `f463d4e` — Update model results and figures; add tuning trials and best parameters for Random Forest

### What was added/updated
- Updated training/evaluation notebooks:
  - `notebooks/04_model_training_evaluation.ipynb`
  - `notebooks/05_results_comparison.ipynb`
  - `notebooks/06_live_demo.ipynb`
- Updated model result tables:
  - `results/tables/xpqrs_model_results.csv`
  - `results/tables/pq_model_results.csv`
  - `results/tables/final_comparison.csv`
- Added tuning artifacts:
  - `results/tables/random_forest_tuning_trials.csv`
  - `results/tables/random_forest_best_params.json`
  - `results/tables/random_forest_baseline_vs_tuned.csv`
- Added tuned model artifact:
  - `results/models/xpqrs_random_forest_tuned.pkl`
- Updated visual outputs:
  - `results/figures/xpqrs_cm_random_forest.png`
  - `results/figures/pq_cm_random_forest.png`
  - `results/figures/xpqrs_f1_heatmap.png`
  - `results/figures/final_feature_importance.png`
  - `results/figures/importance_by_domain.png`
  - `results/figures/cross_dataset_comparison.png`
  - `results/figures/roc_curves.png`
  - `results/figures/tsne_xpqrs.png`

### Why it matters
- Refreshes evaluation outputs and visual evidence across datasets.
- Captures full Random Forest tuning provenance (trials + best parameters + baseline comparison).

---

## Overall Impact of These 4 Commits

- Added two major analysis notebooks:
  - Model comparison (`08_all_model_comparison.ipynb`)
  - Hyperparameter tuning diagnostics (`07_hyperparameter_tuning_analysis.ipynb`)
- Expanded interpretability in inference via class-wise guidance.
- Improved experiment traceability through tuning artifacts and summary tables.
- Updated multiple figures/tables used in reports and presentations.

## Note

These commit summaries reflect the historical changes in the last 4 commits. If newer local edits exist (for example, model-selection updates), they may not yet be reflected here until committed.