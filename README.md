# Insurance Pricing Modeling

This repository is a lightweight, notebook-first modeling project. It contains exploratory analysis and feature engineering workflows, plus a small shared `utils.py` module used across notebooks.

## Project Roadmap

This project is intentionally structured as a staged modeling workflow: start with a transparent, inference-friendly GLM to establish a defensible baseline (“indications”), then layer on regularization and resampling to quantify stability, and finally compare against a purely predictive machine-learning baseline.

### Phase 1 — Baseline GLM (Indications)
- **Goal:** Build a strong, interpretable Tweedie GLM that produces an initial set of indicated relativities.
- **Concerns addressed:** Establishes a clear starting point for understanding signal, interactions, and potential overfitting.

### Phase 2 — Elastic Net GLM (Regularized Indications)
- **Goal:** Fit a regularized model (elastic net) to reduce coefficient variance and improve out-of-sample performance.
- **Output:** A second set of “indications,” optimized with a more predictive / stability-oriented objective.

### Phase 3 — Bootstrapped GLM (Uncertainty & Stability)
- **Goal:** Bootstrap the GLM to estimate sampling variability and produce reasonable ranges around key coefficients/relativities.
- **Output:** Coefficient distributions and uncertainty intervals to support factor stability discussions.

### Phase 4 — Factor Selection Exhibit & Final Model
- **Goal:** Combine:
  - Univariate summaries,
  - baseline GLM results,
  - elastic net results,
  - bootstrap stability metrics,
  into a structured “factor selection exhibit.”
- **Output:** A final selected model built from the chosen factor definitions and levels.

Models produced by the end of this phase:
- Baseline GLM
- Elastic Net (regularized GLM)
- Bootstrapped GLM (coefficient distributions)
- Final Selected Model (human-guided factor selection)

### Phase 5 — Predictive ML Benchmark (No Inference Constraint)
This workflow intentionally includes a **human-guided factor selection** phase, which is atypical in many data science settings. In US insurance pricing work, interpretability and controlled factor definitions are often a feature, not a bug.

To benchmark that approach, I will also fit at least one “pure ML” model where prediction is the primary objective and interpretability/inference is not a constraint. The goal is to compare performance and stability across:
- inference-driven GLM approaches, and
- prediction-first machine learning approaches.

## Notebook Guide

This project is organized as a short, linear notebook workflow. Each notebook focuses on one stage of the full modeling process.

### 01_data_eda.ipynb - Data Exploration
Primary goals:
- Understand the structure of the dataset (rows, columns, types)
- Profile missingness and potential data quality issues
- Review univariate distributions and basic summaries
- Review 2-way linear correlations between numeric variables
- Spot obvious outliers and edge cases worth special handling

Outputs:
- Summary tables, distribution plots, missingness diagnostics
- Early hypotheses about which fields will need capping, binning, or recoding

---

### 02_capping_levels.ipynb - Capping / Outlier Treatment
Primary goals:
- Evaluate whether extremely small or extremely large values in the target variable (Pure Premium) should be truncated
- Compare alternative cap thresholds and their impact on out of sample prediction quality of uncapped pure premium
- Document rationale for the chosen caps (stability, reasonableness, robustness)

Outputs:
- Before/after comparisons (summary stats and plots)
- Selected cap rules and implementation notes

---

### 03_numeric_features.ipynb - Numeric Feature Engineering
Primary goals:
- Create/transform numeric predictors into modeling-ready features
- Build and validate feature sets that are stable and interpretable

Outputs:
- Candidate feature lists and transformation decisions

---

### 04_cat_features.ipynb — Categorical Feature Engineering
Primary goals:
- Diagnose and handle high-cardinality categoricals
- Consolidate rare levels and/or build exposure/credibility-aware groupings

Outputs:
- Level frequency/exposure summaries
- Proposed groupings/bins and mapping tables
- Encoding approach decisions and rationale

### 05_pure_premium_glm.ipynb — Pure Premium Tweedie GLM
Primary goals:
- Fit a Tweedie GLM for **pure premium** (loss cost) using the engineered numeric and categorical features
- Apply spline specifications for numeric predictors and factor levels for categorical predictors
- Produce standard GLM outputs for **inference** (coefficients, standard errors, p-values) and diagnostics (e.g., lift by prediction decile)

Outputs:
- Final model formula/specification
- Coefficient table and inference summary
- Predicted pure premium and basic performance diagnostics (lift / calibration by decile)

## Repository Structure

```text
.
├── 01_data_eda.ipynb
├── 02_capping_levels.ipynb
├── 03_numeric_features.ipynb
├── 04_cat_features.ipynb
├── 05_pure_premium_glm.ipynb
├── utils.py
├── requirements.txt
├── .gitignore
└── data/
    └── (data files used by notebooks)