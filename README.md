# Insurance Pricing Modeling

This repository is a lightweight, notebook-first modeling project. It contains exploratory analysis and feature engineering workflows, plus a small shared `utils.py` module used across notebooks.

## Notebook Guide

This project is organized as a short, linear notebook workflow. Each notebook focuses on one stage of understanding the data and engineering features for modeling.

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

Typical outputs:
- Candidate feature lists and transformation decisions

---

### 04_cat_features.ipynb — Categorical Feature Engineering
Primary goals:
- Diagnose and handle high-cardinality categoricals
- Consolidate rare levels and/or build exposure/credibility-aware groupings

Typical outputs:
- Level frequency/exposure summaries
- Proposed groupings/bins and mapping tables
- Encoding approach decisions and rationale

## Repository Structure

```text
.
├── 01_data_eda.ipynb
├── 02_capping_levels.ipynb
├── 03_numeric_features.ipynb
├── 04_cat_features.ipynb
├── utils.py
├── requirements.txt
├── .gitignore
└── data/
    └── (data files used by notebooks)