# Loan Default Prediction

Predicting whether a LendingClub loan will be **Fully Paid (0)** or **Default (1)** using Logistic Regression, Random Forest, and XGBoost  with a focus on handling class imbalance and choosing a sensible decision threshold.

## Overview

Lenders lose money when borrowers default, so ranking and flagging risky loans before approval matters. This project builds an end-to-end binary classification workflow in a Jupyter Notebook: data cleaning, feature engineering, stratified train/test split, hyperparameter tuning with cross-validation, and threshold selection.

## Dataset

- **Source:** Cleaned LendingClub loan data (`final_data_set.csv`)
- **Size:** 50,000 loans, 27 columns
- **Target:** `loan_status` (0 = Fully Paid, 1 = Charged Off)
- **Class balance:** 80% Fully Paid (40,000) vs 20% Default (10,000), so the data is imbalanced

Key columns include loan amount, term, interest rate, instalment, employment length, home ownership, annual income, DTI, delinquency history, credit utilisation, revolving balance and credit limits.

> The dataset file is not included in this repository. Place `final_data_set.csv` in the project root before running the notebook.

## Workflow

### 1. Data cleaning
- Converted `term` from text (e.g. "36 months") to an integer
- Dropped rows with missing values (50,000 → 46,918)
- Removed rows with unknown `emp_length` and rare `home_ownership` categories (`ANY`, `NONE`, `OTHER`)
- Mapped `emp_length` to a numeric scale (`< 1 year` → 0.5 … `10+ years` → 10)
- Dropped `pub_rec_bankruptcies`, `has_prior_delinq` and `tot_coll_amt`
- Final modelling set: **44,084 loans**

### 2. Feature engineering
| Feature | Definition |
|---|---|
| `credit_tenure_years` | 2020 minus the year of `earliest_cr_line` |
| `installment_to_income` | Instalment divided by monthly income |
| `loan_to_income` | Loan amount divided by annual income |
| `available_revol_credit` | Total revolving limit minus revolving balance |
| `open_acc_ratio` | Open accounts divided by total accounts |
| `high_recent_inquiries` | 1 if more than 2 inquiries in the last 6 months |
| `has_recent_delinq` | 1 if any delinquency in the last 2 years |

`purpose`, `home_ownership` and `verification_status` are one-hot encoded (`drop_first=True`), giving **42 features** in total.

### 3. Train/test split
80/20 **stratified** split (`random_state=42`), so the default rate is the same in both sets:
- Train: 35,267 rows
- Test: 8,817 rows

### 4. Modelling
- **Random Forest**: tuned, with threshold selection via Youden's J statistic
- **XGBoost**: `RandomizedSearchCV` (15 iterations, 5-fold CV, ROC-AUC scoring) with `scale_pos_weight` to address imbalance
- **Logistic Regression**: `StandardScaler` + L1/L2 regularisation, `class_weight='balanced'`, `RandomizedSearchCV` (10 iterations, 3-fold CV)

## Results

All metrics are on the held-out test set (8,817 loans, 1,753 defaults).

| Model | ROC-AUC | Default precision | Default recall | Default F1 | Threshold |
|---|---|---|---|---|---|
| Random Forest (tuned) | 0.7031 | 0.32 | 0.64 | 0.43 | 0.3146 (Youden's J) |
| XGBoost (tuned) | **0.7101** | 0.29 | 0.73 | 0.42 | 0.50 |
| Logistic Regression (tuned) | 0.7052 | 0.32 | 0.63 | 0.42 | 0.50 |

**Best XGBoost parameters:** `n_estimators=500`, `max_depth=5`, `learning_rate=0.01`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`.

**Takeaways**
- XGBoost has the best ranking ability (highest ROC-AUC) and the highest default recall, at the cost of lower precision.
- All three models land close together (ROC-AUC ≈ 0.70–0.71), which suggests the available features carry a limited but real signal; a simple regularised Logistic Regression is competitive with the tree ensembles.
- Because defaults are only 20% of the data, plain accuracy is misleading. Precision, recall, F1 and ROC-AUC are used instead.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, ydata-profiling

## Getting started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numpy pandas matplotlib seaborn scikit-learn xgboost ydata-profiling
jupyter notebook Loan_Default_Prediction.ipynb
```

Run the cells from top to bottom.

## Project structure

```
├── Loan_Default_Prediction.ipynb   # Full analysis and modelling
├── final_data_set.csv              # Dataset (add locally)
└── README.md
```

## Limitations and future work

- Features are cast to integers after encoding, which truncates decimals in columns such as `int_rate` and `dti`; keeping them as floats is a straightforward improvement.
- Rows with missing values were dropped rather than imputed.
- Imbalance is handled with class weights; resampling (e.g. SMOTE) and probability calibration could be compared.
- Add PR-AUC, feature-importance plots and a cost-based threshold (weighing the cost of a missed default against a rejected good loan).

