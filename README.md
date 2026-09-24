# Loan Approval Prediction

A binary-classification project that predicts whether a loan application is **Approved** or **Rejected**, comparing **Logistic Regression** and **Random Forest** with leakage-safe scikit-learn pipelines and imbalance-aware evaluation.

> **Status:** model-training phase complete. Deployment has **not** been done yet and is planned as a separate later phase.

## 1. Project Overview

This repository contains the model-training phase of an internship task. Data collection, cleaning, preprocessing and EDA were completed earlier; this notebook trains, tunes (lightly), evaluates and compares two classifiers on the structured applicant data.

## 2. Problem Statement

Given an applicant's income, credit score, requested loan amount, debt-to-income ratio and employment status, predict whether the loan application is approved.

## 3. Objective

* Train and compare Logistic Regression and Random Forest on a binary classification task.
* Evaluate on a held-out test set using metrics that suit an imbalanced target.
* Aim for at least 90 % test accuracy **without** data leakage, test-set tuning, or label manipulation, and report the real result whatever it is.

## 4. Dataset

* File: `data/loan_data.csv`
* Rows: 24,000 · Columns: 7 · Missing values: 0 · Duplicate rows: 0
* Target: `Approval` (`Rejected` = 0, `Approved` = 1), imbalanced at about **83.61 % Rejected / 16.39 % Approved**

The dataset shows very clean, strong patterns and appears to contain synthetic/data-generation structure. Results describe this dataset only and are **not** evidence about real-world lending.

## 5. Dataset Features

| Column | Type | Used in initial models |
|---|---|---|
| `Income` | numeric | Yes |
| `Credit_Score` | numeric | Yes |
| `Loan_Amount` | numeric | Yes |
| `DTI_Ratio` | numeric | Yes |
| `Employment_Status` | categorical (`employed` / `unemployed`) | Yes |
| `Text` | free text (loan purpose) | **No**: needs NLP feature extraction; left for future work |
| `Approval` | target | Target |

## 6. Machine Learning Approach

Supervised binary classification. The data is split 80/20 (`random_state=42`, `stratify=y`), giving 19,200 training and 4,800 test rows. Model selection and tuning use 5-fold stratified cross-validation on the **training set only**; the test set is used once for the final evaluation.

## 7. Data Preprocessing

All preprocessing is inside scikit-learn `Pipeline` / `ColumnTransformer` objects, so it is fitted only on training data:

* Target label encoding: `Rejected → 0`, `Approved → 1`
* One-hot encoding of `Employment_Status`
* `StandardScaler` on numeric features for Logistic Regression (not needed for Random Forest)

Leakage checks in the notebook: no shared rows between train and test, no feature-combination overlap between test and train, target and `Text` excluded from features, and scaler statistics confirmed to match the training data.

## 8. Exploratory Data Analysis

EDA was completed in an earlier phase (not part of this notebook). Its main observations:

* `Credit_Score` has the strongest numerical relationship with approval.
* `Income` is positively related to approval; `DTI_Ratio` is negatively related; `Loan_Amount` has a weaker negative relationship.
* `Employment_Status` is very strongly related to approval. In the training split, none of the unemployed applicants were approved.

## 9. Models

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42`; default vs `class_weight="balanced"` compared by cross-validated F1 (default was selected) |
| Random Forest | `random_state=42`; small grid search over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight` scored by F1 (selected: 200 trees, `max_depth=20`, `min_samples_split=2`, `min_samples_leaf=1`, no class weights) |

SMOTE was not used. Linear Regression was not used because the task is classification.

## 10. Model Evaluation

Metrics: Accuracy, Precision, Recall, F1-score (for the *Approved* class), confusion matrix, classification report. Because about 84 % of applications are *Rejected*, a model that always predicts *Rejected* scores about 83.6 % accuracy with 0 % recall, so accuracy alone is not enough. A threshold analysis (on out-of-fold training predictions) shows the precision/recall trade-off; the reported results use the default 0.5 threshold.

## 11. Results

Test-set results (4,800 held-out rows, threshold 0.5):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 99.58 % | 98.24 % | 99.24 % | 98.74 % |
| Logistic Regression | 92.62 % | 77.44 % | 77.64 % | 77.54 % |
| *Majority-class baseline (reference)* | 83.60 % | 0.00 % | 0.00 % | 0.00 % |

Both models exceeded the 90 % accuracy target. Random Forest was better on every measured metric on this dataset. Logistic Regression's accuracy looks good, but its Approved-class recall and precision are only about 77 %.

Random Forest reaches 100 % accuracy on the training data and 99.58 % on the test set (cross-validated accuracy about 99.5 %). Possible reasons for such a high score, which are hypotheses rather than proven facts: the dataset seems synthetic and rule-like, `Employment_Status` is extremely informative, and trees can capture non-linear rules that a linear model cannot. Scores like this should not be expected on real lending data.

## 12. Feature Importance

Random Forest importances (`Employment_Status` is split over two one-hot columns):

| Feature | Importance |
|---|---|
| `Credit_Score` | 0.351 |
| `Loan_Amount` | 0.168 |
| `Employment_Status` (both columns combined) | 0.257 |
| `DTI_Ratio` | 0.126 |
| `Income` | 0.098 |

Importance shows what the trained model relied on. It does **not** prove causation. Logistic Regression coefficients (on standardized features) point the same way as the EDA: `Credit_Score` and `Income` positive, `DTI_Ratio` and `Loan_Amount` negative for the *Approved* class.

## 13. Technologies Used

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, Jupyter / Google Colab.

## 14. Project Structure

```text
loan-approval-prediction/
│
├── data/
│   └── loan_data.csv
│
├── notebooks/
│   └── loan_approval_prediction.ipynb
│
├── README.md
└── requirements.txt
```

## 15. How to Run

**Locally**

```bash
git clone https://github.com/zillerehman-dev/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
cd notebooks
jupyter notebook loan_approval_prediction.ipynb
```

Run all cells from top to bottom. The notebook reads `../data/loan_data.csv`.

**Google Colab**: upload the notebook, run it, and upload `loan_data.csv` when prompted (or place it next to the notebook). The Random Forest grid search (48 combinations × 5 folds) can take several minutes on a small CPU.

## 16. Key Learnings

* Fit preprocessing only on training data by using pipelines, and tune with cross-validation on the training set.
* Compare accuracy with a majority-class baseline; on imbalanced data, look at precision, recall and F1 for the minority class.
* Feature importance and coefficients describe the model, not real-world causes.
* Near-perfect results on a likely synthetic dataset call for leakage checks and cautious claims.

## 17. Future Improvements

* Explore the `Text` column (e.g. TF-IDF) and check whether it adds signal.
* Compare other models such as gradient boosting.
* Add ROC-AUC, precision-recall curves and calibration; choose a threshold on validation data.
* Validate on more realistic, noisier data.
* Deployment (planned for a later phase).
