# Loan Approval Prediction

**Machine Learning Classification Project with Streamlit Deployment**

This project predicts whether a loan application is likely to be **Approved** or **Rejected**, based on structured applicant information such as income, credit score, requested loan amount, debt-to-income ratio, and employment status.

---

## Project Overview

This is a **binary classification** problem. Given structured information about a loan
applicant, the model predicts one of two outcomes:

- `Approved`
- `Rejected`

The model is trained on historical loan application data and learns patterns between
applicant attributes (income, credit score, loan amount, debt-to-income ratio, and
employment status) and the final approval decision. Classification is the appropriate
approach here because the target is a discrete, two-class outcome rather than a
continuous value.

Given a new applicant's information, the trained model outputs:

- A predicted class (`Approved` or `Rejected`)
- An approval probability
- A rejection probability

---

## Project Workflow

```text
Problem Definition
        ↓
Dataset Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Preparation
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Comparison
        ↓
Model Evaluation
        ↓
Model Saving
        ↓
Prediction Application
        ↓
Streamlit Deployment
```

- **Problem Definition** — Frame loan approval as a binary classification task.
- **Dataset Collection** — Load the provided applicant dataset (`loan_data.csv`).
- **Data Cleaning** — Check for missing values, duplicates, and correct data types.
- **EDA** — Understand class balance and how features relate to approval outcomes.
- **Feature Preparation** — Select structured features and build a preprocessing pipeline.
- **Train/Test Split** — Hold out 20% of the data, stratified by target, for unbiased evaluation.
- **Model Training** — Train Logistic Regression and Random Forest inside a single pipeline.
- **Model Comparison** — Compare both models on the same held-out test set.
- **Model Evaluation** — Assess accuracy, precision, recall, F1, and the confusion matrix.
- **Model Saving** — Persist the winning pipeline with `joblib`.
- **Prediction Application** — Build a Streamlit app that loads the saved pipeline.
- **Streamlit Deployment** — Prepare the app for deployment (see [Deployment](#deployment)).

---

## Dataset

The dataset contains **24,000 records** and **7 original columns**.

| Feature            | Type        | Description                              | Used in Model |
| ------------------ | ----------- | ----------------------------------------- | :-----------: |
| Income              | Numerical   | Applicant's annual income                | Yes           |
| Credit_Score        | Numerical   | Applicant's credit score                 | Yes           |
| Loan_Amount         | Numerical   | Requested loan amount                    | Yes           |
| DTI_Ratio           | Numerical   | Debt-to-income ratio                     | Yes           |
| Employment_Status   | Categorical | Employment category (`employed`/`unemployed`) | Yes      |
| Text                | Text        | Free-form applicant loan request text    | No            |
| Approval            | Target      | `Approved` or `Rejected`                 | Target        |

**Target mapping:**

```text
Rejected = 0
Approved = 1
```

The `Text` column is not used by the current structured-data model

---

## Target Distribution

```text
Rejected: 83.61%  (20,067 records)
Approved: 16.39%  (3,933 records)
```

The target is **imbalanced**, with roughly 5 rejected applications for every approved
one. Because of this imbalance, accuracy alone can be misleading — a model that always
predicted `Rejected` would already score about 83.6% accuracy without learning anything
useful. This is why precision, recall, and F1 score on the minority (`Approved`) class
are reported alongside accuracy throughout this project.

---

## Data Cleaning and Preprocessing

Checks performed on the raw dataset:

```text
Missing values: 0
Duplicate rows: 0
```

Preprocessing steps:

- The target column was encoded as `Rejected = 0`, `Approved = 1`.
- The `Text` column was excluded from the structured-data model.
- **Numerical features** (`Income`, `Credit_Score`, `Loan_Amount`, `DTI_Ratio`) were
  standardized with `StandardScaler`.
- **Categorical feature** (`Employment_Status`) was one-hot encoded with `OneHotEncoder`.
- All preprocessing was implemented inside a single scikit-learn `ColumnTransformer` /
  `Pipeline`, rather than as separate manual steps, so the exact same transformation is
  applied automatically at prediction time.

---

## Train/Test Split

```text
Training samples: 19,200 (80%)
Testing samples:   4,800 (20%)
random_state = 42
stratify = y
```

Stratified splitting was used so that both the training and test sets preserve the same
~83.6% / 16.4% class balance as the full dataset, giving a fair and representative
evaluation.

---

## Leakage Prevention

- All preprocessing (scaling, encoding) is fitted **only** on the training data, inside
  the pipeline — never on the full dataset before splitting.
- The test set was held out and only used once, for final evaluation.
- The target column (`Approval`) was never included as an input feature.
- The `Text` column was excluded entirely from the structured model.
- Because preprocessing and the classifier live in one `Pipeline` object, the saved
  model reproduces the exact training-time transformation on any new data — there is no
  separate, hand-written preprocessing code to drift out of sync.

---

## Exploratory Data Analysis

**Credit Score** — Approved applicants had a substantially higher average credit score.

```text
Approved: ~702.21
Rejected: ~550.93
```

**Income** — Approved applicants had a higher average income.

```text
Approved: ~126,219
Rejected: ~107,273
```

**DTI Ratio** — Approved applicants had a lower average debt-to-income ratio.

```text
Approved: ~22.15
Rejected: ~37.18
```

**Loan Amount** — Approved applicants requested, on average, a smaller loan.

```text
Approved: ~37,664
Rejected: ~45,668
```

**Employment Status** — Applicants with `unemployed` status were approved far less
often than `employed` applicants in this dataset.

**Correlation with the target** (`Approval`, encoded 0/1):

```text
Credit_Score   +0.352
Income         +0.136
Loan_Amount    -0.085
DTI_Ratio      -0.172
```

These are simple linear correlations and do not by themselves imply causation — they
describe the association observed in this dataset.

---

## Models Used

### Logistic Regression (baseline)

A simple, interpretable linear baseline.

```text
max_iter = 1000
class_weight = "balanced"
random_state = 42
```

### Random Forest (final model)

A tree-based ensemble model, better suited to the non-linear relationships in this
dataset.

```text
n_estimators = 400
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
class_weight = None
random_state = 42
```

No other model types were trained for this project.

---

## Model Selection

Both models were trained on the same 19,200-row training set and evaluated on the same
held-out 4,800-row test set, using identical preprocessing.

Logistic Regression achieved reasonable recall on the minority class but at a
significant cost to precision, meaning it flagged many `Rejected` applicants as
`Approved`. Random Forest outperformed it across **every** metric — accuracy,
precision, recall, and F1 — without sacrificing either class. Because of this
consistent, across-the-board improvement, **Random Forest was selected as the final
model** used in the prediction application.

---

## Model Evaluation

Test-set results (4,800 held-out samples):

| Model                | Accuracy | Precision | Recall | F1 Score |
| --------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression   |   90.19% |    63.34% | 95.30% |   76.10% |
| **Random Forest**     | **99.65%** | **98.49%** | **99.36%** | **98.92%** |
| Majority-class baseline (always predict "Rejected") | 83.61% | — | — | — |

Comparing both models against the majority-class baseline shows that Logistic
Regression already improves meaningfully over "always guessing Rejected," and Random
Forest improves substantially further — while also keeping precision and recall
balanced on the minority `Approved` class, not just overall accuracy.

---

## Confusion Matrix

Random Forest, evaluated on the 4,800-row test set:

|                     | Predicted: Rejected | Predicted: Approved |
| ------------------- | -------------------: | -------------------: |
| **Actual: Rejected** | 4,001 (TN)           | 12 (FP)               |
| **Actual: Approved** | 5 (FN)                | 782 (TP)              |

Out of 4,800 test applicants, the model misclassified 17 total (12 false approvals, 5
false rejections).


---

## Feature Importance

Random Forest feature importances (higher = more influence on the model's decisions
within this dataset):

```text
Credit_Score                 0.354
Loan_Amount                  0.168
Employment_Status (combined) 0.253
DTI_Ratio                    0.127
Income                       0.098
```

`Credit_Score` and `Employment_Status` are the most influential features, consistent
with the patterns seen in the EDA section. This reflects how the trained model used
these features on this specific dataset — it is not a causal claim about real-world
loan approval.

---

## Prediction Application

The trained Random Forest pipeline (preprocessing + classifier, saved as a single
object) is loaded directly into a Streamlit application. The app does **not** retrain
the model when it starts — it only loads the already-trained pipeline.

The application collects:

```text
Income
Credit Score
Loan Amount
DTI Ratio
Employment Status
```

and returns:

```text
Predicted class (Approved / Rejected)
Approval probability
Rejection probability
```

Predictions are generated using:

```python
model.predict(input_df)
model.predict_proba(input_df)
```

The app also includes three example test cases that run through the real model (not
hardcoded results), an applicant summary, and expandable "About the Model" and "Model
Performance" sections.

---

## Application Screenshots

### Applicant Input

![Applicant Input](images/app_input.png)

### Approved Prediction

![Approved Prediction](images/approved_prediction.png)

### Rejected Prediction

![Rejected Prediction](images/rejected_prediction.png)

*(Add these screenshots after running the app locally — see [Reproducibility](#reproducibility).)*

---

## Deployment

The application is prepared for deployment. The live deployment link will be added
after deployment.

To run it locally in the meantime:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```text
loan_approval_prediction/
│
├── app.py                          # Streamlit prediction application
├── loan_approval_prediction.ipynb  # Full training & evaluation notebook
├── loan_approval_model.pkl         # Saved, trained scikit-learn pipeline
├── requirements.txt
├── README.md
├── data/
│   └── loan_data.csv
```

---

## Technologies Used

```text
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Joblib
Jupyter Notebook
Streamlit
```

---

## Reproducibility

### Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd loan_approval_prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

### Notebook

`loan_approval_prediction.ipynb` contains the full workflow — data loading, cleaning
checks, EDA, the preprocessing pipeline, training and evaluation of both models, the
comparison table, confusion matrix, feature importance, and the code that saves the
final pipeline to `loan_approval_model.pkl`.

> **Note on reproducing the model file:** a `.pkl` file is tied to the exact
> scikit-learn version that created it. If you re-run the notebook, run it in the same
> environment you'll use to run `app.py`, so the saved pipeline stays loadable.

---

## Limitations

- The current model uses structured features only.
- The `Text` column is not currently included in the model.
- The dataset appears to contain strong, fairly separable patterns (for example,
  `Employment_Status = unemployed` applicants are approved very rarely), which likely
  contributes to the very high Random Forest test performance. The very high test
  performance may be influenced by strong or rule-like patterns in the dataset.
- High test-set performance should not automatically be interpreted as real-world
  lending performance.
- The model has not been validated on an independent, real-world lending dataset.
- This application is an educational machine learning project.
- Predictions should not be treated as financial or lending decisions.

---

## Ethical / Responsible Use

This application is built for **educational and demonstration purposes**. A model
prediction from this project should not, by itself, determine whether a person
receives a real loan. Real-world lending systems require additional validation,
fairness analysis, regulatory compliance, ongoing monitoring, and real-world testing
that are outside the scope of this project.

---

## Future Improvements

```text
- Incorporate the Text feature using TF-IDF or NLP techniques
- Evaluate additional classification algorithms
- Perform ROC-AUC and Precision-Recall analysis
- Improve probability calibration
- Test on more realistic external data
- Add model monitoring
- Improve deployment
```
