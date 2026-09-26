"""
Loan Approval Prediction System

This app loads the already-trained scikit-learn Pipeline (preprocessing + Random
Forest classifier) and uses it to predict loan approval for user-entered
applicant information. No training happens here.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "loan_approval_model.pkl"

NUMERIC_FEATURES = ["Income", "Credit_Score", "Loan_Amount", "DTI_Ratio"]
CATEGORICAL_FEATURES = ["Employment_Status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Exact categories seen in the training data — do not invent new ones.
EMPLOYMENT_STATUS_OPTIONS = ["employed", "unemployed"]

TEST_CASES = {
    "Test Case 1 — Strong Applicant": {
        "Income": 150000,
        "Credit_Score": 760,
        "Loan_Amount": 20000,
        "DTI_Ratio": 12.5,
        "Employment_Status": "employed",
    },
    "Test Case 2 — Higher Risk Applicant": {
        "Income": 45000,
        "Credit_Score": 420,
        "Loan_Amount": 90000,
        "DTI_Ratio": 65.0,
        "Employment_Status": "employed",
    },
    "Test Case 3 — Unemployed Applicant": {
        "Income": 80000,
        "Credit_Score": 690,
        "Loan_Amount": 25000,
        "DTI_Ratio": 20.0,
        "Employment_Status": "unemployed",
    },
}

TEST_SET_METRICS = {
    "Accuracy": "99.65%",
    "Precision": "98.49%",
    "Recall": "99.36%",
    "F1 Score": "98.92%",
}

# ----------------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained scikit-learn pipeline from disk."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def create_input_dataframe(income, credit_score, loan_amount, dti_ratio, employment_status):
    """Build a single-row DataFrame with exactly the columns the pipeline expects."""
    return pd.DataFrame({
        "Income": [income],
        "Credit_Score": [credit_score],
        "Loan_Amount": [loan_amount],
        "DTI_Ratio": [dti_ratio],
        "Employment_Status": [employment_status],
    })


def make_prediction(model, input_df):
    """Run the trained pipeline on a single applicant row.

    Returns (predicted_class, approval_probability, rejection_probability) or
    (None, None, None) on failure. Class 1 = Approved, 0 = Rejected.
    """
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        # predict_proba columns follow model.classes_ order; map explicitly.
        classes = list(model.classes_)
        approval_prob = probabilities[classes.index(1)]
        rejection_prob = probabilities[classes.index(0)]
        return prediction, approval_prob, rejection_prob
    except Exception as e:
        st.error(f"Something went wrong while generating the prediction: {e}")
        return None, None, None


def validate_inputs(income, credit_score, loan_amount, dti_ratio):
    """Return a list of human-readable validation error messages (empty if valid)."""
    errors = []
    if income is None or income <= 0:
        errors.append("Annual income must be a positive number.")
    if credit_score is None or not (300 <= credit_score <= 850):
        errors.append("Credit score must be between 300 and 850.")
    if loan_amount is None or loan_amount <= 0:
        errors.append("Requested loan amount must be a positive number.")
    if dti_ratio is None or dti_ratio < 0:
        errors.append("Debt-to-income ratio cannot be negative.")
    return errors


def display_prediction(prediction, approval_prob, rejection_prob):
    """Render the prediction result section."""
    st.markdown("### Loan Prediction")

    if prediction == 1:
        st.success("### ✅ APPROVED")
    else:
        st.error("### ❌ REJECTED")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Approval Probability", f"{approval_prob * 100:.2f}%")
        st.progress(float(approval_prob))
    with col2:
        st.metric("Rejection Probability", f"{rejection_prob * 100:.2f}%")
        st.progress(float(rejection_prob))


# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💰",
    layout="centered",
)

st.title("Loan Approval Prediction System")
st.caption("Machine Learning powered loan approval prediction")
st.divider()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("About the Model")
    st.markdown(
        """
        - **Model:** Random Forest Classifier
        - **Task:** Binary Classification
        - **Features:** 5
        - **Dataset:** Loan Approval Dataset
        """
    )

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
model = load_model()

if model is None:
    st.error(
        "Model file not found. Please make sure `loan_approval_model.pkl` is in the "
        "same folder as `app.py`, then restart the app."
    )
    st.stop()

# Use session_state so "Load Example" can populate the form inputs.
defaults = {
    "Income": 100000,
    "Credit_Score": 650,
    "Loan_Amount": 25000,
    "DTI_Ratio": 25.0,
    "Employment_Status": "employed",
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# ----------------------------------------------------------------------------
# Example test cases
# ----------------------------------------------------------------------------
st.subheader("Example Test Cases")
st.caption("Load a preset applicant profile, then run it through the model below.")

tc_cols = st.columns(len(TEST_CASES))
for col, (name, values) in zip(tc_cols, TEST_CASES.items()):
    with col:
        st.markdown(f"**{name.split('—')[0].strip()}**")
        st.caption(name.split("—")[1].strip() if "—" in name else "")
        if st.button("Load Example", key=f"load_{name}"):
            for k, v in values.items():
                st.session_state[k] = v
            st.rerun()

st.divider()

# ----------------------------------------------------------------------------
# Applicant Information Form
# ----------------------------------------------------------------------------
st.subheader("Applicant Information")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input(
        "Annual Income",
        min_value=0,
        value=int(st.session_state["Income"]),
        step=1000,
        help="Applicant's total annual income.",
    )
    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0,
        value=int(st.session_state["Loan_Amount"]),
        step=500,
        help="The loan amount the applicant is requesting.",
    )
with col2:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=int(st.session_state["Credit_Score"]),
        step=1,
        help="Credit score between 300 and 850.",
    )
    dti_ratio = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        value=float(st.session_state["DTI_Ratio"]),
        step=0.1,
        format="%.2f",
        help="Debt-to-income ratio as a percentage.",
    )

employment_status = st.selectbox(
    "Employment Status",
    options=EMPLOYMENT_STATUS_OPTIONS,
    index=EMPLOYMENT_STATUS_OPTIONS.index(st.session_state["Employment_Status"]),
)

predict_clicked = st.button("Predict Loan Approval", type="primary", use_container_width=True)

st.divider()

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if predict_clicked:
    errors = validate_inputs(income, credit_score, loan_amount, dti_ratio)

    if errors:
        for err in errors:
            st.error(err)
    else:
        input_df = create_input_dataframe(income, credit_score, loan_amount, dti_ratio, employment_status)
        prediction, approval_prob, rejection_prob = make_prediction(model, input_df)

        if prediction is not None:
            display_prediction(prediction, approval_prob, rejection_prob)

            st.markdown("### Applicant Summary")
            summary_df = pd.DataFrame({
                "Field": ["Income", "Credit Score", "Loan Amount", "DTI Ratio", "Employment Status"],
                "Value": [f"{income:,}", credit_score, f"{loan_amount:,}", f"{dti_ratio:.2f}", employment_status],
            })
            st.table(summary_df.set_index("Field"))

st.divider()

# ----------------------------------------------------------------------------
# About the model / performance
# ----------------------------------------------------------------------------
with st.expander("About the Model"):
    st.markdown(
        """
        This application uses a **Random Forest classification model** trained on
        structured loan applicant information.

        The model uses:
        - Income
        - Credit Score
        - Loan Amount
        - DTI Ratio
        - Employment Status

        The `Text` field from the original dataset is not used in this version of the model.
        """
    )

with st.expander("Model Performance"):
    cols = st.columns(len(TEST_SET_METRICS))
    for col, (metric, value) in zip(cols, TEST_SET_METRICS.items()):
        col.metric(metric, value)
    st.caption(
        "These metrics were measured on the held-out test set during model evaluation. "
        "They do not represent a guarantee for individual predictions."
    )

st.caption(
    "This tool is for educational purposes only and is not suitable for "
    "real-world lending decisions."
)
