from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "churn_model_XGB.pkl"

# Threshold picked in modelV2.ipynb (best F1 on the precision-recall curve).
# It is NOT stored in the .pkl, so it lives here as a constant.
DEFAULT_THRESHOLD = 0.625

LABELS = {
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has a partner",
    "Dependents": "Has dependents",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "TechSupport": "Tech support",
    "Contract": "Contract",
    "PaymentMethod": "Payment method",
    "PaperlessBilling": "Paperless billing",
}


FALLBACK_OPTIONS = {
    "SeniorCitizen": [0, 1],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "OnlineSecurity": ["No", "Yes"],
    "TechSupport": ["No", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": ["Bank transfer", "Credit card", "Electronic check", "Mailed check"],
    "Contract": ["Month-to-month", "One year", "Two year"],
}

st.set_page_config(page_title="Customer churn risk", layout="wide")


@st.cache_resource
def load_model():
    """Load the saved pipeline. Also accepts a {'model', 'threshold'} dict."""
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        return bundle["model"], float(bundle.get("threshold", DEFAULT_THRESHOLD))
    return bundle, DEFAULT_THRESHOLD


def read_options(model):
    """Read the category values the encoders saw in training, so the
    dropdowns always match the training data exactly."""
    options = dict(FALLBACK_OPTIONS)
    try:
        preprocessor = model.named_steps["preprocessor"]
        for name, transformer, columns in preprocessor.transformers_:
            if name not in ("bin", "multi"):
                continue
            categories = transformer.named_steps["encoder"].categories_
            for column, values in zip(columns, categories):
                options[column] = [v.item() if hasattr(v, "item") else v for v in values]
    except Exception:
        pass  # keep the fallback options
    return options


if not MODEL_PATH.exists():
    st.error(f"Model file not found at `{MODEL_PATH}`. Save the trained pipeline there and reload.")
    st.stop()

model, saved_threshold = load_model()
OPTIONS = read_options(model)


def pick(container, column):
    """Dropdown for one categorical column."""
    if column == "SeniorCitizen":
        fmt = lambda v: "Yes" if v == 1 else "No"
    else:
        fmt = str
    return container.selectbox(LABELS[column], OPTIONS[column], format_func=fmt, key=column)


with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(saved_threshold),
        step=0.005,
        format="%.3f",
        help="A customer is flagged when their churn probability is at or above this value.",
    )
    st.caption(
        "Lower it to catch more churners at the cost of more false alarms. "
        "Raise it to flag fewer customers with higher confidence."
    )

    with st.expander("About this model"):
        classifier_name = type(model.named_steps["classifier"]).__name__
        st.markdown(
            f"""
**Model:** {classifier_name}, tuned with 5-fold cross-validation.

**Test-set results** at threshold {DEFAULT_THRESHOLD}:
- ROC-AUC: about 0.845
- Churners caught (recall): 69%
- Flagged churners that really churn (precision): 59%

The threshold was chosen on the test set, so these figures are slightly optimistic.
"""
        )

# ------------------------------------------------------------------- main
st.title("Customer churn risk")
st.write("Enter a customer's details to estimate how likely they are to leave.")

left, right = st.columns([3, 2], gap="large")
inputs = {}

with left:
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)
    inputs["SeniorCitizen"] = pick(c1, "SeniorCitizen")
    inputs["Partner"] = pick(c2, "Partner")
    inputs["Dependents"] = pick(c3, "Dependents")

    st.subheader("Services")
    c1, c2, c3 = st.columns(3)
    inputs["InternetService"] = pick(c1, "InternetService")
    inputs["OnlineSecurity"] = pick(c2, "OnlineSecurity")
    inputs["TechSupport"] = pick(c3, "TechSupport")

    st.subheader("Contract and billing")
    c1, c2, c3 = st.columns(3)
    inputs["Contract"] = pick(c1, "Contract")
    inputs["PaymentMethod"] = pick(c2, "PaymentMethod")
    inputs["PaperlessBilling"] = pick(c3, "PaperlessBilling")

    c1, c2 = st.columns(2)
    inputs["Tenure"] = c1.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    inputs["MonthlyCharges"] = c2.number_input(
        "Monthly charges", min_value=0.0, max_value=200.0, value=70.0, step=0.05, format="%.2f"
    )

    estimate_total = st.checkbox("Estimate total charges as tenure × monthly charges", value=True)
    if estimate_total:
        inputs["TotalCharges"] = float(inputs["Tenure"] * inputs["MonthlyCharges"])
        st.caption(f"Total charges used: {inputs['TotalCharges']:,.2f}")
    else:
        inputs["TotalCharges"] = st.number_input(
            "Total charges",
            min_value=0.0,
            value=float(inputs["Tenure"] * inputs["MonthlyCharges"]),
            step=10.0,
            format="%.2f",
        )

row = pd.DataFrame([inputs])
expected = getattr(model.named_steps["preprocessor"], "feature_names_in_", None)
if expected is not None:
    # Columns the model was fitted on but doesn't use (e.g. Gender) are dropped by
    # the preprocessor, so leaving them empty is harmless.
    row = row.reindex(columns=expected)

probability = float(model.predict_proba(row)[0, 1])
flagged = probability >= threshold

with right:
    st.subheader("Prediction")
    st.metric("Probability of churn", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))

    if flagged:
        st.error(f"High risk: this customer is above the {threshold:.3f} threshold and is flagged as likely to churn.")
    else:
        st.success(f"Lower risk: this customer is below the {threshold:.3f} threshold and is not flagged.")

    gap = probability - threshold
    st.caption(f"{abs(gap):.1%} points {'above' if gap >= 0 else 'below'} the threshold.")

    with st.expander("Input sent to the model"):
        st.dataframe(pd.DataFrame([inputs]).T.rename(columns={0: "Value"}))