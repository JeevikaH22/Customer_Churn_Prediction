import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = os.getenv("CHURN_MODEL_PATH", "models/churn_model_XGB.pkl")
DECISION_THRESHOLD = 0.625  
FEATURE_COLUMNS = [
    "Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure",
    "PhoneService", "InternetService", "OnlineSecurity", "TechSupport",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges",
]

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def get_model():
    """Load the model from MODEL_PATH, or let the user upload a .pkl instead."""
    model = load_model(MODEL_PATH)
    if model is not None:
        return model, f"Loaded model from `{MODEL_PATH}`"

    st.sidebar.warning(
        f"No model found at `{MODEL_PATH}`.\n\n"
        "Set the `CHURN_MODEL_PATH` environment variable, place your "
        "`churn_model_XGB.pkl` there, or upload it below."
    )
    uploaded = st.sidebar.file_uploader("Upload trained pipeline (.pkl)", type=["pkl"])
    if uploaded is not None:
        model = joblib.load(uploaded)
        return model, "Loaded model from uploaded file"
    return None, None


# ----------------------------------------------------------------------------
# Prediction helpers
# ----------------------------------------------------------------------------
def predict_df(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run the pipeline on a raw dataframe and attach probability / label columns."""
    X = df[FEATURE_COLUMNS].copy()

    # Coerce numeric columns in case string/blank spaces exist in uploaded CSVs
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0.0)
    X["Tenure"] = pd.to_numeric(X["Tenure"], errors="coerce").fillna(0)
    X["MonthlyCharges"] = pd.to_numeric(X["MonthlyCharges"], errors="coerce").fillna(0.0)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    out = df.copy()
    out["ChurnProbability"] = proba.round(4)
    out["ChurnPrediction"] = np.where(pred == 1, "Yes", "No")
    return out


def risk_bucket(p: float) -> str:
    if p >= 0.7:
        return "🔴 High risk"
    if p >= DECISION_THRESHOLD:
        return "🟠 Elevated risk"
    if p >= 0.35:
        return "🟡 Watch"
    return "🟢 Low risk"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("📉 Churn Predictor")
st.sidebar.caption("Telco customer churn — XGBoost pipeline")

model, status_msg = get_model()
if status_msg:
    st.sidebar.success(status_msg)

st.sidebar.divider()
st.sidebar.metric("Decision threshold", f"{DECISION_THRESHOLD:.3f}")
st.sidebar.caption(
    "Threshold tuned on the held-out test set to maximize F1 "
    "(precision 0.59 / recall 0.69 / ROC-AUC 0.845 at this cut)."
)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("Customer Churn Prediction")
st.write(
    "Predict whether a telecom customer is likely to churn, using the trained "
    "XGBoost pipeline from `modelV2.ipynb`."
)

if model is None:
    st.info("👈 Load a trained model in the sidebar to get predictions.")
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction", "ℹ️ About"])

# ---- Single prediction -------------------------------------------------
with tab_single:
    st.subheader("Enter customer details")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col3:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5, format="%.2f")
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, value=float(tenure) * 70.0, step=0.5, format="%.2f"
            )

        submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "Gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure": tenure,
            "PhoneService": phone_service,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        result = predict_df(model, row).iloc[0]
        proba = float(result["ChurnProbability"])
        label = result["ChurnPrediction"]

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediction", "Will Churn" if label == "Yes" else "Will Stay")
        r2.metric("Churn Probability", f"{proba:.1%}")
        r3.metric("Risk Level", risk_bucket(proba))

        st.progress(min(max(proba, 0.0), 1.0))

        if label == "Yes":
            st.error(
                f"⚠️ This customer is predicted to **churn** "
                f"(probability {proba:.1%} ≥ threshold {DECISION_THRESHOLD:.3f})."
            )
        else:
            st.success(
                f"✅ This customer is predicted to **stay** "
                f"(probability {proba:.1%} < threshold {DECISION_THRESHOLD:.3f})."
            )

        with st.expander("Show input passed to the model"):
            st.dataframe(row, use_container_width=True)

# ---- Batch prediction ---------------------------------------------------
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.caption(
        "Upload a CSV with (at least) these columns: "
        + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS)
    )

    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    if csv_file is not None:
        try:
            batch_df = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
        if missing:
            st.error(f"CSV is missing required columns: {', '.join(missing)}")
        else:
            results = predict_df(model, batch_df)
            results["RiskLevel"] = results["ChurnProbability"].apply(risk_bucket)

            n_churn = int((results["ChurnPrediction"] == "Yes").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows scored", len(results))
            c2.metric("Predicted to churn", n_churn)
            
            churn_rate = f"{n_churn / len(results):.1%}" if len(results) > 0 else "0.0%"
            c3.metric("Churn rate", churn_rate)

            st.dataframe(results, use_container_width=True)

            st.download_button(
                "⬇️ Download predictions as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )