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
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling — refined dark mode: slate / charcoal / warm stone, muted accents
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        --bg-0: #17161A;         /* deep slate base */
        --bg-1: #1D1C21;         /* soft charcoal panel */
        --bg-2: #232227;         /* raised charcoal */
        --stone: #EDE7DD;        /* warm stone text */
        --stone-muted: #9C968A;  /* muted warm-gray */
        --stone-faint: #6E695F;  /* faint warm-gray */
        --hairline: rgba(237,231,221,0.08);
        --hairline-soft: rgba(237,231,221,0.05);
        --accent: #C7A97B;       /* muted warm gold/stone accent */
        --accent-soft: rgba(199,169,123,0.14);
        --sage: #8FA893;         /* muted success */
        --clay: #C08770;         /* muted warning/elevated */
        --rust: #B96A63;         /* muted danger */
        --ease: cubic-bezier(0.22, 1, 0.36, 1);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
        color: var(--stone);
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    /* App background */
    .stApp {
        background:
            radial-gradient(ellipse 900px 500px at 15% -10%, rgba(199,169,123,0.05), transparent),
            var(--bg-0);
    }
    .main .block-container {
        animation: fadeUp 0.55s var(--ease) both;
        padding-top: 2.2rem;
    }

    /* Headings — a quiet serif for a bespoke touch */
    h1, h2, h3 {
        font-family: 'Fraunces', 'Inter', serif !important;
        color: var(--stone) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }
    h4, h5, h6 { color: var(--stone) !important; }
    p, span, label, .stMarkdown { color: var(--stone); }
    small, .stCaption, [data-testid="stCaptionContainer"] { color: var(--stone-muted) !important; }

    hr { border-color: var(--hairline) !important; }

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {
        background: var(--bg-1);
        border-right: 1px solid var(--hairline);
    }
    section[data-testid="stSidebar"] * { color: var(--stone) !important; }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--stone-muted) !important;
    }
    section[data-testid="stSidebar"] hr { border-color: var(--hairline) !important; }

    .brand-mark {
        text-align: left;
        padding: 0.2rem 0 1.1rem 0;
        animation: fadeIn 0.6s var(--ease) both;
    }
    .brand-mark .glyph {
        width: 38px; height: 38px;
        border-radius: 10px;
        background: var(--accent-soft);
        border: 1px solid var(--hairline);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        margin-bottom: 0.65rem;
    }
    .brand-mark .title {
        font-family: 'Fraunces', serif;
        font-size: 1.28rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--stone);
    }
    .brand-mark .subtitle {
        font-size: 0.82rem;
        color: var(--stone-muted);
        margin-top: 0.15rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stAlert"] {
        background: var(--bg-2) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 10px;
        box-shadow: none;
    }

    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: var(--bg-2);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-family: 'Fraunces', serif;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {
        color: var(--stone-muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .legend-box {
        font-size: 0.8rem;
        line-height: 2;
        color: var(--stone-muted);
        border-top: 1px solid var(--hairline);
        padding-top: 0.9rem;
        margin-top: 0.4rem;
    }

    /* ---------------- Hero ---------------- */
    .hero {
        border: 1px solid var(--hairline);
        background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
        border-radius: 16px;
        padding: 2.1rem 2.3rem;
        margin-bottom: 1.7rem;
        animation: fadeUp 0.6s var(--ease) both;
    }
    .hero .eyebrow {
        font-size: 0.76rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.55rem;
        font-weight: 600;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem !important;
        line-height: 1.2;
    }
    .hero p {
        margin: 0.6rem 0 0 0;
        color: var(--stone-muted);
        font-size: 1rem;
        max-width: 680px;
        line-height: 1.55;
    }
    .hero code {
        background: var(--bg-2);
        border: 1px solid var(--hairline);
        color: var(--accent);
        padding: 1px 7px;
        border-radius: 6px;
        font-size: 0.88em;
    }

    /* ---------------- Tabs ---------------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: var(--bg-1);
        border: 1px solid var(--hairline);
        padding: 5px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 9px 18px;
        font-weight: 500;
        color: var(--stone-muted);
        transition: all 0.25s var(--ease);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--stone);
        background: var(--hairline-soft);
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-2) !important;
        color: var(--stone) !important;
        box-shadow: inset 0 0 0 1px var(--hairline);
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------------- Form / cards ---------------- */
    div[data-testid="stForm"] {
        background: var(--bg-1);
        border: 1px solid var(--hairline);
        border-radius: 16px;
        padding: 1.7rem 1.9rem;
        transition: box-shadow 0.3s var(--ease), border-color 0.3s var(--ease);
    }
    div[data-testid="stForm"]:hover {
        border-color: rgba(237,231,221,0.14);
        box-shadow: 0 8px 28px rgba(0,0,0,0.28);
    }

    /* Inputs */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background: var(--bg-2) !important;
        border: 1px solid var(--hairline) !important;
        color: var(--stone) !important;
        border-radius: 9px !important;
        transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease);
    }
    .stSelectbox div[data-baseweb="select"] > div:hover,
    .stNumberInput input:hover {
        border-color: rgba(199,169,123,0.35) !important;
    }
    .stSelectbox div[data-baseweb="select"] > div:focus-within,
    .stNumberInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }
    label, .stSelectbox label, .stNumberInput label {
        color: var(--stone-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-baseweb="popover"] { background: var(--bg-2); }
    ul[role="listbox"] { background: var(--bg-2) !important; border: 1px solid var(--hairline) !important; }
    li[role="option"] { color: var(--stone) !important; }
    li[role="option"]:hover { background: var(--hairline-soft) !important; }

    /* ---------------- Metrics (main area) ---------------- */
    div[data-testid="stMetric"] {
        background: var(--bg-1);
        border: 1px solid var(--hairline);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        transition: transform 0.28s var(--ease), border-color 0.28s var(--ease), box-shadow 0.28s var(--ease);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(199,169,123,0.3);
        box-shadow: 0 10px 26px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--stone-muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--stone) !important;
        font-family: 'Fraunces', serif;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
    }

    /* ---------------- Buttons ---------------- */
    .stButton button, .stFormSubmitButton button {
        background: var(--accent) !important;
        color: #201A10 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.68rem 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 1px 0 rgba(255,255,255,0.15) inset, 0 6px 18px rgba(199,169,123,0.18);
        transition: transform 0.22s var(--ease), box-shadow 0.22s var(--ease), filter 0.22s var(--ease);
    }
    .stButton button:hover, .stFormSubmitButton button:hover {
        transform: translateY(-1.5px);
        filter: brightness(1.06);
        box-shadow: 0 10px 24px rgba(199,169,123,0.26);
    }
    .stButton button:active, .stFormSubmitButton button:active {
        transform: translateY(0);
    }

    .stDownloadButton button {
        background: transparent !important;
        color: var(--stone) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.22s var(--ease);
    }
    .stDownloadButton button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        transform: translateY(-1.5px);
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-2) !important;
        border: 1px dashed var(--hairline) !important;
        border-radius: 12px !important;
        transition: border-color 0.25s var(--ease);
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(199,169,123,0.4) !important;
    }

    /* Progress bar */
    div[data-testid="stProgress"] > div {
        background: var(--bg-2) !important;
        border-radius: 8px;
    }
    div[data-testid="stProgress"] > div > div {
        background: var(--accent) !important;
        border-radius: 8px;
    }

    /* Alerts — muted, not neon */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--hairline);
        animation: fadeUp 0.4s var(--ease) both;
    }
    div[data-testid="stAlert"] p { color: var(--stone) !important; }
    div[data-baseweb="notification"][kind="positive"],
    div.stAlert:has(> div[data-baseweb="notification"][kind="positive"]) {
        background: rgba(143,168,147,0.10) !important;
    }

    /* Expander */
    details[data-testid="stExpander"] {
        background: var(--bg-1);
        border: 1px solid var(--hairline);
        border-radius: 12px;
    }
    details[data-testid="stExpander"] summary {
        color: var(--stone-muted) !important;
        font-weight: 500;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--hairline);
    }

    /* Divider spacing */
    div[data-testid="stVerticalBlock"] > div:has(> hr) { margin: 0.4rem 0; }

    /* Section label above result blocks */
    .section-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--stone-faint);
        font-weight: 600;
        margin: 0.2rem 0 0.7rem 0;
    }

    .about-card {
        background: var(--bg-1);
        border: 1px solid var(--hairline);
        border-radius: 14px;
        padding: 1.4rem 1.5rem;
        height: 100%;
        transition: border-color 0.25s var(--ease), transform 0.25s var(--ease);
    }
    .about-card:hover {
        border-color: rgba(199,169,123,0.25);
        transform: translateY(-2px);
    }
    .about-card h5 {
        font-family: 'Fraunces', serif;
        font-size: 0.95rem;
        color: var(--accent);
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    .about-card p, .about-card li { color: var(--stone-muted); font-size: 0.92rem; line-height: 1.6; }
    .about-card strong { color: var(--stone); }

    .footer-note {
        text-align: center;
        color: var(--stone-faint);
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        margin-top: 3rem;
        padding-top: 1.4rem;
        border-top: 1px solid var(--hairline-soft);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
st.sidebar.markdown(
    """
    <div class="brand-mark">
        <div class="glyph">📉</div>
        <div class="title">Churn Predictor</div>
        <div class="subtitle">Telco customer churn · XGBoost</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model, status_msg = get_model()
if status_msg:
    st.sidebar.success(status_msg)

st.sidebar.divider()
st.sidebar.metric("Decision threshold", f"{DECISION_THRESHOLD:.3f}")
st.sidebar.caption(
    "Threshold tuned on the held-out test set to maximize F1 "
    "(precision 0.59 / recall 0.69 / ROC-AUC 0.845 at this cut)."
)

st.sidebar.markdown(
    """
    <div class="legend-box">
        🔴 High risk &nbsp;·&nbsp; ≥ 70%<br>
        🟠 Elevated &nbsp;·&nbsp; ≥ threshold<br>
        🟡 Watch &nbsp;·&nbsp; ≥ 35%<br>
        🟢 Low risk &nbsp;·&nbsp; below that
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Churn Intelligence</div>
        <h1>Customer Churn Prediction</h1>
        <p>Predict whether a telecom customer is likely to churn, using the trained
        XGBoost pipeline from <code>modelV2.ipynb</code>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.info("👈 Load a trained model in the sidebar to get predictions.")
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(["🔍  Single Prediction", "📁  Batch Prediction", "ℹ️  About"])

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
        st.markdown('<div class="section-label">Prediction result</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="section-label">Batch summary</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows scored", len(results))
            c2.metric("Predicted to churn", n_churn)

            churn_rate = f"{n_churn / len(results):.1%}" if len(results) > 0 else "0.0%"
            c3.metric("Churn rate", churn_rate)

            st.markdown('<div class="section-label" style="margin-top:1.2rem;">Results</div>', unsafe_allow_html=True)
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "⬇️  Download predictions as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ---- About ---------------------------------------------------------------
with tab_about:
    st.subheader("About this app")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            f"""
            <div class="about-card">
                <h5>Model</h5>
                <p>An <strong>XGBoost</strong> classification pipeline trained in
                <strong>modelV2.ipynb</strong>, predicting the probability a
                telecom customer will churn.</p>
                <h5 style="margin-top:1.1rem;">Decision rule</h5>
                <p>A customer is flagged as <strong>Yes (will churn)</strong> when
                the predicted probability is <strong>≥ {DECISION_THRESHOLD:.3f}</strong>,
                the threshold tuned on the held-out test set to maximize F1.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="about-card">
                <h5>Reported test-set performance</h5>
                <p>
                Precision &nbsp;<strong>0.59</strong><br>
                Recall &nbsp;<strong>0.69</strong><br>
                ROC-AUC &nbsp;<strong>0.845</strong>
                </p>
                <h5 style="margin-top:1.1rem;">How to use</h5>
                <p>Use <strong>Single Prediction</strong> for one customer at a
                time, or <strong>Batch Prediction</strong> to score a whole CSV
                of customers at once.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<div class='footer-note'>Built with Streamlit · XGBoost churn pipeline</div>",
    unsafe_allow_html=True,
)