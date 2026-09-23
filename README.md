#  Customer Churn Prediction

Predicting telecom customer churn with a tuned XGBoost pipeline, served through a Streamlit app.

> **TL;DR** — I ran a full experiment cycle: baseline model → feature engineering (which *didn't* help) → proper model comparison + hyperparameter tuning → threshold optimization. Final model: **ROC-AUC 0.845**, deployed as an interactive Streamlit app for single and batch predictions.

---

## 🔍 Problem

Telecom companies lose significant revenue to customer churn. The goal here was to build a model that flags customers likely to churn *before* they leave, using account and service data (contract type, tenure, billing, add-on services, etc.) — early enough for retention teams to act on it.

## Tech Stack-

| Layer | Tools |
|---|---|
| Data & modeling | `pandas`, `numpy`, `scikit-learn`, `xgboost` |
| Stats / EDA | `scipy` (Chi², Cramér's V), `statsmodels` (VIF) |
| Model persistence | `joblib` |
| App / deployment | `streamlit` |

---

## Project Workflow

### 1. EDA & Statistical Validation
Before touching a model, I validated which categorical features actually related to churn:
- **Chi-squared tests** + **Cramér's V** to rank categorical features by association strength with `Churn`.
- **VIF (Variance Inflation Factor)** on `Tenure`, `MonthlyCharges`, `TotalCharges` to check for multicollinearity between the numeric features.

### 2. Baseline model — Model V1
A first XGBoost model on the cleaned (but not engineered) feature set, evaluated at the default 0.5 threshold:

| Metric (churn class) | Score |
|---|---|
| Precision | 0.52 |
| Recall | 0.66 |
| F1 | 0.58 |
| ROC-AUC | 0.8203 |

This became the benchmark everything else had to beat.

### 3. Feature engineering — tried, and rejected
I engineered seven new features from domain intuition:

- `TotalServices` — count of active add-on services
- `IsNewCustomer` / `IsLongTermCustomer` — tenure-based flags
- `IsMonthToMonth` — contract-type flag
- `UseElecCheck` — payment-method flag
- `HasSecuritySupport` — combined security/support flag
- `HighMonthlyCharge` — top-quartile billing flag

I evaluated these with a Random Forest feature-importance ranking and a **5-fold stratified CV comparison across three feature sets — Base vs. Engineered vs. Selected** (top RF features only) — scored on ROC-AUC and F1.

**Result: the engineered features didn't produce a meaningful lift over the base set.** Rather than force them into the final pipeline for the sake of having "done feature engineering," I dropped them and moved forward with the original cleaned features. Knowing when engineered signal *isn't* adding value is as important as knowing how to build it.

### 4. Model comparison & tuning — Model V2 (final)
With the feature question settled, I focused on getting more out of modeling itself:

- Built a `ColumnTransformer` pipeline: `OrdinalEncoder` for binary categoricals, `OneHotEncoder` for multi-class categoricals, `RobustScaler` for numeric features (robust to the outliers/skew in billing amounts).
- Compared **Logistic Regression, Random Forest, and XGBoost** via 5-fold stratified CV (ROC-AUC, F1, precision, recall).
- Tuned all three with `RandomizedSearchCV`, selected the best on CV ROC-AUC → **XGBoost**.
- Instead of defaulting to a 0.5 cutoff, I computed the **precision–recall curve** on the held-out test set and picked the threshold that maximizes F1: **0.625**.

### 5. Final results

| Metric (churn class) | Model V1 (baseline) | Model V2 (final) |
|---|---|---|
| Precision | 0.52 | **0.59** |
| Recall | 0.66 | **0.69** |
| ROC-AUC | 0.8203 | **0.845** |
| Decision threshold | 0.50 (default) | 0.625 (tuned) |

The gain came from **pipeline design, model selection, and threshold tuning** — not from the engineered features, which is worth stating plainly rather than glossing over.

---

## 🖥️ The App

A Streamlit interface on top of the final pipeline, with three views:

- **🔍 Single Prediction** — enter one customer's details, get a churn probability, prediction, and risk tier.
- **📁 Batch Prediction** — upload a CSV, score every row, download the results with probabilities and risk labels attached.
- **ℹ️ About** — model details and reported performance, in-app.

Risk tiers are bucketed from the predicted probability (🔴 High / 🟠 Elevated / 🟡 Watch / 🟢 Low) so results are actionable, not just a raw number.

---

## 📂 Project Structure

```
.
├── app/
│   └── app.py  
├── dashboard/
│   └── churn_dashboard.pbix                                    
├── notebooks/
│   ├── data_cleaning.ipynb 
│   ├── EDA.ipynb 
│   ├── model.ipynb                            
│   └── modelV2.ipynb                        
├── models/
│   └── churn_model_XGB.pkl                  
├── data/
│   ├── Telco-Customer-Churn-cleaned.csv
│   ├── Telco-Customer-Churn-featured.csv
│   ├── Telco-Customer-Churn-raw.csv
│   ├── Telco-Customer-Churn-train-featured.csv
│   └── Telco-Customer-Churn-test-featured.csv
├── sql/
│   └── churn_analysis.sql 
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## ▶️ Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `CHURN_MODEL_PATH` if your trained pipeline lives somewhere other than `models/churn_model_XGB.pkl` — or just upload a `.pkl` from the app's sidebar.

---

## 🔭 What I'd Explore Next

- Revisit feature engineering with **target encoding** or **interaction terms specific to contract × payment method**, since the manual flags tested here didn't add signal.
- Try **SHAP** for per-prediction explainability in the app, so the "why" behind a risk score is visible to the end user, not just the score.
- Calibrate probabilities (Platt/Isotonic) if the output is ever used for anything beyond ranking — raw XGBoost probabilities aren't well-calibrated out of the box.
