import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_salary_regressor.pkl"
DATA_PATH = ROOT / "job_salary_prediction_dataset.csv"
METRICS_PATH = ROOT / "model_metrics.json"
COMPARISON_PATH = ROOT / "model_comparison.json"
AUDIT_PATH = ROOT / "model_audit.json"

st.set_page_config(page_title="SmartPay | Salary Intelligence", page_icon="💼", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .stApp { background: #0b1220; }
    [data-testid="stHeader"] { background: rgba(11, 18, 32, 0.85); }
    .hero { padding: 1.5rem 0 1rem; }
    .hero h1 { color: #f4f8ff; font-size: 2.7rem; margin: 0; letter-spacing: -1px; }
    .hero p { color: #9fb1c9; font-size: 1.05rem; margin-top: .35rem; }
    .status-card { background: #132238; border: 1px solid #2f8ef6; border-radius: 14px; padding: 1rem 1.2rem; }
    .status-card strong { color: #8be9fd; }
    .status-card .verified { color: #7cf0bf; }
    .status-card .pending { color: #ffd166; }
    .section-card { background: #111c2e; border: 1px solid #243753; border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
    div[data-testid="stMetric"] { background: #132238; border: 1px solid #243753; border-radius: 12px; padding: .8rem; }
    div[data-testid="stMetricValue"] { color: #f4f8ff; }
    .result-box { background: linear-gradient(135deg, #1f4f77, #17304d); border: 1px solid #50b8ff; border-radius: 14px; padding: 1.4rem; text-align: center; color: white; font-size: 1.8rem; font-weight: 700; }
    .small-note { color: #9fb1c9; font-size: .85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("The trained model or dataset is missing. Run every cell in the training notebook first.")
    st.stop()

MODEL = joblib.load(MODEL_PATH)
METRICS = load_json(METRICS_PATH, {})
MODEL_RESULTS = load_json(COMPARISON_PATH, METRICS.get("model_results", []))
AUDIT = load_json(AUDIT_PATH, {})
DATASET = pd.read_csv(DATA_PATH)

st.markdown(
    """
    <div class="hero">
      <h1>💼 SmartPay Salary Intelligence</h1>
      <p>Evidence-based salary prediction with model diagnostics, batch scoring, and transparent evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

leakage_free = AUDIT.get("leakage_free")
fit_diagnosis = AUDIT.get("fit_diagnosis", "Run the training notebook to generate fit diagnostics.")
model_name = METRICS.get("best_model", "Trained salary model")

status_label = "Verified training run" if leakage_free is True else "Training audit pending"
status_class = "verified" if leakage_free is True else "pending"
st.markdown(
    f"<div class='status-card'><strong class='{status_class}'>{status_label}</strong> &nbsp;|&nbsp; Model: {model_name} &nbsp;|&nbsp; {fit_diagnosis}</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("💾 Candidate profile")
    st.caption("Choose a profile and generate an estimate.")
    job_titles = sorted(DATASET["job_title"].dropna().astype(str).unique())
    industries = sorted(DATASET["industry"].dropna().astype(str).unique())
    locations = sorted(DATASET["location"].dropna().astype(str).unique())
    company_sizes = sorted(DATASET["company_size"].dropna().astype(str).unique())
    education_levels = sorted(DATASET["education_level"].dropna().astype(str).unique())
    remote_modes = ["No", "Hybrid", "Yes"]
    job_title = st.selectbox("Job title", job_titles)
    experience_years = st.slider("Experience (years)", 0, 40, 5)
    education_level = st.selectbox("Education", education_levels)
    skills_count = st.slider("Skills count", 0, 30, 10)
    industry = st.selectbox("Industry", industries)
    company_size = st.selectbox("Company size", company_sizes)
    location = st.selectbox("Location", locations)
    remote_work = st.selectbox("Remote work", remote_modes)
    certifications = st.slider("Certifications", 0, 15, 2)
    predict_clicked = st.button("🚀 Estimate salary", type="primary", use_container_width=True)


def prepare_input_dataframe(frame):
    prepared = frame.copy()
    for column in ["job_title", "education_level", "industry", "company_size", "location", "remote_work"]:
        prepared[column] = prepared[column].astype("string").str.strip().str.title()
    prepared["experience_level"] = pd.cut(
        prepared["experience_years"],
        bins=[0, 3, 7, 12, 20, float("inf")],
        labels=["Junior", "Mid", "Experienced", "Senior", "Lead"],
        right=False,
    )
    prepared["skills_experience_ratio"] = (prepared["skills_count"] + 1) / (prepared["experience_years"] + 1)
    prepared["remote_binary"] = prepared["remote_work"].map({"No": 0, "Hybrid": 1, "Yes": 1}).fillna(0)
    return prepared


MODEL_FEATURES = [
    "job_title", "experience_years", "education_level", "skills_count", "industry",
    "company_size", "location", "remote_work", "certifications", "experience_level",
    "skills_experience_ratio", "remote_binary",
]


def predict(frame):
    prepared = prepare_input_dataframe(frame)
    return MODEL.predict(prepared[MODEL_FEATURES])


profile = pd.DataFrame([{
    "job_title": job_title,
    "experience_years": experience_years,
    "education_level": education_level,
    "skills_count": skills_count,
    "industry": industry,
    "company_size": company_size,
    "location": location,
    "remote_work": remote_work,
    "certifications": certifications,
}])

if predict_clicked:
    prediction = float(predict(profile)[0])
    st.session_state["latest_prediction"] = prediction

if "latest_prediction" in st.session_state:
    st.markdown(f"<div class='result-box'>Estimated annual salary<br>₹{st.session_state['latest_prediction']:,.2f}</div>", unsafe_allow_html=True)

st.divider()
tab_predict, tab_batch, tab_quality = st.tabs(["🔍 Prediction workspace", "📂 Batch scoring", "📊 Model quality"])

with tab_predict:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Current profile")
        st.dataframe(profile, use_container_width=True, hide_index=True)
    with right:
        st.subheader("How to interpret this")
        st.info("The estimate is model-based decision support. Compare it with market data and review the model quality panel before using it for compensation decisions.")
        st.caption("Feature engineering and categorical handling are applied by the same pipeline used during training.")

with tab_batch:
    st.subheader("Score multiple candidates")
    st.caption("Upload a CSV containing the nine raw input columns. Engineered features are created automatically.")
    required_columns = ["job_title", "experience_years", "education_level", "skills_count", "industry", "company_size", "location", "remote_work", "certifications"]
    uploaded = st.file_uploader("Candidate CSV", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
        missing = sorted(set(required_columns) - set(batch.columns))
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            scored = batch.copy()
            scored["PredictedSalary"] = predict(batch)
            st.success(f"Scored {len(scored):,} candidates")
            st.dataframe(scored.head(25), use_container_width=True, hide_index=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False).encode("utf-8"), "smartpay_predictions.csv", "text/csv", type="primary")

with tab_quality:
    st.subheader("Model quality and governance")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Test R²", f"{METRICS.get('r2', 0):.4f}")
    metric_columns[1].metric("MAE", f"₹{METRICS.get('mae', 0):,.0f}")
    metric_columns[2].metric("RMSE", f"₹{METRICS.get('rmse', 0):,.0f}")
    metric_columns[3].metric("MAPE", f"{METRICS.get('mape', 0) * 100:.2f}%")
    metric_columns[4].metric("Median error", f"₹{METRICS.get('median_absolute_error', 0):,.0f}")

    quality_left, quality_right = st.columns(2)
    with quality_left:
        st.markdown("#### Leakage audit")
        checks = AUDIT.get("leakage_checks", {})
        if leakage_free is True:
            st.success("No data leakage detected by the notebook audit.")
        else:
            st.warning("Run the notebook to generate the leakage audit.")
        st.json(checks)
    with quality_right:
        st.markdown("#### Generalization")
        st.write(f"**Diagnosis:** {fit_diagnosis}")
        st.metric("Train R²", f"{METRICS.get('train_r2', 0):.4f}")
        st.metric("Generalization gap", f"{METRICS.get('generalization_gap', 0):.4f}")
        cv = AUDIT.get("cross_validation", {})
        cv_r2_mean = cv.get("cv_r2_mean", cv.get("r2_mean", 0))
        cv_r2_std = cv.get("cv_r2_std", cv.get("r2_std", 0))
        st.write(f"5-fold CV R²: **{cv_r2_mean:.4f} ± {cv_r2_std:.4f}**")

    st.markdown("#### Candidate model comparison")
    if MODEL_RESULTS:
        comparison = pd.DataFrame(MODEL_RESULTS)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        if {"model", "r2"}.issubset(comparison.columns):
            st.bar_chart(comparison.set_index("model")["r2"])
    else:
        st.info("Run the notebook to generate comparison results.")

    learning_curve_data = AUDIT.get("learning_curve", [])
    if learning_curve_data:
        st.markdown("#### Learning curve")
        learning_curve_frame = pd.DataFrame(learning_curve_data).set_index("training_rows")
        st.line_chart(learning_curve_frame[["train_r2_mean", "validation_r2_mean"]])

    train_rows = METRICS.get("train_rows", "N/A")
test_rows = METRICS.get("test_rows", "N/A")
dataset_rows = METRICS.get("dataset_rows", "N/A")
st.caption(f"Training rows: {train_rows:,} | Test rows: {test_rows:,} | Dataset rows: {dataset_rows:,}" if all(isinstance(value, int) for value in [train_rows, test_rows, dataset_rows]) else "Training metrics are unavailable until the notebook is rerun.")

st.divider()
st.caption("SmartPay Salary Intelligence · Use predictions responsibly and validate against current market information.")
