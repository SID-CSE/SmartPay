import json

import joblib
import pandas as pd
import streamlit as st


def prepare_input_dataframe(frame):
    frame = frame.copy()
    for col in [
        "job_title",
        "education_level",
        "industry",
        "company_size",
        "location",
        "remote_work",
    ]:
        if col in frame.columns:
            frame[col] = frame[col].astype(str).str.strip().str.title()

    frame["experience_level"] = pd.cut(
        frame["experience_years"],
        bins=[0, 3, 7, 12, 20, 100],
        labels=["Junior", "Mid", "Experienced", "Senior", "Lead"],
        right=False,
    )
    frame["skills_experience_ratio"] = (frame["skills_count"] + 1) / (frame["experience_years"] + 1)
    frame["is_high_certified"] = (frame["certifications"] > 2).astype(int)
    frame["remote_binary"] = frame["remote_work"].map({"No": 0, "Hybrid": 1, "Yes": 1}).fillna(0)
    return frame


MODEL_PATH = "best_salary_regressor.pkl"
MODEL = joblib.load(MODEL_PATH)

with open("model_metrics.json", "r", encoding="utf-8") as f:
    METRICS = json.load(f)

st.set_page_config(page_title="SmartPay4 Salary Predictor", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 900;
        background: linear-gradient(90deg, #8be9fd, #50b8ff, #7cf0bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #dfe6f1;
        margin-bottom: 18px;
    }
    .summary-box {
        background: rgba(28, 37, 52, 0.9);
        border: 1px solid #2e7ce6;
        border-radius: 12px;
        padding: 14px 18px;
        color: #edf5ff;
        margin-bottom: 18px;
    }
    .result-box {
        background: linear-gradient(180deg, #1f2d3d, #132238);
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #2f8ef6;
        margin-top: 12px;
    }
    .footer {
        text-align: center;
        color: #9aa9bd;
        font-size: 13px;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='main-title'>💼 SmartPay4</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Salary Forecasting System for Real-World Job Profiles</div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='summary-box'>
      Final-year machine learning project for salary prediction using end-to-end data science workflow: EDA, feature engineering, model comparison, hyperparameter tuning, evaluation, and deployment-ready prediction.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("💾 Employee Profile")

dataset = pd.read_csv("job_salary_prediction_dataset.csv")
job_titles = sorted(dataset["job_title"].dropna().astype(str).unique().tolist())
industries = sorted(dataset["industry"].dropna().astype(str).unique().tolist())
locations = sorted(dataset["location"].dropna().astype(str).unique().tolist())
company_sizes = sorted(dataset["company_size"].dropna().astype(str).unique().tolist())
remote_modes = ["No", "Hybrid", "Yes"]
education_levels = ["High School", "Diploma", "Bachelor", "Master", "PhD"]

def build_input_frame():
    return pd.DataFrame([
        {
            "job_title": st.sidebar.selectbox("Job Title", job_titles),
            "experience_years": st.sidebar.selectbox("Experience (Years)", list(range(0, 31))),
            "education_level": st.sidebar.selectbox("Education Level", education_levels),
            "skills_count": st.sidebar.selectbox("Skills Count", list(range(0, 21))),
            "industry": st.sidebar.selectbox("Industry", industries),
            "company_size": st.sidebar.selectbox("Company Size", company_sizes),
            "location": st.sidebar.selectbox("Location", locations),
            "remote_work": st.sidebar.selectbox("Remote Work", remote_modes),
            "certifications": st.sidebar.selectbox("Certifications", list(range(0, 11))),
        }
    ])

single_input_df = build_input_frame()

feature_columns = [
    "job_title",
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "certifications",
    "experience_level",
    "skills_experience_ratio",
    "is_high_certified",
    "remote_binary",
]


def predict_salary(frame):
    prepared = prepare_input_dataframe(frame)
    return float(MODEL.predict(prepared[[
        "job_title",
        "experience_years",
        "education_level",
        "skills_count",
        "industry",
        "company_size",
        "location",
        "remote_work",
        "certifications",
        "experience_level",
        "skills_experience_ratio",
        "is_high_certified",
        "remote_binary",
    ]])[0])


tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction", "📊 Model Evaluation"])

with tab1:
    st.subheader("👤 Predict Salary for One Candidate")
    st.dataframe(single_input_df, use_container_width=True)

    if st.button("🚀 Predict Salary"):
        with st.spinner("Running prediction..."):
            predicted = predict_salary(single_input_df)
            st.markdown(f"<div class='result-box'>💰 Predicted Salary: ₹{predicted:,.2f}</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📂 Batch Prediction from CSV")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        required_cols = [
            "job_title",
            "experience_years",
            "education_level",
            "skills_count",
            "industry",
            "company_size",
            "location",
            "remote_work",
            "certifications",
        ]

        if not all(col in batch_df.columns for col in required_cols):
            st.error(f"CSV must contain: {required_cols}")
        else:
            st.success("Validated dataset schema")
            st.dataframe(batch_df.head(), use_container_width=True)
            prepared_batch = prepare_input_dataframe(batch_df)
            output_df = batch_df.copy()
            output_df["PredictedSalary"] = MODEL.predict(
                prepared_batch[[
                    "job_title",
                    "experience_years",
                    "education_level",
                    "skills_count",
                    "industry",
                    "company_size",
                    "location",
                    "remote_work",
                    "certifications",
                    "experience_level",
                    "skills_experience_ratio",
                    "is_high_certified",
                    "remote_binary",
                ]]
            )
            st.dataframe(output_df.head(10), use_container_width=True)
            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions", csv, "salary_predictions.csv", mime="text/csv")

with tab3:
    st.subheader("📊 Model Evaluation Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"₹{METRICS['mae']:,.2f}")
    col2.metric("RMSE", f"₹{METRICS['rmse']:,.2f}")
    col3.metric("R²", f"{METRICS['r2']:.4f}")

    st.caption(f"Best model: {METRICS['best_model']} | Train rows: {METRICS['train_rows']} | Test rows: {METRICS['test_rows']}")

    try:
        st.image("model_comparison.png", caption="Model Comparison by R² Score", use_column_width=True)
        st.image("actual_vs_predicted.png", caption="Actual vs Predicted Salary", use_column_width=True)
        st.image("residuals_plot.png", caption="Residual Plot", use_column_width=True)
        st.image("feature_importance.png", caption="Top Feature Importance", use_column_width=True)
        st.image("correlation_heatmap.png", caption="Correlation Heatmap", use_column_width=True)
    except Exception:
        st.warning("Evaluation images are missing. Run the training script first.")

st.markdown("<div class='footer'>SmartPay4 | Final-year ML portfolio project</div>", unsafe_allow_html=True)
