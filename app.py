"""
SmartPay Streamlit Application
Complete integration with the SmartPay salary prediction pipeline.

Run with: streamlit run app.py
"""

import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import numpy as np


# ============================================================================
# Configuration & Paths
# ============================================================================

ROOT = Path(__file__).resolve().parent

# Try multiple possible locations for model and data
POSSIBLE_MODEL_PATHS = [
    ROOT / "best_salary_regressor.pkl",
    ROOT / "smartpay_project" / "models" / "best_salary_regressor.pkl",
    ROOT / "models" / "best_salary_regressor.pkl",
]

POSSIBLE_DATA_PATHS = [
    ROOT / "job_salary_prediction_dataset.csv",
    ROOT / "smartpay_project" / "data" / "job_salary_prediction_dataset.csv",
    ROOT / "data" / "job_salary_prediction_dataset.csv",
]

POSSIBLE_METRICS_PATHS = [
    ROOT / "model_metrics.json",
    ROOT / "smartpay_project" / "results" / "model_metrics.json",
    ROOT / "results" / "model_metrics.json",
]

POSSIBLE_COMPARISON_PATHS = [
    ROOT / "model_comparison.json",
    ROOT / "smartpay_project" / "results" / "model_comparison.json",
    ROOT / "results" / "model_comparison.json",
]

POSSIBLE_AUDIT_PATHS = [
    ROOT / "model_audit.json",
    ROOT / "smartpay_project" / "results" / "model_audit.json",
    ROOT / "results" / "model_audit.json",
]

def find_file(possible_paths):
    """Find first existing file from list of paths."""
    for path in possible_paths:
        if path.exists():
            return path
    return None

MODEL_PATH = find_file(POSSIBLE_MODEL_PATHS)
DATA_PATH = find_file(POSSIBLE_DATA_PATHS)
METRICS_PATH = find_file(POSSIBLE_METRICS_PATHS)
COMPARISON_PATH = find_file(POSSIBLE_COMPARISON_PATHS)
AUDIT_PATH = find_file(POSSIBLE_AUDIT_PATHS)


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="SmartPay | Salary Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .error-box { background: #3a1f1f; border: 1px solid #ff6b6b; border-radius: 14px; padding: 1rem; }
    .success-box { background: #1f3a2a; border: 1px solid #51cf66; border-radius: 14px; padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Helper Functions
# ============================================================================

def load_json(path, default):
    """Load JSON file safely."""
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Failed to load {path.name}: {e}")
        return default


@st.cache_resource
def load_model():
    """Load the trained model."""
    if MODEL_PATH is None:
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_data
def load_dataset():
    """Load the dataset."""
    if DATA_PATH is None:
        return None
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


def prepare_input_dataframe(frame):
    """Prepare input dataframe with feature engineering."""
    prepared = frame.copy()
    
    # Clean categorical features
    for column in ["job_title", "education_level", "industry", "company_size", "location", "remote_work"]:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("string").str.strip().str.title()
    
    # Create experience level bins
    prepared["experience_level"] = pd.cut(
        pd.to_numeric(prepared["experience_years"], errors='coerce').clip(lower=0),
        bins=[0, 3, 7, 12, 20, float("inf")],
        labels=["Junior", "Mid", "Experienced", "Senior", "Lead"],
        right=False,
    )
    
    # Create ratio features
    experience = pd.to_numeric(prepared["experience_years"], errors='coerce').clip(lower=0)
    skills = pd.to_numeric(prepared["skills_count"], errors='coerce').clip(lower=0)
    certifications = pd.to_numeric(prepared["certifications"], errors='coerce').clip(lower=0)
    
    prepared["skills_experience_ratio"] = (skills + 1) / (experience + 1)
    prepared["certifications_experience_ratio"] = (certifications + 1) / (experience + 1)
    
    # Binary remote work
    prepared["remote_binary"] = prepared["remote_work"].map(
        {"No": 0, "Hybrid": 1, "Yes": 1}
    ).fillna(0)
    
    return prepared


def predict(frame, model):
    """Make predictions on a dataframe."""
    if model is None:
        st.error("Model not loaded. Please train the model first.")
        return None
    
    try:
        prepared = prepare_input_dataframe(frame)
        predictions = model.predict(prepared)
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def check_system_status():
    """Check if all required files are available."""
    status = {
        'model': MODEL_PATH is not None,
        'data': DATA_PATH is not None,
        'metrics': METRICS_PATH is not None,
        'comparison': COMPARISON_PATH is not None,
        'audit': AUDIT_PATH is not None,
    }
    return status


# ============================================================================
# Main Application
# ============================================================================

# Check system status
system_status = check_system_status()
all_ready = all(system_status.values())

if not all_ready:
    st.markdown(
        """
        <div class="error-box">
        <h3>⚠️ System Not Ready</h3>
        <p>Some required files are missing. Please ensure:</p>
        <ol>
        <li>You've run the SmartPay Jupyter notebook completely</li>
        <li>The <code>smartpay_project/</code> folder exists with models and results</li>
        <li>All files are in the correct locations</li>
        </ol>
        <p><strong>Missing files:</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    for component, ready in system_status.items():
        status_icon = "✓" if ready else "✗"
        status_text = "Ready" if ready else "Missing"
        color = "green" if ready else "red"
        st.markdown(f"<span style='color: {color}'>{status_icon} {component.title()}: {status_text}</span>", unsafe_allow_html=True)
    
    st.info("""
    **Quick Fix:**
    1. Run: `jupyter notebook employee_salary_prediction.ipynb`
    2. Execute all cells (Ctrl+A, then Shift+Enter)
    3. Refresh this Streamlit app
    """)
    st.stop()

# Load all resources
MODEL = load_model()
DATASET = load_dataset()
METRICS = load_json(METRICS_PATH, {})
MODEL_RESULTS = load_json(COMPARISON_PATH, [])
AUDIT = load_json(AUDIT_PATH, {})

if MODEL is None or DATASET is None:
    st.error("Failed to load critical resources. Please check the logs above.")
    st.stop()

# Hero Section
st.markdown(
    """
    <div class="hero">
      <h1>💼 SmartPay Salary Intelligence</h1>
      <p>Evidence-based salary prediction with model diagnostics, batch scoring, and transparent evaluation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status Card
leakage_free = AUDIT.get("leakage_free", False)
fit_diagnosis = AUDIT.get("fit_diagnosis", "Model training diagnostics available")
model_name = METRICS.get("model", "Trained salary prediction model")

status_label = "✓ Verified training" if leakage_free else "⚠ Audit pending"
status_class = "verified" if leakage_free else "pending"

st.markdown(
    f"""
    <div class='status-card'>
    <strong class='{status_class}'>{status_label}</strong> &nbsp;|&nbsp; 
    Model: <strong>{model_name}</strong> &nbsp;|&nbsp; 
    {fit_diagnosis}
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Sidebar: Candidate Profile Input
# ============================================================================

with st.sidebar:
    st.title("💾 Candidate Profile")
    st.caption("Configure a profile and generate an estimate.")
    
    # Extract unique values from dataset
    job_titles = sorted(DATASET["job_title"].dropna().astype(str).unique())
    industries = sorted(DATASET["industry"].dropna().astype(str).unique())
    locations = sorted(DATASET["location"].dropna().astype(str).unique())
    company_sizes = sorted(DATASET["company_size"].dropna().astype(str).unique())
    education_levels = sorted(DATASET["education_level"].dropna().astype(str).unique())
    remote_modes = ["No", "Hybrid", "Yes"]
    
    # Input widgets
    job_title = st.selectbox("🎯 Job Title", job_titles, index=0)
    experience_years = st.slider("📅 Experience (years)", 0, 40, 7)
    education_level = st.selectbox("🎓 Education Level", education_levels, index=0)
    skills_count = st.slider("🛠️ Skills Count", 0, 30, 12)
    industry = st.selectbox("🏢 Industry", industries, index=0)
    company_size = st.selectbox("📊 Company Size", company_sizes, index=0)
    location = st.selectbox("🌍 Location", locations, index=0)
    remote_work = st.selectbox("🏠 Remote Work", remote_modes, index=1)
    certifications = st.slider("🏆 Certifications", 0, 15, 3)
    
    predict_clicked = st.button(
        "🚀 Estimate Salary",
        type="primary",
        use_container_width=True
    )
    
    st.divider()
    st.caption("Tips: Adjust sliders and dropdowns, then click 'Estimate Salary' to see the prediction.")


# ============================================================================
# Profile Creation & Prediction
# ============================================================================

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
    prediction = predict(profile, MODEL)
    if prediction is not None:
        st.session_state["latest_prediction"] = float(prediction[0])
        st.session_state["latest_profile"] = {
            "job_title": job_title,
            "experience_years": experience_years,
            "location": location,
            "company_size": company_size,
        }

if "latest_prediction" in st.session_state:
    st.markdown(
        f"""
        <div class='result-box'>
        Estimated Annual Salary<br>
        ₹{st.session_state['latest_prediction']:,.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if "latest_profile" in st.session_state:
        prof = st.session_state["latest_profile"]
        st.caption(
            f"📊 {prof['job_title']} with {prof['experience_years']} years "
            f"at {prof['company_size']} in {prof['location']}"
        )

st.divider()

# ============================================================================
# Tabs: Prediction, Batch Scoring, Quality
# ============================================================================

tab_predict, tab_batch, tab_quality, tab_info = st.tabs([
    "🔍 Prediction Workspace",
    "📂 Batch Scoring",
    "📊 Model Quality",
    "ℹ️ Information"
])

# --- TAB 1: Prediction Workspace ---
with tab_predict:
    col_left, col_right = st.columns([1.1, 1])
    
    with col_left:
        st.subheader("Current Profile")
        st.dataframe(profile, use_container_width=True, hide_index=True)
    
    with col_right:
        st.subheader("How to Use")
        st.info(
            """
            ✓ Set up a candidate profile in the sidebar
            ✓ Click **Estimate Salary** to generate prediction
            ✓ Compare with market data before decisions
            ✓ Review model quality in the "Model Quality" tab
            
            **Note:** Estimates are decision support tools, not absolute predictions.
            """
        )
        st.caption(
            "Feature engineering and categorical handling are applied by the same "
            "pipeline used during model training, ensuring consistency."
        )


# --- TAB 2: Batch Scoring ---
with tab_batch:
    st.subheader("Score Multiple Candidates")
    st.caption("Upload a CSV with the nine required columns to score all candidates at once.")
    
    required_columns = [
        "job_title", "experience_years", "education_level", "skills_count",
        "industry", "company_size", "location", "remote_work", "certifications"
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded = st.file_uploader("📤 Upload Candidate CSV", type=["csv"])
    
    if uploaded:
        try:
            batch = pd.read_csv(uploaded)
            missing = sorted(set(required_columns) - set(batch.columns))
            
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
                st.info(f"Required columns: {', '.join(required_columns)}")
            else:
                st.success(f"✓ CSV validated ({len(batch):,} candidates)")
                
                # Make predictions
                predictions = predict(batch, MODEL)
                
                if predictions is not None:
                    scored = batch.copy()
                    scored["PredictedSalary"] = predictions
                    
                    # Show statistics
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    with stats_col1:
                        st.metric("Total Candidates", f"{len(scored):,}")
                    with stats_col2:
                        st.metric("Avg Salary", f"₹{scored['PredictedSalary'].mean():,.0f}")
                    with stats_col3:
                        st.metric("Min Salary", f"₹{scored['PredictedSalary'].min():,.0f}")
                    with stats_col4:
                        st.metric("Max Salary", f"₹{scored['PredictedSalary'].max():,.0f}")
                    
                    # Show data
                    st.subheader("Predictions Preview")
                    st.dataframe(scored.head(25), use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv_data = scored.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Scored CSV",
                        csv_data,
                        "smartpay_predictions.csv",
                        "text/csv",
                        type="primary",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# --- TAB 3: Model Quality ---
with tab_quality:
    st.subheader("Model Quality & Governance")
    
    # Key Metrics
    st.markdown("#### 📈 Performance Metrics")
    metric_cols = st.columns(5)
    
    test_r2 = METRICS.get("test_metrics", {}).get("R2", METRICS.get("r2", 0))
    test_mae = METRICS.get("test_metrics", {}).get("MAE", METRICS.get("mae", 0))
    test_rmse = METRICS.get("test_metrics", {}).get("RMSE", METRICS.get("rmse", 0))
    test_mape = METRICS.get("test_metrics", {}).get("MAPE", METRICS.get("mape", 0))
    
    metric_cols[0].metric("Test R²", f"{test_r2:.4f}", help="Variance explained by model")
    metric_cols[1].metric("MAE", f"₹{test_mae:,.0f}", help="Mean absolute error")
    metric_cols[2].metric("RMSE", f"₹{test_rmse:,.0f}", help="Root mean squared error")
    metric_cols[3].metric("MAPE", f"{test_mape * 100:.2f}%", help="Mean absolute % error")
    metric_cols[4].metric("Samples", f"{METRICS.get('train_rows', 0):,}", help="Training samples")
    
    col_left, col_right = st.columns(2)
    
    # Leakage Audit
    with col_left:
        st.markdown("#### 🔒 Leakage Audit")
        checks = AUDIT.get("leakage_checks", {})
        
        if leakage_free is True:
            st.success("✓ No data leakage detected")
            st.json(checks)
        else:
            st.warning("⚠ Run the training notebook to generate audit")
    
    # Generalization
    with col_right:
        st.markdown("#### 📊 Generalization")
        st.write(f"**Diagnosis:** {fit_diagnosis}")
        
        train_r2 = METRICS.get("train_metrics", {}).get("R2", METRICS.get("train_r2", 0))
        st.metric("Train R²", f"{train_r2:.4f}")
        
        gap = train_r2 - test_r2
        st.metric("Generalization Gap", f"{gap:.4f}")
        
        cv = AUDIT.get("cross_validation", {})
        cv_r2_mean = cv.get("cv_r2_mean", 0)
        cv_r2_std = cv.get("cv_r2_std", 0)
        st.write(f"5-Fold CV R²: **{cv_r2_mean:.4f} ± {cv_r2_std:.4f}**")
    
    # Model Comparison
    st.markdown("#### 🔀 Candidate Model Comparison")
    if isinstance(MODEL_RESULTS, list) and len(MODEL_RESULTS) > 0:
        comparison_df = pd.DataFrame(MODEL_RESULTS)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        if {"Model", "CV R2"}.issubset(set(comparison_df.columns)) or {"model", "cv_r2"}.issubset(set(comparison_df.columns)):
            col_name = "CV R2" if "CV R2" in comparison_df.columns else "cv_r2"
            model_col = "Model" if "Model" in comparison_df.columns else "model"
            st.bar_chart(comparison_df.set_index(model_col)[col_name])
    else:
        st.info("Model comparison data available after training.")
    
    # Learning Curve
    learning_curve_data = AUDIT.get("learning_curve", [])
    if learning_curve_data:
        st.markdown("#### 📈 Learning Curve")
        learning_df = pd.DataFrame(learning_curve_data)
        if "training_rows" in learning_df.columns:
            learning_df = learning_df.set_index("training_rows")
        st.line_chart(learning_df)
    
    # Data Split Information
    st.markdown("#### 📋 Dataset Information")
    train_rows = METRICS.get("train_rows", "N/A")
    test_rows = METRICS.get("test_rows", "N/A")
    dataset_rows = METRICS.get("dataset_rows", "N/A")
    
    if all(isinstance(v, int) for v in [train_rows, test_rows, dataset_rows]):
        st.caption(
            f"🔹 Total Rows: {dataset_rows:,} | "
            f"🔹 Training: {train_rows:,} (80%) | "
            f"🔹 Test: {test_rows:,} (20%)"
        )
    else:
        st.info("Training metadata will appear after running the Jupyter notebook.")


# --- TAB 4: Information ---
with tab_info:
    st.subheader("About SmartPay")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        #### 🎯 What is SmartPay?
        
        SmartPay is an end-to-end machine learning project that predicts employee salaries based on:
        
        - **Job & Industry**: Role type and sector
        - **Experience**: Years in the field
        - **Education**: Degree level
        - **Skills**: Technical and professional competencies
        - **Company**: Size and location
        - **Work Setup**: Remote/Hybrid/On-site
        
        #### 🚀 Features
        
        ✓ Single & batch predictions
        ✓ Model quality diagnostics
        ✓ Feature importance analysis
        ✓ Leakage-free pipeline
        ✓ Reproducible results
        """)
    
    with col_right:
        st.markdown("""
        #### 🛠️ Technical Stack
        
        - **ML Framework**: Scikit-learn, XGBoost
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Streamlit
        - **Deployment**: Docker, Cloud-ready
        
        #### 📖 How to Use
        
        1. **Single Prediction**: Use sidebar to configure profile
        2. **Batch Scoring**: Upload CSV with multiple candidates
        3. **Quality Check**: Review model metrics and diagnostics
        
        #### ⚠️ Important Notes
        
        - Predictions are decision support, not absolute truth
        - Always validate against current market data
        - Model trained on historical data (may drift over time)
        - Periodically retrain with new salary information
        """)
    
    st.divider()
    
    st.markdown("#### 📚 Documentation")
    st.info("""
    For more information:
    - **Setup Guide**: `SETUP_GUIDE.md`
    - **Project Summary**: `PROJECT_SUMMARY.md`
    - **README**: `README.md`
    - **Training Notebook**: `employee_salary_prediction.ipynb`
    """)
    
    st.divider()
    st.caption(
        f"SmartPay Salary Intelligence • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} • "
        "Use predictions responsibly and validate against current market information."
    )
