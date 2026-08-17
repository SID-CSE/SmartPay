import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_salary_regressor.pkl")

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cccccc;
        margin-bottom: 25px;
    }
    .result {
        font-size: 28px;
        color: #00e676;
        font-weight: bold;
        padding: 15px;
        border-radius: 12px;
        background-color: #263238;
        text-align: center;
        margin-top: 20px;
    }
    .footer {
        font-size: 13px;
        text-align: center;
        color: #888;
        padding-top: 40px;
    }
    .stTabs [role="tab"] {
        background-color: #1e1e1e;
        border: 1px solid #444;
        border-radius: 12px;
        margin-right: 10px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        color: #00c6ff;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0072ff;
        color: white;
        border: 2px solid #00c6ff;
    }
    .sidebar .sidebar-content {
        background-color: #1f1f1f;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<div class='main-title'>💼 SmartPay - Employee Salary Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI/ML Regression Model to Predict Accurate Salaries Based on Employee Profile</div>", unsafe_allow_html=True)
st.info("Developed under **AICTE-IBM SkillsBuild Virtual Internship** by **Edunet Foundation**, Domain: AI/ML", icon="🎓")
st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/siddharthkumaryo)  |  📧 siddharthkumaryo@gmail.com")

# ---------- Sidebar Inputs ----------
st.sidebar.title("💾 Input Employee Data")

age = st.sidebar.selectbox("Select Age", list(range(18, 66)))
gender = st.sidebar.selectbox("Select Gender", ["Male", "Female", "Other"])
education = st.sidebar.selectbox("Select Education", [
    "10th", "12th", "Diploma", "Bachelors", "Masters", "PhD", "Professional Certification"
])
all_occupations = sorted(pd.read_csv("DATASET.csv")["Occupation"].dropna().unique())
occupation = st.sidebar.selectbox("Select Occupation", all_occupations)
experience = st.sidebar.selectbox("Select Experience (Years)", list(range(0, 41)))

# ---------- Live Warnings ----------
warning_msgs = []

if education == "10th" and "ceo" in occupation.lower():
    warning_msgs.append("⚠️ 10th pass student cannot be a CEO.")

if experience > age - 18:
    warning_msgs.append("⚠️ Experience exceeds realistic limit (Age - 18).")

if education in ["10th", "12th"] and occupation.lower() in ["software engineer", "data scientist"]:
    warning_msgs.append("⚠️ This role typically requires Diploma/Bachelors or higher education.")

if warning_msgs:
    for msg in warning_msgs:
        st.sidebar.warning(msg)

st.sidebar.markdown("### 📜 Logical Rules Applied")

st.sidebar.markdown("""
- 👶 Age must be **18–65**
- 🧠 **Experience ≤ Age - 18**
- 💼 CEO/Manager roles need **Age ≥ 30**
- 🧪 PhD holders must be **≥ 24 years**
- 💻 Software/Data roles need **Bachelors+**
- 🡩‍🏫 Interns should have **≤ 2 years experience**
- ⚠️ Senior roles require **≥ 5 years**
- ❌ Overqualified roles (e.g. PhD as Driver) flagged
""")

input_df = pd.DataFrame([{
    "Age": age,
    "Experience": experience,
    "Gender": gender,
    "Education": education,
    "Occupation": occupation
}])

# ---------- Big Tabs ----------
tab1, tab2, tab3 = st.tabs(["🔍 Predict Salary", "📂 Batch Prediction", "📊 Model Evaluation"])

# ---------- Tab 1: Predict Salary ----------
with tab1:
    st.subheader("👤 Individual Salary Prediction")
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Predict Salary Now"):
        try:
            with st.spinner("Predicting..."):
                prediction = model.predict(input_df)
                salary = float(prediction[0])
                st.markdown(f"<div class='result'>💰 Predicted Salary: ₹{salary:,.2f}</div>", unsafe_allow_html=True)
                st.toast("🎯 Prediction Complete!", icon="✅")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ---------- Tab 2: Batch Prediction ----------
with tab2:
    st.subheader("📂 Upload CSV for Batch Prediction")
    st.markdown("Required columns: `Age`, `Experience`, `Gender`, `Education`, `Occupation`")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            required_cols = ["Age", "Experience", "Gender", "Education", "Occupation"]
            if not all(col in batch_data.columns for col in required_cols):
                st.error(f"❌ CSV must contain columns: {required_cols}")
            else:
                st.success("✅ File looks good!")
                st.dataframe(batch_data.head(), use_container_width=True)
                batch_preds = model.predict(batch_data)
                batch_data["PredictedSalary"] = batch_preds
                st.subheader("📈 Predictions")
                st.dataframe(batch_data, use_container_width=True)

                csv = batch_data.to_csv(index=False).encode("utf-8")
                st.download_button("📅 Download Predictions as CSV", csv, file_name="predicted_salaries.csv", mime="text/csv")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# ---------- Tab 3: Model Evaluation ----------
with tab3:
    st.subheader("📊 Model Evaluation & Insights")
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.image("r2_scores.png", caption="R² Score", use_container_width=True)
        with col2:
            st.image("rmse_scores.png", caption="RMSE Score", use_container_width=True)

        st.image("residuals_plot.png", caption="Residuals Plot", use_container_width=True)
        st.image("feature_importance.png", caption="Feature Importances", use_container_width=True)
        st.image("correlation_heatmap.png", caption="Correlation Heatmap", use_container_width=True)
    except:
        st.warning("📉 Some evaluation images are missing. Run `train.py` to generate them.")
# ---------- Footer ----------
st.markdown("<div class='footer'>✨ Crafted with 💻 by <b>Siddharth Kumar</b> | 📧 siddharthkumaryo@gmail.com | 🌐 <a href='https://www.linkedin.com/in/siddharthkumaryo' style='color:#82b1ff;'>LinkedIn</a></div>", unsafe_allow_html=True)
