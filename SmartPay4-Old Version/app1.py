import pickle
import pandas as pd
import numpy as np
import streamlit as st
import pathlib

# ============================================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================================
BASE_DIR = pathlib.Path(__file__).parent.resolve()

@st.cache_resource
def load_model_and_objects():
    """Load the trained model and preprocessing objects"""
    try:
        # Load the best model
        with open(BASE_DIR / 'best_salary_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load label encoders
        with open(BASE_DIR / 'label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load scaler
        with open(BASE_DIR / 'feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metadata
        with open(BASE_DIR / 'model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return model, label_encoders, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None, None

# Load all objects
model, label_encoders, scaler, metadata = load_model_and_objects()

if model is None:
    st.error("Failed to load model. Make sure all .pkl files are in the same directory as app.py")
    st.stop()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_salary(age, gender, education, occupation, experience):
    """Predict salary using the loaded model"""
    try:
        # Encode categorical variables
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        education_encoded = label_encoders['Education'].transform([education])[0]
        occupation_encoded = label_encoders['Occupation'].transform([occupation])[0]
        
        # Create input array
        input_data = np.array([[age, experience, gender_encoded, education_encoded, occupation_encoded]])
        
        # Scale if needed
        if metadata['needs_scaling']:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        predicted_salary = model.predict(input_data)[0]
        
        return predicted_salary
    except Exception as e:
        return None

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="💼 SmartPay - Salary Predictor", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size: 48px;
    font-weight: bold;
    color: #2ecc71;
    text-align: center;
    margin-bottom: 10px;
}
.subtitle {
    font-size: 18px;
    color: #95a5a6;
    text-align: center;
    margin-bottom: 30px;
}
.result {
    background-color: #2ecc71;
    color: white;
    padding: 20px;
    border-radius: 10px;
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
}
.metrics {
    background-color: #ecf0f1;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>💼 SmartPay - Employee Salary Predictor</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>Using {metadata['model_name']} | Model R² Score: {metadata['test_r2']:.4f}</div>", unsafe_allow_html=True)

# Display model performance
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model R² Score", f"{metadata['test_r2']:.4f}", "Higher is Better")
with col2:
    st.metric("RMSE", f"₹{metadata['test_rmse']:,.0f}", "Lower is Better")
with col3:
    st.metric("MAE", f"₹{metadata['test_mae']:,.0f}", "Lower is Better")
with col4:
    st.metric("CV Score", f"{metadata['cv_score']:.4f}", "Cross-Validation")

st.divider()

# Input section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Employee Information")
    age = st.slider("Age", min_value=18, max_value=65, value=30)
    experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5)

with col2:
    st.subheader("📚 Education & Role")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox("Education Level", label_encoders['Education'].classes_)
    occupation = st.selectbox("Occupation", sorted(label_encoders['Occupation'].classes_))

# Prediction button
if st.button("🚀 Predict Salary", use_container_width=True):
    with st.spinner("Making prediction..."):
        predicted_salary = predict_salary(age, gender, education, occupation, experience)
        
        if predicted_salary is not None:
            st.markdown(f"<div class='result'>💰 Predicted Salary: ₹{predicted_salary:,.2f}</div>", unsafe_allow_html=True)
            
            # Show confidence
            if metadata['test_r2'] > 0.85:
                confidence = "🟢 EXCELLENT"
            elif metadata['test_r2'] > 0.75:
                confidence = "🟡 VERY GOOD"
            else:
                confidence = "🟠 GOOD"
            
            st.markdown(f"<div class='metrics'><b>Model Confidence:</b> {confidence} (±₹{metadata['test_mae']:,.0f})</div>", unsafe_allow_html=True)
        else:
            st.error("❌ Error making prediction. Please check your inputs.")

st.divider()

# Batch prediction
st.subheader("📂 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV file with columns: Age, Experience, Gender, Education, Occupation", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        required_cols = ['Age', 'Experience', 'Gender', 'Education', 'Occupation']
        
        if all(col in batch_data.columns for col in required_cols):
            # Make predictions for all rows
            predictions = []
            for idx, row in batch_data.iterrows():
                pred = predict_salary(row['Age'], row['Gender'], row['Education'], row['Occupation'], row['Experience'])
                predictions.append(pred)
            
            batch_data['Predicted_Salary'] = predictions
            st.dataframe(batch_data, use_container_width=True)
            
            # Download button
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="salary_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"❌ CSV must contain columns: {required_cols}")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")