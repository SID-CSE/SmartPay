# SmartPay Streamlit Integration Guide

Complete guide to integrate your Streamlit app with the SmartPay salary prediction project.

---

## 📋 Quick Overview

You now have:
1. **SmartPay Training Notebook** - Trains models and generates outputs
2. **Streamlit App** - Web interface for predictions and analysis
3. **Python API** - For backend integration

This guide shows how to connect them all.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
# Core ML + Streamlit
pip install -r requirements.txt
pip install streamlit>=1.28.0
```

### Step 2: Train the Model
```bash
jupyter notebook employee_salary_prediction.ipynb

# Run all cells (Ctrl+A, Shift+Enter)
# This creates: smartpay_project/ folder with models and results
```

### Step 3: Run the App
```bash
streamlit run app.py
```

✓ App opens at `http://localhost:8501`

---

## 📂 Project Structure

After training, you'll have:

```
your-project/
├── employee_salary_prediction.ipynb    # Training notebook
├── app.py                          # Your Streamlit app
├── requirements.txt                           # Dependencies
│
├── smartpay_project/                          # Created by notebook
│   ├── models/
│   │   └── best_salary_regressor.pkl         # Trained model
│   ├── results/
│   │   ├── model_metrics.json                # Performance metrics
│   │   ├── model_comparison.json             # Model comparison
│   │   ├── model_audit.json                  # Data quality audit
│   │   └── example_predictions.json          # Sample predictions
│   ├── data/
│   │   └── job_salary_prediction_dataset.csv # Your training data
│
├── salary_predictor.py                        # API/inference module
├── example_predictions.py                     # Example usage
│
└── Documentation/
    ├── README.md                              # Main guide
    ├── SETUP_GUIDE.md                         # Setup instructions
    ├── PROJECT_SUMMARY.md                     # Project overview
    └── STREAMLIT_INTEGRATION_GUIDE.md        # This file
```

---

## 🔄 How It Works

### Data Flow

```
Training Phase:
─────────────────────────────────
notebook → processes data → trains models
    ↓
creates: smartpay_project/
    ├── models/best_salary_regressor.pkl
    ├── results/model_metrics.json
    ├── results/model_comparison.json
    └── results/model_audit.json


Prediction Phase:
─────────────────────────────────
app.py
    ↓ (finds files automatically)
    ├── Loads model from smartpay_project/models/
    ├── Loads dataset from smartpay_project/data/
    ├── Loads metrics from smartpay_project/results/
    ↓
    → Single predictions (sidebar)
    → Batch scoring (CSV upload)
    → Model diagnostics (quality tab)
```

---

## 🎯 Features Explained

### Tab 1: 🔍 Prediction Workspace

**What it does:**
- Shows your current candidate profile
- Explains how to interpret predictions
- Links to model quality information

**How to use:**
1. Adjust settings in left sidebar
2. Click "🚀 Estimate Salary"
3. View result and profile summary

### Tab 2: 📂 Batch Scoring

**What it does:**
- Upload CSV with multiple candidates
- Score all at once
- Download results with predictions

**CSV Format Required:**
```
job_title,experience_years,education_level,skills_count,industry,company_size,location,remote_work,certifications
Software Engineer,7,Bachelor,12,Technology,Medium,India,Hybrid,3
Data Scientist,5,Master,15,Finance,Large,USA,Yes,4
```

**Output:**
```
Original columns + new "PredictedSalary" column
Download as CSV or view in app
```

### Tab 3: 📊 Model Quality

**Performance Metrics:**
- R² (what % of salary variance is explained)
- MAE (average prediction error)
- RMSE (root mean squared error)
- MAPE (percentage error)

**Generalization:**
- Train vs Test performance
- Overfitting detection
- 5-fold cross-validation results

**Model Comparison:**
- All 6 models benchmarked
- R² comparison chart
- Learning curves

**Data Split:**
- Total rows, training rows, test rows
- 80-20 train-test split

### Tab 4: ℹ️ Information

**What it shows:**
- Project description
- Technical stack
- How to use guide
- Links to documentation

---

## 🔧 Customization

### Change the Default Profile

Edit in `app.py`:

```python
# Sidebar - default values
experience_years = st.slider("📅 Experience (years)", 0, 40, 7)  # default = 7
skills_count = st.slider("🛠️ Skills Count", 0, 30, 12)          # default = 12
certifications = st.slider("🏆 Certifications", 0, 15, 3)        # default = 3
```

### Change Colors/Theme

Edit the CSS section:

```python
st.markdown(
    """
    <style>
    .stApp { background: #0b1220; }  # Dark blue background
    # ... more styles ...
    """,
    unsafe_allow_html=True,
)
```

### Add Custom Features

1. Retrain notebook with new features
2. Update `prepare_input_dataframe()` in app:

```python
def prepare_input_dataframe(frame):
    # ... existing code ...
    prepared["your_new_feature"] = ...
    return prepared
```

---

## 🚀 Deployment Options

### Option 1: Local Development

```bash
streamlit run app.py
# http://localhost:8501
```

### Option 2: Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo
4. Deploy (free tier available)

```
https://your-username-smartpay.streamlit.app/
```

### Option 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY smartpay_project/ ./smartpay_project/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

Build and run:

```bash
docker build -t smartpay-app .
docker run -p 8501:8501 smartpay-app
```

### Option 4: Virtual Server (AWS, Azure, GCP)

```bash
# On server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with production server
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --logger.level=error
```

---

## 🔄 Updating the Model

### When to Retrain

- Every quarter with new salary data
- After significant market changes
- If predictions drift from actual salaries
- When new features are available

### How to Retrain

```bash
# 1. Update your dataset CSV
# 2. Run the notebook again
jupyter notebook employee_salary_prediction.ipynb

# 3. Execute all cells
# 4. Refresh your Streamlit app
streamlit run app.py
```

The app will automatically:
- Reload the new model
- Update metrics and diagnostics
- Show improved performance (if any)

---

## 🐛 Troubleshooting

### Problem: "System Not Ready" Error

**Cause:** Files are missing or in wrong location

**Solution:**
1. Run the Jupyter notebook completely (all cells)
2. Check that `smartpay_project/` folder exists
3. Verify files are in correct locations:
   - `smartpay_project/models/best_salary_regressor.pkl`
   - `smartpay_project/results/model_metrics.json`
   - `smartpay_project/data/job_salary_prediction_dataset.csv`

### Problem: "ModuleNotFoundError"

**Solution:**
```bash
pip install -r requirements.txt
pip install streamlit
```

### Problem: Slow App Loading

**Solution:**
```bash
# Clear Streamlit cache
streamlit cache clear
streamlit run app.py
```

### Problem: Batch Upload Fails

**Check:**
1. CSV columns match exactly:
   ```
   job_title, experience_years, education_level, skills_count,
   industry, company_size, location, remote_work, certifications
   ```
2. No extra spaces or different column names
3. CSV is UTF-8 encoded

### Problem: Wrong Predictions

**Check:**
1. Model is fully trained (notebook run completely)
2. Metrics show reasonable R² (>0.70)
3. Test sample predictions look reasonable

---

## 📊 Monitoring & Logging

### View App Logs

```bash
# Terminal shows logs in real-time
streamlit run app.py

# Or use with logging
streamlit run app.py --logger.level=debug
```

### Check Model Performance

View the "Model Quality" tab in the app:
- See R², MAE, RMSE
- Compare all 6 trained models
- Review learning curves
- Check data quality audit

### Monitor Predictions

Option 1: CSV export from batch scoring
```bash
# Download predictions and analyze
```

Option 2: Custom logging (add to app):

```python
# Log predictions to file
with open('predictions_log.csv', 'a') as f:
    f.write(f"{datetime.now()},{prediction}\n")
```

---

## 🎯 Best Practices

### For Users

✓ **Always validate predictions** against market data
✓ **Compare with baseline** (simple average salary)
✓ **Check model quality tab** before major decisions
✓ **Use batch mode** for multiple candidates
✓ **Download results** for record-keeping

### For Deployment

✓ **Keep model and app synchronized** - retrain together
✓ **Monitor prediction drift** - track actual vs predicted
✓ **Update regularly** - quarterly retraining recommended
✓ **Use version control** - Git for all code
✓ **Backup models** - Keep old models for comparison

### For Development

✓ **Test batch uploads** before production
✓ **Validate CSV format** carefully
✓ **Review metrics** after every retrain
✓ **Check for data leakage** (notebook audit)
✓ **Document changes** - keep changelog

---

## 🔐 Security Considerations

### File Access

The app looks for files in multiple locations:
1. Current directory
2. `smartpay_project/` folder
3. Subdirectories (models/, data/, results/)

### Data Privacy

✓ No data is stored from predictions
✓ Batch CSV is processed in-memory only
✓ Results only downloaded by user
✓ No external API calls by default

### Model Protection

If deploying publicly:
```python
# Add password protection
import streamlit as st

password = "your-secure-password"
user_pwd = st.sidebar.text_input("Password:", type="password")

if user_pwd != password:
    st.error("Unauthorized")
    st.stop()
```

---

## 📚 Integration with Python API

If you also want to use the `salary_predictor.py` API:

```python
# In your app or scripts
from salary_predictor import SalaryPredictor

predictor = SalaryPredictor('smartpay_project/models/best_salary_regressor.pkl')

salary = predictor.predict({
    'job_title': 'Software Engineer',
    'experience_years': 7,
    'education_level': 'Bachelor',
    'skills_count': 12,
    'industry': 'Technology',
    'company_size': 'Medium',
    'location': 'India',
    'remote_work': 'Hybrid',
    'certifications': 3
})

print(f"Predicted: ₹{salary:,.2f}")
```

---

## 🎓 Learning Resources

### Streamlit Documentation
- https://docs.streamlit.io/
- Tutorials and API reference
- Community components

### Model Interpretation
- Feature importance
- SHAP values (advanced)
- Prediction intervals

### Deployment
- https://docs.streamlit.io/deploy
- Streamlit Cloud (free tier)
- Docker containerization

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Run app locally | `streamlit run app.py` |
| View logs | Check terminal output |
| Clear cache | `streamlit cache clear` |
| Train model | `jupyter notebook employee_salary_prediction.ipynb` |
| Test API | `python example_predictions.py` |
| Deploy Docker | `docker run -p 8501:8501 smartpay-app` |

---

## ✅ Integration Checklist

- [ ] Downloaded SmartPay project files
- [ ] Installed requirements: `pip install -r requirements.txt`
- [ ] Ran training notebook completely
- [ ] Verified `smartpay_project/` folder created
- [ ] Checked model file exists: `smartpay_project/models/best_salary_regressor.pkl`
- [ ] Ran Streamlit app: `streamlit run app.py`
- [ ] Tested single prediction in sidebar
- [ ] Tested batch scoring with sample CSV
- [ ] Reviewed model quality metrics
- [ ] Read documentation tabs

---

## 🚀 You're Ready!

Everything is set up for:
✓ Single salary predictions
✓ Batch candidate scoring
✓ Model diagnostics
✓ Production deployment

**Start here:**
```bash
streamlit run app.py
```

For questions, check:
- README.md - Overview
- SETUP_GUIDE.md - Installation
- PROJECT_SUMMARY.md - Deep dive
- This file - Streamlit integration

Good luck! 🎉
