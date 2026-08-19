# SmartPay Setup & Usage Guide

## 🎯 5-Minute Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data
Place your CSV file (`job_salary_prediction_dataset.csv`) in one of these locations:
- Current directory
- `./data/` folder
- Parent directory

**CSV Format Required:**
```
job_title,experience_years,education_level,skills_count,industry,company_size,location,remote_work,certifications,salary
Software Engineer,7,Bachelor,12,Technology,Medium,India,Hybrid,3,85000
Data Scientist,5,Master,15,Finance,Large,USA,Yes,4,95000
```

### Step 3: Run the Notebook
```bash
jupyter notebook employee_salary_prediction.ipynb
```

Click "Run All" or execute cells sequentially. The notebook will:
- ✓ Load and validate data
- ✓ Train 6 different models
- ✓ Compare performance
- ✓ Tune the best model
- ✓ Save artifacts
- ✓ Generate predictions

**Runtime**: ~2-5 minutes (depending on dataset size)

### Step 4: Use Trained Model
```python
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

print(f"Predicted Salary: ₹{salary:,.2f}")
```

---

## 📊 Detailed Usage Guide

### Option A: Jupyter Notebook (Recommended for Learning)

**Best for**: Data exploration, understanding the pipeline, trying different configurations

```bash
# 1. Start Jupyter
jupyter notebook

# 2. Open: employee_salary_prediction.ipynb

# 3. Run cells in order or "Run All" from menu
```

**What happens:**
- Automatic package installation
- Comprehensive data validation
- Model benchmarking with 5-fold CV
- Hyperparameter tuning
- Detailed visualizations
- Export trained model

**Output files:**
- `smartpay_project/models/best_salary_regressor.pkl` - Trained model
- `smartpay_project/results/model_metrics.json` - Performance metrics
- `smartpay_project/results/model_comparison.json` - Benchmark comparison

### Option B: Python Script (For Inference)

**Best for**: Making predictions with a trained model

```bash
# Make a single prediction
python salary_predictor.py --mode predict --job "Software Engineer" --exp 7
```

Or use as a library:

```python
from salary_predictor import SalaryPredictor
import json

# Load model
predictor = SalaryPredictor('smartpay_project/models/best_salary_regressor.pkl')

# Single prediction
pred = predictor.predict({
    'job_title': 'Data Scientist',
    'experience_years': 5,
    'education_level': 'Master',
    'skills_count': 15,
    'industry': 'Finance',
    'company_size': 'Large',
    'location': 'USA',
    'remote_work': 'Yes',
    'certifications': 4
})

print(f"Predicted: ₹{pred:,.2f}")

# Batch predictions
employees = [
    {...},  # employee 1
    {...},  # employee 2
]

results = predictor.predict_batch(employees)
```

### Option C: REST API (For Production Deployment)

**Best for**: Integration with web applications, dashboards, mobile apps

```bash
# Start the API server
python salary_predictor.py

# Server will run at http://localhost:5000
```

**API Endpoints:**

1. **Health Check**
   ```bash
   curl http://localhost:5000/health
   ```
   Response:
   ```json
   {"status": "healthy", "model_loaded": true}
   ```

2. **Model Info**
   ```bash
   curl http://localhost:5000/info
   ```
   Response:
   ```json
   {
     "metrics": {
       "test_metrics": {"R2": 0.82, "MAE": 8500}
     }
   }
   ```

3. **Single Prediction**
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "job_title": "Software Engineer",
       "experience_years": 7,
       "education_level": "Bachelor",
       "skills_count": 12,
       "industry": "Technology",
       "company_size": "Medium",
       "location": "India",
       "remote_work": "Hybrid",
       "certifications": 3
     }'
   ```
   Response:
   ```json
   {
     "prediction": 85500.00,
     "currency": "INR",
     "timestamp": "2026-08-19T10:30:45.123456"
   }
   ```

4. **Batch Predictions**
   ```bash
   curl -X POST http://localhost:5000/predict_batch \
     -H "Content-Type: application/json" \
     -d '{
       "employees": [
         {"job_title": "...", ...},
         {"job_title": "...", ...}
       ]
     }'
   ```

---

## 🔧 Advanced Configuration

### Customizing the Model

Edit the notebook to change:

```python
# Reproducibility
RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.20  # 80-20 split

# Cross-validation
CV_SPLITS = 5

# Model-specific parameters
candidate_models = {
    'XGBoost': (XGBRegressor(
        n_estimators=250,  # Increase for better accuracy (slower)
        learning_rate=0.05,  # Lower = slower but better
        max_depth=6,  # Decrease to prevent overfitting
        subsample=0.9,  # % of samples per iteration
        colsample_bytree=0.9,  # % of features per iteration
    ), False),
}
```

### Adding Custom Features

Modify the `SmartPayFeatureBuilder` class:

```python
class SmartPayFeatureBuilder(BaseEstimator, TransformerMixin):
    def transform(self, X):
        features = X.copy()
        
        # Add your custom features here
        features['custom_feature'] = features['experience_years'] ** 2
        features['interaction'] = features['skills_count'] * features['certifications']
        
        return features
```

### Handling New Categories

The model handles new values in categorical features automatically:

```python
OneHotEncoder(handle_unknown='ignore')  # Already configured
```

If you get a "new category" error, it means the production data differs significantly from training data.

---

## 📈 Monitoring & Maintenance

### Check Model Performance

```bash
# View metrics
cat smartpay_project/results/model_metrics.json

# View model comparison
cat smartpay_project/results/model_comparison.json
```

### Retraining

1. **Update your data file**
2. **Run the notebook again** (it will automatically use new data)
3. **Review the new metrics** (compare R², MAE, etc.)
4. **Deploy if improved**

Best practices:
- Keep models organized by date: `best_salary_regressor_2026-08-19.pkl`
- Always use a validation set to measure improvement
- Monitor for data drift (distribution changes)

### Production Deployment

**Using Gunicorn (recommended):**

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 \
  "salary_predictor:create_flask_app('smartpay_project/models/best_salary_regressor.pkl')"
```

**Using Docker:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", \
     "salary_predictor:create_flask_app('smartpay_project/models/best_salary_regressor.pkl')"]
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError"
```bash
# Solution: Install packages
pip install -r requirements.txt
```

### Problem: "Dataset not found"
Check that your CSV file is in the correct location:
- Current directory
- `./data/` folder
- Same parent as notebook

**Also verify filename:**
```bash
ls -la *.csv
# Should contain: job_salary_prediction_dataset.csv
```

### Problem: VS Code Notebook Won't Open
- This notebook uses standard JSON format (fully compatible)
- If issues persist:
  1. Update VS Code: `Code → Check for Updates`
  2. Update Python extension: Marketplace → Python (Microsoft)
  3. Try reopening the notebook

### Problem: Slow Training
- **Reduce CV_SPLITS** from 5 to 3
- **Use smaller sample** of data for testing
- **Reduce ESTIMATORS** in XGBoost (250 → 100)
- **Parallel processing** is already enabled (check your CPU)

### Problem: Memory Issues
If dataset is very large (>1M rows):
```python
# In notebook, reduce data:
df = df.sample(100000, random_state=RANDOM_STATE)  # Use 100k samples

# Or use chunks:
chunk_size = 50000
for chunk in pd.read_csv(file, chunksize=chunk_size):
    # Process chunk
    pass
```

### Problem: Poor Predictions
- Verify input data format matches training data
- Check for missing or invalid values
- Ensure categorical values match training set
- Check if data distribution has changed significantly

---

## 📚 Understanding the Output

### Model Metrics
```json
{
  "model": "XGBoost",
  "test_metrics": {
    "MAE": 8500,          // ₹8,500 average error
    "RMSE": 12300,        // Root mean squared error
    "R2": 0.8234,         // 82.34% variance explained
    "MAPE": 10.5          // 10.5% average % error
  },
  "improvement_over_baseline": {
    "R2 Gain": 0.62,      // Baseline was much worse
    "MAE Reduction": 32000 // 32k improvement
  }
}
```

### Feature Importance
Top features for salary prediction:
1. Experience level (most important)
2. Job title
3. Education level
4. Company size
5. Industry

**Note**: Importance ≠ Causation. These are correlations, not causal relationships.

### Learning Curve
- **Upward trend**: Model learns with more data
- **Plateauing**: Adding more data won't help much
- **Gap between curves**: Overfitting (model doesn't generalize)

---

## 🚀 Next Steps

1. **Experiment**: Try different models and hyperparameters
2. **Feature Engineering**: Add domain-specific features
3. **Production Deploy**: Use the API server in your application
4. **Monitor**: Track prediction accuracy in production
5. **Retrain**: Periodically update with new salary data

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Install packages | `pip install -r requirements.txt` |
| Run notebook | `jupyter notebook employee_salary_prediction.ipynb` |
| Load model | `from salary_predictor import SalaryPredictor` |
| Make prediction | `predictor.predict({...})` |
| Start API | `python salary_predictor.py` |
| Check health | `curl http://localhost:5000/health` |
| View metrics | `cat smartpay_project/results/model_metrics.json` |

---

**Happy salary predicting! 🎉**

For more help, review the README.md and check the notebook comments.
