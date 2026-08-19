# SmartPay: Employee Salary Prediction

A production-ready machine learning project that predicts employee salaries based on professional profile information.

## 📋 Project Overview

**SmartPay** is a supervised regression model that estimates employee compensation from:
- Job title and industry
- Experience level and education
- Technical skills and certifications
- Company size and work arrangement

### Key Features

✅ **End-to-End Pipeline**
- Data validation and quality checks
- Feature engineering with domain knowledge
- Automated preprocessing and encoding
- Model selection and hyperparameter tuning

✅ **Comprehensive Evaluation**
- Cross-validation with multiple metrics
- Baseline comparison
- Overfitting/underfitting analysis
- Learning curves and error analysis

✅ **Production Ready**
- Serialized models with versioning
- Feature importance analysis
- Residual diagnostics
- Deployment-ready inference

✅ **No Data Leakage**
- Proper train/test split
- Pipeline-based preprocessing
- Validated feature engineering

## 🚀 Quick Start

### Installation

```bash
# Clone or download the project
cd smartpay-salary-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
# Start Jupyter
jupyter notebook employee_salary_prediction.ipynb
```

The notebook will:
1. Load your dataset (auto-detects location)
2. Perform comprehensive data validation
3. Train and compare 6 different models
4. Tune the best model
5. Save artifacts for deployment
6. Generate example predictions

### Dataset Format

Your CSV file should contain these columns:

| Column | Type | Example |
|--------|------|---------|
| job_title | string | Software Engineer |
| experience_years | integer | 7 |
| education_level | string | Bachelor |
| skills_count | integer | 12 |
| industry | string | Technology |
| company_size | string | Medium |
| location | string | India |
| remote_work | string | Hybrid / Yes / No |
| certifications | integer | 3 |
| salary | integer | 85000 |

## 📊 Models Evaluated

| Model | Type | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| Linear Regression | Linear | ⚡ Fast | Baseline | Interpretability |
| Ridge | Linear | ⚡ Fast | Good | Regularized linear |
| Lasso | Linear | ⚡ Fast | Good | Feature selection |
| Random Forest | Tree | ⚡⚡ Medium | Excellent | Balance |
| Extra Trees | Tree | ⚡⚡ Medium | Excellent | Speed |
| **XGBoost** | Boosting | ⚡⚡⚡ Fast | Best | Top performance |

The notebook automatically selects the best model based on cross-validation R².

## 📁 Project Structure

```
smartpay_project/
├── models/
│   ├── best_salary_regressor.pkl      # Trained pipeline
│   └── feature_encoder.pkl             # Fitted encoder
├── data/
│   └── job_salary_prediction_dataset.csv
├── results/
│   ├── model_metrics.json              # Performance metrics
│   ├── model_comparison.json           # Benchmark results
│   └── model_audit.json                # Data quality audit
├── employee_salary_prediction.ipynb
├── salary_predictor.py                 # Inference API
├── requirements.txt
└── README.md
```

## 🔍 Key Metrics

### Performance Indicators

- **R² Score**: Explains what % of salary variance the model captures
- **MAE**: Average prediction error in rupees
- **RMSE**: Root mean squared error (penalizes large errors)
- **MAPE**: Mean absolute percentage error

### Example Results

```
Test Set Performance:
  R² Score: 0.8234
  MAE: ₹8,500
  RMSE: ₹12,300
  MAPE: 10.5%

Improvement over baseline:
  +0.62 R²
  ₹32,000 MAE reduction
```

## 🛠️ Usage

### Option 1: Run the Jupyter Notebook

The notebook is self-contained and includes everything:
- Data loading and validation
- Model training and evaluation
- Visualization and analysis
- Model saving and testing

### Option 2: Use the Python Inference Script

```python
from salary_predictor import SalaryPredictor

# Load the trained model
predictor = SalaryPredictor('smartpay_project/models/best_salary_regressor.pkl')

# Make predictions
prediction = predictor.predict({
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

print(f"Predicted Salary: ₹{prediction:,.2f}")
```

### Option 3: API Deployment

```bash
python salary_predictor.py
# Server runs at http://localhost:5000
```

**POST /predict**
```json
{
  "job_title": "Software Engineer",
  "experience_years": 7,
  "education_level": "Bachelor",
  "skills_count": 12,
  "industry": "Technology",
  "company_size": "Medium",
  "location": "India",
  "remote_work": "Hybrid",
  "certifications": 3
}
```

## 📈 Understanding the Results

### Learning Curves
The notebook generates learning curves showing:
- Training performance increasing with more data ✓
- Validation performance approaching training performance
- The generalization gap (difference between train and validation)

### Feature Importance
For tree-based models (Random Forest, XGBoost):
- Shows which features contribute most to predictions
- Useful for business insights
- **Does NOT imply causation**

### Residual Analysis
Reveals:
- Systematic bias (model consistently over/under-predicting)
- Heteroscedasticity (errors varying with salary level)
- Outliers (individuals with unusual salary patterns)

## ⚠️ Important Notes

### Data Leakage Prevention

✅ **What we did right:**
- Split train/test BEFORE any preprocessing
- Fit all transformers on training data only
- Feature engineering happens inside the pipeline
- No information from test set used during training

### Assumptions

The model assumes:
- Salary depends on the provided features
- No significant temporal trends
- Relationships are relatively stable
- Training data is representative

### Limitations

- Predictions are estimates, not guarantees
- Real salary depends on many unmeasured factors (negotiation, location nuances, etc.)
- Model trained on historical data (may become outdated)
- Feature importance doesn't prove causation

## 📊 Configuration

Edit these in the notebook to customize:

```python
RANDOM_STATE = 42      # Seed for reproducibility
TEST_SIZE = 0.20       # 80-20 train-test split
CV_SPLITS = 5          # 5-fold cross-validation
VERBOSE = True         # Print detailed output
```

## 🔄 Model Retraining

To retrain with new data:

1. Place updated CSV in `data/` folder
2. Run the notebook end-to-end
3. New models are automatically saved to `models/` folder
4. Old models are overwritten

For production:
- Archive old models with timestamps
- Implement A/B testing before full deployment
- Monitor prediction performance continuously

## 🐛 Troubleshooting

### "Dataset not found"
- Ensure CSV is in current directory or `data/` folder
- Check file naming: `job_salary_prediction_dataset.csv`

### "Module not found"
```bash
pip install -r requirements.txt
```

### VS Code Notebook Not Opening
- Update VS Code and Python extension
- The fixed notebook uses standard JSON format (✓ compatible)
- Try: File → Revert to Saved

### Slow Model Training
- Reduce CV_SPLITS to 3
- Reduce train data size for testing
- Use fewer candidates in benchmarking

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Guide](https://xgboost.readthedocs.io/)
- [Feature Engineering Guide](https://machinelearningmastery.com/)
- [Model Evaluation Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)

## 📝 License

This project is provided for educational and commercial use.

## 🤝 Contributing

To improve the model:
1. Analyze residuals for patterns
2. Engineer new features based on domain knowledge
3. Try different hyperparameters
4. Compare with ensemble methods
5. Validate on holdout data

## 📞 Support

For issues:
1. Check the troubleshooting section
2. Review error messages in the notebook
3. Verify dataset format
4. Check that all packages are installed

---

**Last Updated**: 2026-08-19
**Python Version**: 3.8+
**Status**: Production Ready ✓
