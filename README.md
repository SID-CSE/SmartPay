# SmartPay4 - Salary Prediction Project

SmartPay4 is a final-year machine learning project built for salary prediction using realistic job market data. The project follows a professional ML workflow: dataset exploration, preprocessing, feature engineering, model comparison, hyperparameter tuning, evaluation, and a Streamlit deployment for interactive prediction.

## Repository structure

```text
SmartPay/
├── old_smartpay/                 # legacy project backup and historical files
│   └── ...
├── new_smartpay/                 # active ML project used for Git push
│   ├── app.py
│   ├── train_model.py
│   ├── employee salary prediction.ipynb
│   ├── job_salary_prediction_dataset.csv
│   ├── requirements.txt
│   ├── best_salary_regressor.pkl
│   ├── model_metrics.json
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   ├── residuals_plot.png
│   ├── feature_importance.png
│   ├── correlation_heatmap.png
│   ├── salary_distribution.png
│   ├── salary_vs_experience.png
│   └── ...
├── README.md
├── .git/
└── .devcontainer/
```

## Why this dataset was selected

The active dataset is:
- `new_smartpay/job_salary_prediction_dataset.csv`

This is a proper regression problem because the target variable `salary` is continuous and the features represent employee and job profile information. The earlier dataset in the legacy folder was not suitable because it was designed for income classification rather than salary prediction.

## Project goals

- Explore and understand the salary dataset
- Clean and preprocess raw data
- Engineer useful features such as `experience_level`, `skills_experience_ratio`, and `remote_binary`
- Compare multiple regression algorithms
- Tune the best-performing model using hyperparameter optimization
- Evaluate the model using MAE, RMSE, and R²
- Save the trained model and evaluation artifacts
- Build an interactive Streamlit app with single and batch prediction

## Tech stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- Streamlit
- Joblib

## Folder to use

Use the active project folder for all work and final submission:
- `new_smartpay/`

The backup folder is retained only for reference:
- `old_smartpay/`

## Workflow used in the project

1. Load the salary dataset
2. Inspect missing values and target distribution
3. Create data visualizations and correlation checks
4. Engineer additional features
5. Train and compare multiple models
6. Tune the selected model using RandomizedSearchCV
7. Evaluate the final model on test data
8. Save trained model and metrics artifacts
9. Deploy prediction interface with Streamlit

## Run the project

```bash
cd new_smartpay
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Current model performance

The best tuned model is an XGBoost regressor trained on the salary dataset.

Verified metrics:
- MAE: about ₹4,076.27
- RMSE: about ₹5,109.60
- R²: about 0.9812

## Final-year portfolio value

This project is suitable for a resume because it demonstrates:
- end-to-end ML workflow
- data understanding and visualization
- preprocessing and feature engineering
- model comparison and tuning
- regression evaluation and reporting
- deployment-ready app with batch prediction

## Notes

- The repo root is kept clean for Git push.
- Old and irrelevant files are intentionally archived under `old_smartpay/`.
- The active production-ready project is kept in `new_smartpay/`.

## Author

Siddharth Kumar
