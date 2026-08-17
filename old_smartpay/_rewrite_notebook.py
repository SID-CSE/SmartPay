import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SmartPay4 Salary Prediction\n",
                "\n",
                "This notebook trains and evaluates the active salary-regression model used by the project.\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import joblib\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.preprocessing import OneHotEncoder\n",
                "from xgboost import XGBRegressor\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "root = Path.cwd()\n",
                "df = pd.read_csv(root / 'job_salary_prediction_dataset.csv')\n",
                "required = [\n",
                "    'job_title', 'experience_years', 'education_level', 'skills_count',\n",
                "    'industry', 'company_size', 'location', 'remote_work', 'certifications', 'salary'\n",
                "]\n",
                "missing = [c for c in required if c not in df.columns]\n",
                "assert not missing, f'Missing columns: {missing}'\n",
                "df = df[required].copy()\n",
                "df.head()\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_cols = [\n",
                "    'job_title', 'experience_years', 'education_level', 'skills_count',\n",
                "    'industry', 'company_size', 'location', 'remote_work', 'certifications'\n",
                "]\n",
                "X = df[feature_cols]\n",
                "y = df['salary']\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                "categorical_cols = [col for col in feature_cols if X[col].dtype == 'object']\n",
                "preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)], remainder='passthrough')\n",
                "model = XGBRegressor(\n",
                "    n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9,\n",
                "    colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1\n",
                ")\n",
                "pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])\n",
                "pipeline.fit(X_train, y_train)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "predictions = pipeline.predict(X_test)\n",
                "mae = mean_absolute_error(y_test, predictions)\n",
                "rmse = (mean_squared_error(y_test, predictions)) ** 0.5\n",
                "r2 = r2_score(y_test, predictions)\n",
                "metrics = {\n",
                "    'mae': round(float(mae), 4),\n",
                "    'rmse': round(float(rmse), 4),\n",
                "    'r2': round(float(r2), 4)\n",
                "}\n",
                "print(metrics)\n",
                "joblib.dump(pipeline, root / 'best_salary_regressor.pkl')\n",
                "with open(root / 'model_metrics.json', 'w', encoding='utf-8') as f:\n",
                "    json.dump(metrics, f, indent=2)\n"
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample = pd.DataFrame([\n",
                "    {\n",
                "        'job_title': 'Software Engineer',\n",
                "        'experience_years': 5,\n",
                "        'education_level': 'Bachelor',\n",
                "        'skills_count': 8,\n",
                "        'industry': 'Technology',\n",
                "        'company_size': 'Large',\n",
                "        'location': 'New York',\n",
                "        'remote_work': 'Hybrid',\n",
                "        'certifications': 2\n",
                "    }\n",
                "])\n",
                "print(round(float(joblib.load(root / 'best_salary_regressor.pkl').predict(sample)[0]), 2))\n"
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = Path(r"C:\Users\Siddharth\Desktop\7th Sem\SmartPay\employee salary prediction.ipynb")
path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Notebook updated:", path)
