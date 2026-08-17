from pathlib import Path
import json
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "job_salary_prediction_dataset.csv"
MODEL_PATH = ROOT / "best_salary_regressor.pkl"
METRICS_PATH = ROOT / "model_metrics.json"

REQUIRED_COLUMNS = [
    "job_title",
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "certifications",
    "salary",
]

BASE_FEATURES = [
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

NUMERIC_FEATURES = [
    "experience_years",
    "skills_count",
    "certifications",
    "skills_experience_ratio",
    "remote_binary",
]
CATEGORICAL_FEATURES = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work",
    "experience_level",
]


def prepare_dataset(df):
    df = df.copy()
    for col in [
        "job_title",
        "education_level",
        "industry",
        "company_size",
        "location",
        "remote_work",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    df["experience_level"] = pd.cut(
        df["experience_years"],
        bins=[0, 3, 7, 12, 20, 100],
        labels=["Junior", "Mid", "Experienced", "Senior", "Lead"],
        right=False,
    )
    df["skills_experience_ratio"] = (df["skills_count"] + 1) / (df["experience_years"] + 1)
    df["is_high_certified"] = (df["certifications"] > 2).astype(int)
    df["remote_binary"] = df["remote_work"].map({"No": 0, "Hybrid": 1, "Yes": 1}).fillna(0)
    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def evaluate_model(name, pipeline, X_test, y_test):
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {
        "model": name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def plot_model_comparison(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="r2", y="model", palette="viridis")
    plt.title("Model Comparison by R² Score")
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(ROOT / "model_comparison.png", dpi=300)
    plt.close()


def plot_prediction_quality(y_test, predictions):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predictions, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.title("Actual vs Predicted Salary")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.tight_layout()
    plt.savefig(ROOT / "actual_vs_predicted.png", dpi=300)
    plt.close()

    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.4)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residual Plot")
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(ROOT / "residuals_plot.png", dpi=300)
    plt.close()


def plot_feature_importance(best_pipeline):
    model = best_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return
    transformed_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=importances[indices],
        y=[transformed_names[i] for i in indices],
        palette="viridis",
    )
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(ROOT / "feature_importance.png", dpi=300)
    plt.close()


def plot_eda(df):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["salary"], bins=40, kde=True, color="steelblue")
    plt.title("Salary Distribution")
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(ROOT / "salary_distribution.png", dpi=300)
    plt.close()

    corr = df[["experience_years", "skills_count", "certifications", "salary"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(ROOT / "correlation_heatmap.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="experience_years", y="salary", alpha=0.4)
    plt.title("Salary vs Experience")
    plt.xlabel("Experience (Years)")
    plt.ylabel("Salary")
    plt.tight_layout()
    plt.savefig(ROOT / "salary_vs_experience.png", dpi=300)
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    raw_df = pd.read_csv(DATA_PATH)
    missing = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = prepare_dataset(raw_df[REQUIRED_COLUMNS].copy())
    print(f"Dataset shape: {df.shape}")
    print(df.head().to_string(index=False))
    print("\nMissing values:\n", df.isnull().sum().to_dict())
    print("\nSalary statistics:\n", df["salary"].describe().to_string())

    plot_eda(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["salary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(model_name, pipeline, X_test, y_test)
        results.append(metrics)

    comparison_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    print("\nModel comparison:\n")
    print(comparison_df.to_string(index=False))
    plot_model_comparison(comparison_df)

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", XGBRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    param_dist = {
        "model__n_estimators": [300, 500, 800],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [4, 6, 8],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [0.5, 1.0, 2.0],
    }

    search = RandomizedSearchCV(
        estimator=final_pipeline,
        param_distributions=param_dist,
        n_iter=12,
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    print("\nRunning hyperparameter tuning for XGBoost...")
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    y_pred = best_pipeline.predict(X_test)
    final_mae = mean_absolute_error(y_test, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_r2 = r2_score(y_test, y_pred)

    metrics = {
        "best_model": "XGBoostRegressor",
        "best_params": search.best_params_,
        "mae": round(float(final_mae), 4),
        "rmse": round(float(final_rmse), 4),
        "r2": round(float(final_r2), 4),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    print("\nBest model metrics:\n", json.dumps(metrics, indent=2))

    plot_prediction_quality(y_test, y_pred)
    plot_feature_importance(best_pipeline)

    joblib.dump(best_pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    sample = pd.DataFrame(
        [{
            "job_title": "Software Engineer",
            "experience_years": 7,
            "education_level": "Bachelor",
            "skills_count": 12,
            "industry": "Technology",
            "company_size": "Medium",
            "location": "India",
            "remote_work": "Hybrid",
            "certifications": 3,
        }]
    )
    sample = prepare_dataset(sample)
    sample_prediction = best_pipeline.predict(sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0]
    print(f"\nSample prediction: ₹{sample_prediction:,.2f}")
    print(f"\nSaved model at: {MODEL_PATH}")
    print(f"Saved metrics at: {METRICS_PATH}")


if __name__ == "__main__":
    main()
