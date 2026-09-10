"""
train_model.py -- THE canonical training pipeline. Run this to produce every
file the app needs, from the raw dataset, with no sandbox-specific paths.

Usage:
    python train_model.py
    python train_model.py --data /path/to/other-file.xlsx

Reads:  ../data/raw/indian-job-market-dataset-2025.xlsx (relative to this file)
Writes: ./artifacts/*.pkl, ./artifacts/metadata.json
        ./data/cleaned_training_data.csv   <- used by retrain_pipeline.py later,
                                               so retraining never depends on a
                                               path that only exists on one machine.

This exact logic (feature engineering, target encoding, model choice, hyperparameters)
is also walked through step-by-step, with full explanations, in notebook/employee_
salary_prediction_INDIA.ipynb -- that notebook and this script are kept consistent
by both importing their feature-engineering rules from feature_engineering.py.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

import feature_engineering as fe

RANDOM_STATE = 42
APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_DIR.parent / "data" / "raw" / "indian-job-market-dataset-2025.xlsx"
ARTIFACTS_DIR = APP_DIR / "artifacts"
DATA_DIR = APP_DIR / "data"


def load_and_clean(data_path: Path) -> pd.DataFrame:
    print(f"Loading raw data from {data_path} ...")
    raw = pd.read_excel(data_path, engine='openpyxl')
    print(f"  Raw shape: {raw.shape}")

    df = raw[(raw['currency'] == 'INR') & (raw['minimumSalary'] > 0) & (raw['maximumSalary'] > 0)].copy()
    df = df.reset_index(drop=True)
    df['salary_mid'] = (df['minimumSalary'] + df['maximumSalary']) / 2.0
    print(f"  After INR + disclosed-salary filter: {df.shape}")

    lo, hi = df['salary_mid'].quantile([0.01, 0.99])
    before = len(df)
    df = df[(df['salary_mid'] >= lo) & (df['salary_mid'] <= hi)].reset_index(drop=True)
    print(f"  After outlier capping (1st-99th pct): {df.shape} (removed {before - len(df)})")

    df['log_salary'] = np.log1p(df['salary_mid'])
    return df


def main(data_path: Path):
    t_start = time.time()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    df = load_and_clean(data_path)
    df, top_locations = fe.engineer_dataframe(df)
    print(f"Feature engineering complete. {len(fe.FEATURE_COLS)} feature columns (before encoding).")

    # Save the cleaned + engineered dataset -- this is what retrain_pipeline.py
    # will load later, instead of depending on any machine-specific path.
    cleaned_path = DATA_DIR / "cleaned_training_data.csv"
    df.drop(columns=['location_tokens']).to_csv(cleaned_path, index=False)
    print(f"Saved cleaned training data to {cleaned_path}")

    # Stratified split (by salary decile) so both sets represent the full salary spectrum
    salary_bins = pd.qcut(df['salary_mid'], q=10, labels=False, duplicates='drop')
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=salary_bins)
    global_mean = df.loc[train_idx, 'log_salary'].mean()

    smoothing = {'companyName': 10, 'title_norm': 15, 'primary_location': 20}
    print("Computing out-of-fold target encodings (company, title, location) ...")
    df['company_enc'] = fe.oof_target_encode(df, 'companyName', 'log_salary', train_idx, test_idx, smoothing['companyName'], global_mean)
    df['title_enc'] = fe.oof_target_encode(df, 'title_norm', 'log_salary', train_idx, test_idx, smoothing['title_norm'], global_mean)
    df['location_enc'] = fe.oof_target_encode(df, 'primary_location', 'log_salary', train_idx, test_idx, smoothing['primary_location'], global_mean)

    # Production lookup tables use ALL data (not just train fold) since at inference
    # time we want the best available estimate for every known company/title/location.
    company_lookup = fe.smoothed_target_encode_fit(df, 'companyName', 'log_salary', smoothing['companyName'], global_mean)
    title_lookup = fe.smoothed_target_encode_fit(df, 'title_norm', 'log_salary', smoothing['title_norm'], global_mean)
    location_lookup = fe.smoothed_target_encode_fit(df, 'primary_location', 'log_salary', smoothing['primary_location'], global_mean)

    X_train = df.loc[train_idx, fe.FEATURE_COLS]
    X_test = df.loc[test_idx, fe.FEATURE_COLS]
    ylog_train, ylog_test = df.loc[train_idx, 'log_salary'].values, df.loc[test_idx, 'log_salary'].values
    yraw_train, yraw_test = df.loc[train_idx, 'salary_mid'].values, df.loc[test_idx, 'salary_mid'].values

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), fe.NUMERIC_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), fe.CATEGORICAL_FEATURES),
    ])
    Xtr_p = preprocessor.fit_transform(X_train)
    Xte_p = preprocessor.transform(X_test)
    print(f"Encoded feature matrix (structured features only): {Xtr_p.shape}")

    print("Fitting TF-IDF text vectorizers (skills always available; description optional at inference) ...")
    skills_vec = fe.fit_skills_tfidf(df.loc[train_idx, 'tagsAndSkills'])
    desc_vec = fe.fit_description_tfidf(df.loc[train_idx, 'jobDescription'])
    Xtr_skills = skills_vec.transform(df.loc[train_idx, 'tagsAndSkills'].fillna(''))
    Xte_skills = skills_vec.transform(df.loc[test_idx, 'tagsAndSkills'].fillna(''))
    Xtr_desc = desc_vec.transform(df.loc[train_idx, 'jobDescription'].fillna(''))
    Xte_desc = desc_vec.transform(df.loc[test_idx, 'jobDescription'].fillna(''))

    Xtr_full = hstack([csr_matrix(Xtr_p), Xtr_skills, Xtr_desc]).tocsr()
    Xte_full = hstack([csr_matrix(Xte_p), Xte_skills, Xte_desc]).tocsr()
    print(f"Full feature matrix (structured + skills TF-IDF + description TF-IDF): {Xtr_full.shape}")

    print("Training LightGBM (tuned hyperparameters + regularization for the larger text feature space) ...")
    model = LGBMRegressor(n_estimators=600, num_leaves=100, learning_rate=0.025, subsample=0.8,
                           colsample_bytree=0.5, min_child_samples=15, reg_alpha=1.5, reg_lambda=1.5,
                           random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    model.fit(Xtr_full, ylog_train)

    pred_test = np.clip(np.expm1(model.predict(Xte_full)), 0, None)
    pred_train = np.clip(np.expm1(model.predict(Xtr_full)), 0, None)
    test_r2 = r2_score(yraw_test, pred_test)
    test_rmse = root_mean_squared_error(yraw_test, pred_test)
    test_mae = mean_absolute_error(yraw_test, pred_test)
    train_r2 = r2_score(yraw_train, pred_train)
    print(f"  Train R2={train_r2:.4f}  Test R2={test_r2:.4f}  Test RMSE={test_rmse:,.0f}  Test MAE={test_mae:,.0f}")

    print("Training quantile range models (P10 / P90) ...")
    quantile_models = {}
    for alpha, tag in [(0.10, 'p10'), (0.90, 'p90')]:
        qm = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=200, max_depth=4,
                                         learning_rate=0.05, random_state=RANDOM_STATE)
        qm.fit(Xtr_full, ylog_train)
        quantile_models[tag] = qm
    p10 = np.expm1(quantile_models['p10'].predict(Xte_full))
    p90 = np.expm1(quantile_models['p90'].predict(Xte_full))
    coverage = float(np.mean((yraw_test >= p10) & (yraw_test <= p90)))
    print(f"  Quantile [P10,P90] empirical coverage: {coverage:.1%}")

    print("Computing industry lookup table ...")
    df['guessed_industry'] = df['companyName'].apply(fe.guess_industry)
    industry_lookup = df.groupby('guessed_industry')['log_salary'].mean().to_dict()

    # ---- Save everything the app needs, with the EXACT names prediction_engine.py expects ----
    joblib.dump(preprocessor, ARTIFACTS_DIR / 'preprocessor.pkl')
    joblib.dump(skills_vec, ARTIFACTS_DIR / 'skills_tfidf.pkl')
    joblib.dump(desc_vec, ARTIFACTS_DIR / 'description_tfidf.pkl')
    joblib.dump(model, ARTIFACTS_DIR / 'model.pkl')
    joblib.dump(quantile_models['p10'], ARTIFACTS_DIR / 'quantile_p10.pkl')
    joblib.dump(quantile_models['p90'], ARTIFACTS_DIR / 'quantile_p90.pkl')
    joblib.dump(company_lookup, ARTIFACTS_DIR / 'company_lookup.pkl')
    joblib.dump(title_lookup, ARTIFACTS_DIR / 'title_lookup.pkl')
    joblib.dump(location_lookup, ARTIFACTS_DIR / 'location_lookup.pkl')
    joblib.dump(industry_lookup, ARTIFACTS_DIR / 'industry_lookup.pkl')
    joblib.dump(fe.INDUSTRY_KEYWORDS, ARTIFACTS_DIR / 'industry_keywords.pkl')

    metadata = {
        'global_mean_log_salary': float(global_mean),
        'smoothing': smoothing,
        'test_r2': float(test_r2), 'test_rmse': float(test_rmse), 'test_mae': float(test_mae), 'train_r2': float(train_r2),
        'quantile_coverage': coverage,
        'n_training_rows': int(len(df)),
        'known_companies_count': int(len(company_lookup)),
        'known_titles_count': int(len(title_lookup)),
        'top_locations': sorted(top_locations),
        'feature_cols_numeric': fe.NUMERIC_FEATURES, 'feature_cols_categorical': fe.CATEGORICAL_FEATURES,
        'skills_tfidf_features': fe.SKILLS_TFIDF_MAX_FEATURES,
        'description_tfidf_features': fe.DESCRIPTION_TFIDF_MAX_FEATURES,
        'model_version': 'v2_target_encoded_lightgbm_plus_tfidf',
        'trained_from': str(data_path),
    }
    with open(ARTIFACTS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll artifacts saved to {ARTIFACTS_DIR}")
    for p in sorted(ARTIFACTS_DIR.glob('*')):
        print(f"  - {p.name}")
    print(f"\nDone in {time.time() - t_start:.1f}s. Test R2 = {test_r2:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the SmartPay India model from raw data.")
    parser.add_argument('--data', type=str, default=str(DEFAULT_DATA_PATH),
                        help='Path to the raw indian-job-market-dataset xlsx file')
    args = parser.parse_args()
    main(Path(args.data))
