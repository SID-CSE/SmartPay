"""
retrain_pipeline.py -- Full batch retrain incorporating verified community contributions.

Reads the cleaned training data from data/cleaned_training_data.csv -- a file
produced by train_model.py, shipped as part of this project. This is a portable,
relative path: retraining works identically on any machine that has this project
folder, with no dependency on where the model was originally trained.

Two tiers of "the model learns from user data" exist in this app, by design:

  1. IMMEDIATE (see prediction_engine.py's _lookup_with_blend): the moment a
     contribution is marked 'verified', its company/title/location gets blended
     into that lookup on the very next prediction -- no retrain needed, instant.

  2. PERIODIC FULL RETRAIN (this file): re-fits the actual LightGBM model from
     scratch on [original training data + all verified contributions]. Meant to
     be run occasionally (admin button, or a scheduled job), not on every single
     submission -- refitting a gradient-boosted model per row would be wasteful
     and would let a single bad submission destabilize things before it's had
     time to accumulate corroborating data points.

Run standalone: python retrain_pipeline.py
Or call retrain(data_store) from the admin panel in the Streamlit app.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import warnings
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

BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"
CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_training_data.csv"  # produced by train_model.py
RANDOM_STATE = 42


def _contributions_to_training_rows(contributions: list) -> pd.DataFrame:
    """Converts verified user-contribution dicts into rows matching the base training schema."""
    rows = []
    for c in contributions:
        exp_mid = c['experience_years']
        rows.append({
            'salary_mid': c['actual_salary'], 'log_salary': np.log1p(c['actual_salary']),
            'minimumExperience': exp_mid, 'maximumExperience': exp_mid,
            'title': c['title'], 'companyName': c['company_name'], 'location': c['location'],
            'tagsAndSkills': c.get('skills') or '',
            'AggregateRating': np.nan, 'ReviewsCount': np.nan, 'jobDescription': '',
        })
    return pd.DataFrame(rows)


def retrain(data_store, min_new_contributions: int = 1) -> dict:
    """
    Runs a full retrain on [cleaned base training data + verified contributions].
    Returns a result dict with new metrics, or {'skipped': True, ...} if not enough new data.
    """
    contributions = data_store.get_contributions(status='verified')
    if len(contributions) < min_new_contributions:
        return {'skipped': True, 'reason': f'Only {len(contributions)} verified contributions (need {min_new_contributions}+)'}

    if not CLEANED_DATA_PATH.exists():
        return {'skipped': True, 'reason': (
            f"{CLEANED_DATA_PATH} not found. Run `python train_model.py` at least once "
            "before retraining -- it produces this file as a side effect of the initial training run."
        )}

    base_df = pd.read_csv(CLEANED_DATA_PATH)
    # Re-engineer only the NEW contribution rows (base_df is already fully engineered)
    contrib_raw = _contributions_to_training_rows(contributions)
    contrib_engineered, _ = fe.engineer_dataframe(contrib_raw, top_locations=set(base_df['primary_location_bucketed'].unique()) - {'other'})
    contrib_engineered = contrib_engineered.drop(columns=['location_tokens'], errors='ignore')

    df = pd.concat([base_df, contrib_engineered], ignore_index=True, sort=False)
    df['log_salary'] = np.log1p(df['salary_mid'])

    salary_bins = pd.qcut(df['salary_mid'], q=10, labels=False, duplicates='drop')
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=salary_bins)
    global_mean = df.loc[train_idx, 'log_salary'].mean()
    smoothing = {'companyName': 10, 'title_norm': 15, 'primary_location': 20}

    df['company_enc'] = fe.oof_target_encode(df, 'companyName', 'log_salary', train_idx, test_idx, smoothing['companyName'], global_mean)
    df['title_enc'] = fe.oof_target_encode(df, 'title_norm', 'log_salary', train_idx, test_idx, smoothing['title_norm'], global_mean)
    df['location_enc'] = fe.oof_target_encode(df, 'primary_location', 'log_salary', train_idx, test_idx, smoothing['primary_location'], global_mean)
    company_lookup = fe.smoothed_target_encode_fit(df, 'companyName', 'log_salary', smoothing['companyName'], global_mean)
    title_lookup = fe.smoothed_target_encode_fit(df, 'title_norm', 'log_salary', smoothing['title_norm'], global_mean)
    location_lookup = fe.smoothed_target_encode_fit(df, 'primary_location', 'log_salary', smoothing['primary_location'], global_mean)

    X_train, X_test = df.loc[train_idx, fe.FEATURE_COLS], df.loc[test_idx, fe.FEATURE_COLS]
    ylog_train, ylog_test = df.loc[train_idx, 'log_salary'].values, df.loc[test_idx, 'log_salary'].values
    yraw_train, yraw_test = df.loc[train_idx, 'salary_mid'].values, df.loc[test_idx, 'salary_mid'].values

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), fe.NUMERIC_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), fe.CATEGORICAL_FEATURES),
    ])
    Xtr_p = preprocessor.fit_transform(X_train)
    Xte_p = preprocessor.transform(X_test)

    skills_vec = fe.fit_skills_tfidf(df.loc[train_idx, 'tagsAndSkills'])
    desc_vec = fe.fit_description_tfidf(df.loc[train_idx, 'jobDescription'])
    Xtr_skills = skills_vec.transform(df.loc[train_idx, 'tagsAndSkills'].fillna(''))
    Xte_skills = skills_vec.transform(df.loc[test_idx, 'tagsAndSkills'].fillna(''))
    Xtr_desc = desc_vec.transform(df.loc[train_idx, 'jobDescription'].fillna(''))
    Xte_desc = desc_vec.transform(df.loc[test_idx, 'jobDescription'].fillna(''))
    Xtr_full = hstack([csr_matrix(Xtr_p), Xtr_skills, Xtr_desc]).tocsr()
    Xte_full = hstack([csr_matrix(Xte_p), Xte_skills, Xte_desc]).tocsr()

    model = LGBMRegressor(n_estimators=600, num_leaves=100, learning_rate=0.025, subsample=0.8,
                           colsample_bytree=0.5, min_child_samples=15, reg_alpha=1.5, reg_lambda=1.5,
                           random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    model.fit(Xtr_full, ylog_train)
    pred_test = np.clip(np.expm1(model.predict(Xte_full)), 0, None)
    test_r2 = r2_score(yraw_test, pred_test)
    test_mae = mean_absolute_error(yraw_test, pred_test)
    test_rmse = root_mean_squared_error(yraw_test, pred_test)

    q_models = {}
    for alpha, tag in [(0.10, 'p10'), (0.90, 'p90')]:
        qm = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=200, max_depth=4,
                                         learning_rate=0.05, random_state=RANDOM_STATE)
        qm.fit(Xtr_full, ylog_train)
        q_models[tag] = qm

    df['guessed_industry'] = df['companyName'].apply(fe.guess_industry)
    industry_lookup = df.groupby('guessed_industry')['log_salary'].mean().to_dict()

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    new_dir = BASE_DIR / "artifacts_versions" / f"v_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, new_dir / 'preprocessor.pkl')
    joblib.dump(skills_vec, new_dir / 'skills_tfidf.pkl')
    joblib.dump(desc_vec, new_dir / 'description_tfidf.pkl')
    joblib.dump(model, new_dir / 'model.pkl')
    joblib.dump(q_models['p10'], new_dir / 'quantile_p10.pkl')
    joblib.dump(q_models['p90'], new_dir / 'quantile_p90.pkl')
    joblib.dump(company_lookup, new_dir / 'company_lookup.pkl')
    joblib.dump(title_lookup, new_dir / 'title_lookup.pkl')
    joblib.dump(location_lookup, new_dir / 'location_lookup.pkl')
    joblib.dump(industry_lookup, new_dir / 'industry_lookup.pkl')
    joblib.dump(fe.INDUSTRY_KEYWORDS, new_dir / 'industry_keywords.pkl')

    top_locations = set(df['primary_location_bucketed'].value_counts().head(15).index) - {'other'}
    metadata = {
        'global_mean_log_salary': float(global_mean), 'smoothing': smoothing,
        'test_r2': float(test_r2), 'test_rmse': float(test_rmse), 'test_mae': float(test_mae),
        'n_training_rows': int(len(df)), 'n_contributions_included': len(contributions),
        'known_companies_count': int(len(company_lookup)), 'known_titles_count': int(len(title_lookup)),
        'top_locations': sorted(top_locations),
        'feature_cols_numeric': fe.NUMERIC_FEATURES, 'feature_cols_categorical': fe.CATEGORICAL_FEATURES,
        'skills_tfidf_features': fe.SKILLS_TFIDF_MAX_FEATURES,
        'description_tfidf_features': fe.DESCRIPTION_TFIDF_MAX_FEATURES,
        'model_version': f'v_{timestamp}', 'retrained_at': timestamp,
    }
    with open(new_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update the cleaned dataset too, so contributions become part of the "base" for next time
    df.drop(columns=['guessed_industry'], errors='ignore').to_csv(CLEANED_DATA_PATH, index=False)

    # Flip "current" artifacts/ to point at this new version
    for f in ART_DIR.glob('*'):
        if f.is_file():
            f.unlink()
    for f in new_dir.glob('*'):
        shutil.copy(f, ART_DIR / f.name)

    data_store.record_model_version(
        n_training_rows=len(df), n_contributions_included=len(contributions),
        test_r2=test_r2, test_mae=test_mae, artifact_dir=str(new_dir),
        notes=f"Retrained with {len(contributions)} community contributions",
    )

    return {
        'skipped': False, 'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse,
        'n_training_rows': len(df), 'n_contributions_included': len(contributions),
        'artifact_dir': str(new_dir),
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from data_store import DataStore
    store = DataStore()
    result = retrain(store)
    print(json.dumps(result, indent=2))
