"""
retrain_pipeline.py -- Full batch retrain incorporating verified community contributions.

Two tiers of "the model learns from user data" exist in this app, by design:

  1. IMMEDIATE (see prediction_engine.py's _lookup_with_blend): the moment a
     contribution is marked 'verified', its company/title/location gets blended
     into that lookup on the very next prediction -- no retrain needed, instant.

  2. PERIODIC FULL RETRAIN (this file): re-fits the actual LightGBM model from
     scratch on [original training data + all verified contributions]. This is
     heavier (rebuilds encodings + refits the model + quantile models) and is
     meant to be run occasionally (admin button, or a scheduled job), not on
     every single submission -- refitting a gradient-boosted model per row would
     be wasteful and would let a single bad submission destabilize things before
     it's had time to accumulate corroborating data points.

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

from prediction_engine import (
    INDIA_TYPICAL_GRAD_AGE, PLATEAU_YEARS, METRO_TIER1, IT_HUBS,
    HIGH_VALUE_SKILLS, LEADERSHIP_TERMS, SENIORITY_PATTERNS,
)

BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"
BASE_TRAINING_DATA = Path("/home/claude/build/stage4.pkl")  # the original ~32.5K cleaned Naukri rows
RANDOM_STATE = 42


def _contributions_to_training_rows(contributions: list[dict]) -> pd.DataFrame:
    """Converts verified user-contribution dicts into rows matching the base training schema."""
    rows = []
    for c in contributions:
        exp_mid = c['experience_years']
        rows.append({
            'salary_mid': c['actual_salary'],
            'experience_mid': exp_mid,
            'experience_breadth': 0.0,
            'minimumExperience': exp_mid, 'maximumExperience': exp_mid,
            'title': c['title'], 'companyName': c['company_name'], 'location': c['location'],
            'tagsAndSkills': c.get('skills') or '',
            'AggregateRating': np.nan, 'ReviewsCount': np.nan,
            'jobDescription': '', 'jobUploaded': 'Just Now',
        })
    return pd.DataFrame(rows)


def _engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Re-applies the SAME feature engineering as the training notebook, for combined data."""
    import re
    df = df.copy()
    df['experience_mid'] = (df['minimumExperience'] + df['maximumExperience']) / 2.0 if 'experience_mid' not in df or df['experience_mid'].isna().any() else df['experience_mid']
    df['experience_breadth'] = df.get('experience_breadth', df['maximumExperience'] - df['minimumExperience']).fillna(0)
    estimated_age = INDIA_TYPICAL_GRAD_AGE + df['experience_mid']

    def age_band(age):
        if age <= 24: return 'Fresher_Young'
        elif age <= 30: return 'Early_Career'
        elif age <= 38: return 'Mid_Career'
        elif age <= 47: return 'Senior_Career'
        else: return 'Veteran'
    df['age_band'] = estimated_age.apply(age_band)
    df['experience_post_plateau'] = (df['experience_mid'] - PLATEAU_YEARS).clip(lower=0)

    def seniority_score(title):
        if pd.isna(title): return 2
        t = str(title).lower()
        scores = [pts for pat, pts in SENIORITY_PATTERNS if re.search(pat, t)]
        return max(scores) if scores else 2
    df['seniority_score'] = df['title'].apply(seniority_score)
    df['is_management_title'] = df['title'].str.lower().str.contains(
        r'\b(?:manager|head|director|vp|president|chief|lead)\b', regex=True, na=False).astype(int)

    def clean_tokens(loc):
        if pd.isna(loc): return []
        loc = re.sub(r'^(hybrid|remote)\s*-\s*', '', str(loc).strip(), flags=re.IGNORECASE)
        parts = [p.strip().lower() for p in loc.split(',') if p.strip()]
        return [re.sub(r'\(.*?\)', '', p).strip() for p in parts if p.strip()]
    df['location_tokens'] = df['location'].apply(clean_tokens)
    df['num_locations_listed'] = df['location_tokens'].apply(lambda x: max(len(x), 1))
    df['is_remote'] = df['location'].str.contains('remote', case=False, na=False).astype(int)
    df['is_hybrid'] = df['location'].str.contains('hybrid', case=False, na=False).astype(int)
    df['primary_location'] = df['location_tokens'].apply(lambda x: x[0] if x else 'unknown')
    df['is_metro'] = df['location_tokens'].apply(lambda t: int(any(x in METRO_TIER1 for x in t)))
    df['is_it_hub'] = df['location_tokens'].apply(lambda t: int(any(x in IT_HUBS for x in t)))
    top_locs = set(df['primary_location'].value_counts().head(15).index)
    df['primary_location_bucketed'] = df['primary_location'].apply(lambda x: x if x in top_locs else 'other')

    def count_skills(tags):
        if pd.isna(tags) or tags == '': return 0
        return len([t for t in str(tags).split(',') if t.strip()])
    def hv_skill_count(tags):
        if pd.isna(tags) or tags == '': return 0
        t = f",{str(tags).lower()},"
        return sum(1 for kw in HIGH_VALUE_SKILLS if kw in t)
    def has_leadership(tags):
        if pd.isna(tags) or tags == '': return 0
        return int(any(kw in str(tags).lower() for kw in LEADERSHIP_TERMS))
    df['skills_count'] = df['tagsAndSkills'].apply(count_skills)
    df['high_value_skill_count'] = df['tagsAndSkills'].apply(hv_skill_count)
    df['has_leadership_skill'] = df['tagsAndSkills'].apply(has_leadership)

    df['has_rating'] = df['AggregateRating'].notna().astype(int)
    median_rating = df['AggregateRating'].median()
    df['rating_filled'] = df['AggregateRating'].fillna(median_rating if pd.notna(median_rating) else 3.8)
    def rating_bucket(row):
        if row['has_rating'] == 0: return 'Unknown'
        r = row['rating_filled']
        return 'High' if r >= 4.2 else ('Medium' if r >= 3.5 else 'Low')
    df['rating_bucket'] = df.apply(rating_bucket, axis=1)
    df['reviews_count_log'] = np.log1p(df['ReviewsCount'].fillna(0))
    def size_proxy(row):
        if row['has_rating'] == 0: return 'Unknown'
        rc = row.get('ReviewsCount', 0) or 0
        return 'Enterprise' if rc >= 10000 else ('Large' if rc >= 1000 else ('Medium' if rc >= 100 else 'Small'))
    df['company_size_proxy'] = df.apply(size_proxy, axis=1)

    if 'posting_age_days' not in df.columns:
        df['posting_age_days'] = 2
    df['posting_age_days'] = df['posting_age_days'].fillna(2)
    if 'description_word_count' not in df.columns:
        df['description_word_count'] = df['jobDescription'].fillna('').apply(lambda x: len(str(x).split()))
    df['description_word_count'] = df['description_word_count'].fillna(0)
    if 'mentions_qualification' not in df.columns:
        df['mentions_qualification'] = 0
    df['mentions_qualification'] = df['mentions_qualification'].fillna(0)

    df['title_norm'] = df['title'].str.lower().str.strip()
    df['log_salary'] = np.log1p(df['salary_mid'])
    return df


def retrain(data_store, min_new_contributions: int = 1) -> dict:
    """
    Runs a full retrain on [base Naukri data + verified contributions].
    Returns a result dict with new metrics, or {'skipped': True, ...} if not enough new data.
    """
    contributions = [c for c in data_store.get_contributions(status='verified')]
    if len(contributions) < min_new_contributions:
        return {'skipped': True, 'reason': f'Only {len(contributions)} verified contributions (need {min_new_contributions}+)'}

    base_df = pd.read_pickle(BASE_TRAINING_DATA)
    contrib_df = _contributions_to_training_rows(contributions)
    combined_raw = pd.concat([base_df, contrib_df], ignore_index=True, sort=False)
    df = _engineer_base_features(combined_raw)

    NUM = ['experience_mid','experience_breadth','experience_post_plateau','seniority_score','is_management_title',
           'num_locations_listed','is_remote','is_hybrid','is_metro','is_it_hub','skills_count','high_value_skill_count',
           'has_leadership_skill','has_rating','reviews_count_log','posting_age_days','description_word_count','mentions_qualification']
    CAT = ['age_band','rating_bucket','company_size_proxy','primary_location_bucketed']

    salary_bins = pd.qcut(df['salary_mid'], q=10, labels=False, duplicates='drop')
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=salary_bins)
    GLOBAL_MEAN = df.loc[train_idx, 'log_salary'].mean()
    SMOOTH = {'companyName': 10, 'title_norm': 15, 'primary_location': 20}

    def oof_encode(col, sm):
        ti = np.array(train_idx); kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
        oof = pd.Series(index=train_idx, dtype=float)
        for tr, val in kf.split(ti):
            s = df.loc[ti[tr]].groupby(col)['log_salary'].agg(['mean', 'count'])
            smv = (s['mean'] * s['count'] + GLOBAL_MEAN * sm) / (s['count'] + sm)
            oof.loc[ti[val]] = df.loc[ti[val], col].map(smv).fillna(GLOBAL_MEAN).values
        r = pd.Series(GLOBAL_MEAN, index=df.index); r.loc[train_idx] = oof
        full_map = df.loc[train_idx].groupby(col)['log_salary'].agg(['mean', 'count'])
        full_smooth = (full_map['mean'] * full_map['count'] + GLOBAL_MEAN * sm) / (full_map['count'] + sm)
        r.loc[test_idx] = df.loc[test_idx, col].map(full_smooth).fillna(GLOBAL_MEAN).values
        prod_map = df.groupby(col)['log_salary'].agg(['mean', 'count'])
        prod_smooth = (prod_map['mean'] * prod_map['count'] + GLOBAL_MEAN * sm) / (prod_map['count'] + sm)
        return r, prod_smooth

    df['company_enc'], company_lookup = oof_encode('companyName', SMOOTH['companyName'])
    df['title_enc'], title_lookup = oof_encode('title_norm', SMOOTH['title_norm'])
    df['location_enc'], location_lookup = oof_encode('primary_location', SMOOTH['primary_location'])
    NUM_ALL = NUM + ['company_enc', 'title_enc', 'location_enc']

    X_train, X_test = df.loc[train_idx, NUM_ALL + CAT], df.loc[test_idx, NUM_ALL + CAT]
    ylog_train, ylog_test = df.loc[train_idx, 'log_salary'].values, df.loc[test_idx, 'log_salary'].values
    yraw_train, yraw_test = df.loc[train_idx, 'salary_mid'].values, df.loc[test_idx, 'salary_mid'].values

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), NUM_ALL),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), CAT),
    ])
    Xtr_p = preprocessor.fit_transform(X_train)
    Xte_p = preprocessor.transform(X_test)

    model = LGBMRegressor(n_estimators=500, num_leaves=90, learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.6, min_child_samples=20, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    model.fit(Xtr_p, ylog_train)
    pred_test = np.clip(np.expm1(model.predict(Xte_p)), 0, None)
    test_r2 = r2_score(yraw_test, pred_test)
    test_mae = mean_absolute_error(yraw_test, pred_test)
    test_rmse = root_mean_squared_error(yraw_test, pred_test)

    q_models = {}
    for alpha, tag in [(0.10, 'p10'), (0.90, 'p90')]:
        qm = GradientBoostingRegressor(loss='quantile', alpha=alpha, n_estimators=200, max_depth=4,
                                         learning_rate=0.05, random_state=RANDOM_STATE)
        qm.fit(Xtr_p, ylog_train)
        q_models[tag] = qm

    # Save as a NEW versioned artifact dir, then atomically flip the "current" pointer
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    new_dir = BASE_DIR / "artifacts_versions" / f"v_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, new_dir / 'preprocessor.pkl')
    joblib.dump(model, new_dir / 'model.pkl')
    joblib.dump(q_models['p10'], new_dir / 'quantile_p10.pkl')
    joblib.dump(q_models['p90'], new_dir / 'quantile_p90.pkl')
    joblib.dump(company_lookup.to_dict(), new_dir / 'company_lookup.pkl')
    joblib.dump(title_lookup.to_dict(), new_dir / 'title_lookup.pkl')
    joblib.dump(location_lookup.to_dict(), new_dir / 'location_lookup.pkl')
    metadata = {
        'global_mean_log_salary': float(GLOBAL_MEAN), 'smoothing': SMOOTH,
        'test_r2': float(test_r2), 'test_rmse': float(test_rmse), 'test_mae': float(test_mae),
        'n_training_rows': int(len(df)), 'n_contributions_included': len(contributions),
        'known_companies_count': int(len(company_lookup)), 'known_titles_count': int(len(title_lookup)),
        'top_locations': sorted(set(df['primary_location_bucketed'].value_counts().head(15).index) - {'other'}),
        'feature_cols_numeric': NUM_ALL, 'feature_cols_categorical': CAT,
        'model_version': f'v_{timestamp}', 'retrained_at': timestamp,
    }
    with open(new_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Flip "current" artifacts/ to point at this new version
    for f in ART_DIR.glob('*'):
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
