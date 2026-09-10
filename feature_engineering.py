"""
feature_engineering.py -- THE single source of truth for every feature-engineering
rule in this project.

Why this file exists: an earlier version of this project had the same logic
duplicated across the training notebook, a training script, and the live
prediction engine. They drifted out of sync -- the notebook produced a model
with different features than the one the app actually shipped with, using
different file names, in a different folder. This file exists specifically so
that can't happen again: training code, retraining code, and live inference
code ALL import their rules from here. Change a rule once, it's correct
everywhere.

Used by: train_model.py, prediction_engine.py, retrain_pipeline.py, and
the training notebook (which contains this exact code, cell for cell, so it
stays readable as a teaching document without silently diverging from this file).
"""
import re
import numpy as np
import pandas as pd

# ============================================================================
# Constants (India-specific domain knowledge)
# ============================================================================
INDIA_TYPICAL_GRAD_AGE = 22
PLATEAU_YEARS = 14

METRO_TIER1 = {
    'mumbai', 'navi mumbai', 'thane', 'delhi', 'new delhi', 'gurugram', 'gurgaon', 'noida',
    'greater noida', 'faridabad', 'bengaluru', 'bangalore', 'hyderabad', 'chennai', 'pune',
    'kolkata', 'ahmedabad'
}
IT_HUBS = {'bengaluru', 'bangalore', 'hyderabad', 'pune', 'chennai', 'gurugram', 'gurgaon', 'noida', 'greater noida'}

HIGH_VALUE_SKILLS = [
    'python', 'java', 'aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker', 'machine learning',
    'artificial intelligence', 'data science', 'ai', 'ml', 'sql', 'react', 'node', 'devops',
    'cybersecurity', 'security', 'blockchain', 'sap', 'salesforce', 'power bi', 'tableau',
    'product management', 'scrum', 'agile', 'data engineering', 'analytics'
]
LEADERSHIP_TERMS = ['team handling', 'team management', 'stakeholder management', 'leadership',
                     'mentoring', 'people management', 'team lead']

SENIORITY_PATTERNS = [
    (r'\b(intern|trainee|apprentice)\b', 0), (r'\b(fresher|entry level)\b', 0),
    (r'\b(jr\.?|junior|associate)\b', 1), (r'\b(sr\.?|senior)\b', 3),
    (r'\b(lead|principal|staff)\b', 4), (r'\b(manager|mgr)\b', 5),
    (r'\b(senior manager|sr\.? manager|avp|assistant vice president)\b', 6),
    (r'\b(director|head)\b', 7), (r'\b(vice president|\bvp\b)\b', 8),
    (r'\b(chief|cxo|president|ceo|cto|cfo|coo)\b', 9),
]

INDUSTRY_KEYWORDS = {
    'IT/Tech': ['technologies', 'technology', 'software', 'systems', 'infotech', 'digital', 'tech ', 'it services', 'solutions'],
    'Finance/Banking': ['bank', 'financial', 'finance', 'capital', 'investments', 'insurance', 'nbfc'],
    'Consulting': ['consulting', 'consultancy', 'advisory'],
    'Healthcare': ['hospital', 'health', 'pharma', 'medical', 'diagnostics', 'life sciences'],
    'Manufacturing': ['industries', 'manufacturing', 'motors', 'steel', 'engineering works', 'auto'],
    'Retail/E-comm': ['retail', 'mart', 'e-commerce', 'commerce', 'stores'],
    'Education': ['university', 'college', 'school', 'education', 'academy', 'institute'],
}

NUMERIC_FEATURES = [
    'experience_mid', 'experience_breadth', 'experience_post_plateau', 'seniority_score',
    'is_management_title', 'num_locations_listed', 'is_remote', 'is_hybrid', 'is_metro', 'is_it_hub',
    'skills_count', 'high_value_skill_count', 'has_leadership_skill', 'has_rating', 'reviews_count_log',
    'posting_age_days', 'description_word_count', 'mentions_qualification',
    'company_enc', 'title_enc', 'location_enc',
]
CATEGORICAL_FEATURES = ['age_band', 'rating_bucket', 'company_size_proxy', 'primary_location_bucketed']
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def age_band(age):
    if age <= 24: return 'Fresher_Young'
    elif age <= 30: return 'Early_Career'
    elif age <= 38: return 'Mid_Career'
    elif age <= 47: return 'Senior_Career'
    else: return 'Veteran'


def seniority_score(title):
    if pd.isna(title): return 2
    t = str(title).lower()
    scores = [pts for pat, pts in SENIORITY_PATTERNS if re.search(pat, t)]
    return max(scores) if scores else 2


def guess_industry(company_name, keywords=INDUSTRY_KEYWORDS):
    if pd.isna(company_name):
        return 'Other/Unknown'
    name_lower = str(company_name).lower()
    for industry, kws in keywords.items():
        if any(kw in name_lower for kw in kws):
            return industry
    return 'Other/Unknown'


def clean_location_tokens(loc):
    if pd.isna(loc) or loc == '':
        return []
    loc = re.sub(r'^(hybrid|remote)\s*-\s*', '', str(loc).strip(), flags=re.IGNORECASE)
    parts = [p.strip().lower() for p in loc.split(',') if p.strip()]
    return [re.sub(r'\(.*?\)', '', p).strip() for p in parts if p.strip()]


def engineer_dataframe(df: pd.DataFrame, top_locations: set = None) -> tuple:
    """
    Applies every feature-engineering rule to a raw dataframe with columns:
    title, companyName, location, tagsAndSkills, minimumExperience, maximumExperience,
    salary_mid (or NaN if unknown), AggregateRating, ReviewsCount, jobDescription,
    posting_age_days (optional).

    Returns (engineered_df, top_locations_used) -- top_locations is computed from
    the data if not supplied (training time), or passed in for consistent bucketing
    at inference/retrain time.
    """
    df = df.copy()

    df['experience_mid'] = (df['minimumExperience'] + df['maximumExperience']) / 2.0
    df['experience_breadth'] = df['maximumExperience'] - df['minimumExperience']
    estimated_age = INDIA_TYPICAL_GRAD_AGE + df['experience_mid']
    df['age_band'] = estimated_age.apply(age_band)
    df['experience_post_plateau'] = (df['experience_mid'] - PLATEAU_YEARS).clip(lower=0)

    df['seniority_score'] = df['title'].apply(seniority_score)
    df['is_management_title'] = df['title'].astype(str).str.lower().str.contains(
        r'\b(?:manager|head|director|vp|president|chief|lead)\b', regex=True, na=False).astype(int)

    df['location_tokens'] = df['location'].apply(clean_location_tokens)
    df['num_locations_listed'] = df['location_tokens'].apply(lambda x: max(len(x), 1))
    df['is_remote'] = df['location'].astype(str).str.contains('remote', case=False, na=False).astype(int)
    df['is_hybrid'] = df['location'].astype(str).str.contains('hybrid', case=False, na=False).astype(int)
    df['primary_location'] = df['location_tokens'].apply(lambda x: x[0] if x else 'unknown')
    df['is_metro'] = df['location_tokens'].apply(lambda t: int(any(x in METRO_TIER1 for x in t)))
    df['is_it_hub'] = df['location_tokens'].apply(lambda t: int(any(x in IT_HUBS for x in t)))

    if top_locations is None:
        top_locations = set(df['primary_location'].value_counts().head(15).index)
    df['primary_location_bucketed'] = df['primary_location'].apply(lambda x: x if x in top_locations else 'other')

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

    if 'jobDescription' not in df.columns:
        df['jobDescription'] = ''
    df['description_word_count'] = df['jobDescription'].fillna('').apply(lambda x: len(str(x).split()))

    QUALIFICATION_TERMS = ['mba', 'ca ', 'chartered accountant', 'cfa', 'phd', 'ph.d', 'cpa', 'llb', 'ms ', 'm.tech', 'b.tech']
    df['mentions_qualification'] = df['jobDescription'].fillna('').str.lower().apply(
        lambda t: int(any(term in t for term in QUALIFICATION_TERMS)))

    df['title_norm'] = df['title'].astype(str).str.lower().str.strip()

    return df, top_locations


def smoothed_target_encode_fit(df: pd.DataFrame, col: str, target_col: str, smoothing: float, global_mean: float) -> dict:
    """Fits a smoothed mean-target-encoding lookup table on the given rows. Returns {category: encoded_value}."""
    stats = df.groupby(col)[target_col].agg(['mean', 'count'])
    smoothed = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
    return smoothed.to_dict()


def oof_target_encode(df: pd.DataFrame, col: str, target_col: str, train_idx, test_idx, smoothing: float,
                       global_mean: float, n_splits: int = 5, random_state: int = 42) -> pd.Series:
    """
    Out-of-fold smoothed target encoding -- prevents leakage on the training rows
    (each row's encoding is computed from OTHER folds, never itself), while test
    rows get encoded from the full training set's lookup.
    """
    from sklearn.model_selection import KFold
    train_idx = np.array(train_idx)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(index=train_idx, dtype=float)
    for tr, val in kf.split(train_idx):
        fold_lookup = smoothed_target_encode_fit(df.loc[train_idx[tr]], col, target_col, smoothing, global_mean)
        oof.loc[train_idx[val]] = df.loc[train_idx[val], col].map(fold_lookup).fillna(global_mean).values
    result = pd.Series(global_mean, index=df.index)
    result.loc[train_idx] = oof
    full_lookup = smoothed_target_encode_fit(df.loc[train_idx], col, target_col, smoothing, global_mean)
    result.loc[test_idx] = df.loc[test_idx, col].map(full_lookup).fillna(global_mean).values
    return result


# ============================================================================
# Text signal (TF-IDF) -- added after empirical testing showed skills text alone
# (which users already provide via the app form) lifts Test R2 from ~0.68 to
# ~0.71, and an OPTIONAL pasted job description pushes it further to ~0.72.
# Both vectorizers are fit once at training time and saved as artifacts;
# inference-time code transforms new text through the same fitted vectorizers
# (never refits them -- that would be leakage).
# ============================================================================
SKILLS_TFIDF_MAX_FEATURES = 250
DESCRIPTION_TFIDF_MAX_FEATURES = 400
TFIDF_NGRAM_RANGE = (1, 2)  # bigrams catch compound terms like "machine learning", "team lead"


def fit_skills_tfidf(skills_series: pd.Series):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=SKILLS_TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE,
                           token_pattern=r'[a-zA-Z][a-zA-Z\+\#\.]+')
    vec.fit(skills_series.fillna(''))
    return vec


def fit_description_tfidf(description_series: pd.Series):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=DESCRIPTION_TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE,
                           stop_words='english', token_pattern=r'[a-zA-Z][a-zA-Z]+')
    vec.fit(description_series.fillna(''))
    return vec
