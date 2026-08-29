"""
prediction_engine.py -- Core inference logic for SmartPay India.

Handles:
  1. Feature engineering from a raw profile (mirrors the training notebook exactly)
  2. "Do we know this company/title/location?" detection
  3. Blending the trained model's lookup table with any user-contributed data
     for that same company/title (lightweight, immediate -- no full retrain needed)
  4. Falling back to a clearly-flagged "insufficient data" estimate when something
     is genuinely unknown everywhere (live web search intentionally not wired in --
     see external_lookup.py for where to plug that in later)
"""
import re
import json
import difflib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from external_lookup import try_external_lookup

ART = Path(__file__).resolve().parent / "artifacts"

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


class PredictionEngine:
    def __init__(self, artifact_dir: Path = ART, data_store=None):
        self.artifact_dir = Path(artifact_dir)
        self.preprocessor = joblib.load(self.artifact_dir / "preprocessor.pkl")
        self.model = joblib.load(self.artifact_dir / "model.pkl")
        self.q_low = joblib.load(self.artifact_dir / "quantile_p10.pkl")
        self.q_high = joblib.load(self.artifact_dir / "quantile_p90.pkl")
        self.company_lookup = joblib.load(self.artifact_dir / "company_lookup.pkl")
        self.title_lookup = joblib.load(self.artifact_dir / "title_lookup.pkl")
        self.location_lookup = joblib.load(self.artifact_dir / "location_lookup.pkl")
        with open(self.artifact_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.global_mean = self.metadata['global_mean_log_salary']
        self.top_locations = set(self.metadata['top_locations'])
        self.data_store = data_store  # optional DataStore, enables blending user contributions

        # Optional secondary signals (industry fallback) -- load if present, degrade gracefully if not
        try:
            self.industry_lookup = joblib.load(self.artifact_dir / "industry_lookup.pkl")
            self.industry_keywords = joblib.load(self.artifact_dir / "industry_keywords.pkl")
        except FileNotFoundError:
            self.industry_lookup, self.industry_keywords = {}, {}

    def guess_industry(self, company_name: str) -> str:
        name_lower = (company_name or '').lower()
        for industry, keywords in self.industry_keywords.items():
            if any(kw in name_lower for kw in keywords):
                return industry
        return 'Other/Unknown'

    def fuzzy_match_title(self, raw_title: str, cutoff: float = 0.72):
        """Returns (matched_title, log_salary_estimate) or (None, None) if no close match found."""
        norm = (raw_title or '').strip().lower()
        if norm in self.title_lookup:
            return None, None  # exact match already handled elsewhere; fuzzy is only a FALLBACK
        matches = difflib.get_close_matches(norm, self.title_lookup.keys(), n=1, cutoff=cutoff)
        if matches:
            return matches[0], self.title_lookup[matches[0]]
        return None, None

    # ------------------------------------------------------------------
    # Feature engineering (mirrors training notebook exactly)
    # ------------------------------------------------------------------
    @staticmethod
    def _age_band(age):
        if age <= 24: return 'Fresher_Young'
        elif age <= 30: return 'Early_Career'
        elif age <= 38: return 'Mid_Career'
        elif age <= 47: return 'Senior_Career'
        else: return 'Veteran'

    def _lookup_with_blend(self, key_type: str, raw_key: str, base_lookup: dict, smoothing: int):
        """
        Returns (encoded_value, known: bool, source: str).
        Blends the trained lookup with any verified user contributions for the same key,
        weighted by sample count -- new real data immediately nudges predictions even
        before a full retrain happens.
        """
        norm_key = raw_key.strip().lower()
        base_entry = base_lookup.get(norm_key) or base_lookup.get(raw_key.strip())
        base_mean, base_count = (base_entry, 1) if isinstance(base_entry, float) else (self.global_mean, 0)
        # our saved lookups are {key: smoothed_mean} floats -- count isn't stored post-smoothing,
        # so treat any found entry as "known" with a nominal weight for blending purposes.
        known = base_entry is not None
        blend_mean, blend_count = base_mean, (5 if known else 0)

        if self.data_store is not None:
            overrides = self.data_store.get_lookup_overrides(key_type)
            ov = overrides.get(norm_key)
            if ov:
                total = blend_count + ov['count']
                blend_mean = (blend_mean * blend_count + ov['mean'] * ov['count']) / total
                known = True
                return blend_mean, known, ('trained+community' if base_entry is not None else 'community_only')

        if known:
            return blend_mean, True, 'trained'
        return self.global_mean, False, 'unknown'

    def engineer_features(self, profile: dict) -> tuple[pd.DataFrame, dict]:
        """Returns (feature_row_df, provenance_info) where provenance flags what was known/unknown."""
        exp_min, exp_max = profile['experience_min'], profile['experience_max']
        experience_mid = (exp_min + exp_max) / 2
        experience_breadth = exp_max - exp_min
        estimated_age = INDIA_TYPICAL_GRAD_AGE + experience_mid
        band = self._age_band(estimated_age)
        experience_post_plateau = max(0, experience_mid - PLATEAU_YEARS)

        title = (profile.get('title') or '').lower()
        scores = [pts for pat, pts in SENIORITY_PATTERNS if re.search(pat, title)]
        seniority_score = max(scores) if scores else 2
        is_management_title = int(bool(re.search(r'\b(?:manager|head|director|vp|president|chief|lead)\b', title)))

        loc = (profile.get('location') or '').lower()
        tokens = [t.strip() for t in loc.split(',') if t.strip()]
        is_metro = int(any(t in METRO_TIER1 for t in tokens))
        is_it_hub = int(any(t in IT_HUBS for t in tokens))
        is_remote = int('remote' in loc)
        is_hybrid = int('hybrid' in loc)
        num_locations_listed = max(len(tokens), 1)
        primary = tokens[0] if tokens else 'unknown'
        primary_location_bucketed = primary if primary in self.top_locations else 'other'

        skills_str = (profile.get('skills') or '').lower()
        skills_list = [s.strip() for s in skills_str.split(',') if s.strip()]
        skills_count = len(skills_list)
        high_value_skill_count = sum(1 for kw in HIGH_VALUE_SKILLS if kw in f",{skills_str},")
        has_leadership_skill = int(any(kw in skills_str for kw in LEADERSHIP_TERMS))

        has_rating = int(profile.get('company_rating') is not None)
        rating_val = profile.get('company_rating') or 0
        if not has_rating: rating_bucket = 'Unknown'
        elif rating_val >= 4.2: rating_bucket = 'High'
        elif rating_val >= 3.5: rating_bucket = 'Medium'
        else: rating_bucket = 'Low'

        reviews = profile.get('company_reviews') or 0
        reviews_count_log = np.log1p(reviews)
        if not has_rating: company_size_proxy = 'Unknown'
        elif reviews >= 10000: company_size_proxy = 'Enterprise'
        elif reviews >= 1000: company_size_proxy = 'Large'
        elif reviews >= 100: company_size_proxy = 'Medium'
        else: company_size_proxy = 'Small'

        # Known/unknown-aware target encodings (with live blending against user contributions)
        company_enc, company_known, company_src = self._lookup_with_blend(
            'companyName', profile.get('company_name', ''), self.company_lookup, self.metadata['smoothing']['companyName'])
        title_enc, title_known, title_src = self._lookup_with_blend(
            'title_norm', profile.get('title', ''), self.title_lookup, self.metadata['smoothing']['title_norm'])
        location_enc, location_known, location_src = self._lookup_with_blend(
            'primary_location', primary, self.location_lookup, self.metadata['smoothing']['primary_location'])

        row = {
            'experience_mid': experience_mid, 'experience_breadth': experience_breadth,
            'experience_post_plateau': experience_post_plateau, 'seniority_score': seniority_score,
            'is_management_title': is_management_title, 'num_locations_listed': num_locations_listed,
            'is_remote': is_remote, 'is_hybrid': is_hybrid, 'is_metro': is_metro, 'is_it_hub': is_it_hub,
            'skills_count': skills_count, 'high_value_skill_count': high_value_skill_count,
            'has_leadership_skill': has_leadership_skill, 'has_rating': has_rating,
            'reviews_count_log': reviews_count_log,
            'posting_age_days': profile.get('posting_age_days', 2),
            'description_word_count': profile.get('description_word_count', 150),
            'mentions_qualification': profile.get('mentions_qualification', 0),
            'company_enc': company_enc, 'title_enc': title_enc, 'location_enc': location_enc,
            'age_band': band, 'rating_bucket': rating_bucket, 'company_size_proxy': company_size_proxy,
            'primary_location_bucketed': primary_location_bucketed,
        }
        cols = self.metadata['feature_cols_numeric'] + self.metadata['feature_cols_categorical']
        feature_df = pd.DataFrame([row])[cols]

        provenance = {
            'company_known': company_known, 'company_source': company_src,
            'title_known': title_known, 'title_source': title_src,
            'location_known': location_known, 'location_source': location_src,
            'fully_known': company_known and title_known,
        }
        return feature_df, provenance

    def predict(self, profile: dict) -> dict:
        """Full pipeline: raw profile -> prediction with range, confidence, and provenance."""
        return self.predict_raw(profile)

    def predict_raw(self, profile: dict) -> dict:
        """The core model-only prediction (used directly by reconciliation_engine.py as one signal)."""
        feature_df, provenance = self.engineer_features(profile)
        X = self.preprocessor.transform(feature_df)
        point = float(np.clip(np.expm1(self.model.predict(X)[0]), 0, None))
        low = float(np.clip(np.expm1(self.q_low.predict(X)[0]), 0, None))
        high = float(np.clip(np.expm1(self.q_high.predict(X)[0]), 0, None))

        result = {
            'point_estimate_inr': point, 'range_low_inr': low, 'range_high_inr': high,
            'provenance': provenance,
        }

        if not provenance['fully_known']:
            # Company and/or title unrecognized by both the trained model and community data.
            # Live web search intentionally not wired in (see external_lookup.py) -- try it anyway,
            # it will cleanly return "unavailable" until a real backend is configured.
            external = try_external_lookup(profile)
            result['external_lookup'] = external
            result['confidence'] = 'low'
            result['confidence_note'] = (
                f"'{profile.get('company_name','This company')}' and/or the exact title weren't found in "
                "our training data or community submissions. This estimate uses role, experience, and "
                "location only -- treat it as a rough market baseline, not a company-specific figure."
            )
        else:
            result['confidence'] = 'high' if provenance['company_source'] != 'community_only' else 'medium'
            result['confidence_note'] = None

        return result
