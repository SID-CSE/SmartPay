"""
reconciliation_engine.py -- Combines every available signal into ONE final answer.

Rather than showing the model's prediction and a web search result as two separate,
disconnected numbers, this module gathers every signal we can find for a given
profile, weights each by how much it should actually be trusted, and blends them
in log-salary space (consistent with how the whole system is modeled) into a
single reconciled estimate -- while staying fully transparent about what went in.

Signal sources, ranked by trust (highest weight first):

  1. Community-verified EXACT company+title match     -- real reported data, most trusted
  2. Trained model, company AND title both KNOWN        -- the validated ~0.68 R2 model
  3. Fuzzy-matched title (e.g. "Sr Dev" -> "Senior Developer") -- reasonable proxy
  4. Live web search result (Tavily)                    -- external corroboration, single-source
  5. Industry-similar company average                   -- e.g. unknown fintech -> Finance/Banking avg
  6. Pure role+experience+location baseline             -- always available, lowest trust, floor

When multiple signals disagree substantially, the final range is widened rather than
silently averaging away the disagreement -- an honest reconciliation flags conflict
instead of hiding it.
"""
import numpy as np

# Pseudo-sample weights per signal tier -- higher = trusted more in the blend.
# These aren't fitted; they're a deliberate, documented judgment call (see README).
WEIGHTS = {
    'community_exact': 40,
    'model_known': 30,
    'fuzzy_title': 15,
    'web_search': 12,
    'industry_similar': 8,
    'baseline': 5,   # always included as a floor signal
}

DISAGREEMENT_THRESHOLD_LOG = 0.35  # ~42% relative gap between signals triggers a widened range + warning


def reconcile(profile: dict, engine, use_web_search: bool = True) -> dict:
    """
    Full reconciliation pipeline for one profile.
    `engine` is a PredictionEngine instance (already loaded with model + lookups + data_store).
    """
    signals = []  # list of dicts: {source, log_value, weight, detail}

    # --- Signal 1 & 2: run the core model, which already knows about community blending ---
    base_result = engine.predict_raw(profile)
    company_known = base_result['provenance']['company_known']
    title_known = base_result['provenance']['title_known']
    model_log_value = np.log1p(base_result['point_estimate_inr'])

    if company_known and base_result['provenance']['company_source'] == 'community_only':
        signals.append({'source': 'community_exact', 'log_value': model_log_value,
                         'weight': WEIGHTS['community_exact'],
                         'detail': f"Community-reported salary for '{profile.get('company_name')}'"})
    elif company_known and title_known:
        signals.append({'source': 'model_known', 'log_value': model_log_value,
                         'weight': WEIGHTS['model_known'],
                         'detail': "Trained model recognizes this company and title"})
    else:
        # Always keep the model's baseline estimate in the mix (role+exp+location signal is real too)
        signals.append({'source': 'baseline', 'log_value': model_log_value,
                         'weight': WEIGHTS['baseline'],
                         'detail': "Role + experience + location only (company/title not recognized)"})

    # --- Signal 3: fuzzy title match, only useful when title itself wasn't already known ---
    if not title_known:
        matched_title, fuzzy_log_val = engine.fuzzy_match_title(profile.get('title', ''))
        if matched_title:
            signals.append({'source': 'fuzzy_title', 'log_value': fuzzy_log_val,
                             'weight': WEIGHTS['fuzzy_title'],
                             'detail': f"Closest known title match: '{matched_title}'"})

    # --- Signal 4: live web search, only attempted when company is genuinely unknown ---
    web_result = None
    if not company_known and use_web_search:
        from external_lookup import try_external_lookup
        web_result = try_external_lookup(profile)
        if web_result.get('available'):
            mid_lpa = (web_result['low_lpa'] + web_result['high_lpa']) / 2
            web_log_val = np.log1p(mid_lpa * 100000)
            signals.append({'source': 'web_search', 'log_value': web_log_val,
                             'weight': WEIGHTS['web_search'],
                             'detail': web_result['message']})

    # --- Signal 5: industry-similar company fallback, when company is unknown ---
    industry_used = None
    if not company_known and engine.industry_lookup:
        industry_used = engine.guess_industry(profile.get('company_name', ''))
        industry_log_val = engine.industry_lookup.get(industry_used)
        if industry_log_val is not None and industry_used != 'Other/Unknown':
            signals.append({'source': 'industry_similar', 'log_value': industry_log_val,
                             'weight': WEIGHTS['industry_similar'],
                             'detail': f"Average for companies classified as '{industry_used}'"})

    # --- Blend: weighted average in log-space ---
    total_weight = sum(s['weight'] for s in signals)
    blended_log = sum(s['log_value'] * s['weight'] for s in signals) / total_weight
    blended_point = float(np.expm1(blended_log))

    # --- Disagreement check: how spread out are the signals? ---
    log_values = [s['log_value'] for s in signals]
    spread = max(log_values) - min(log_values) if len(log_values) > 1 else 0.0
    disagreement = spread > DISAGREEMENT_THRESHOLD_LOG

    # --- Range: start from the model's own quantile range, widen if signals disagree ---
    low, high = base_result['range_low_inr'], base_result['range_high_inr']
    if disagreement:
        # Widen proportionally (not by stacking an absolute half-width, which can blow past zero
        # for already-wide senior-level ranges) -- capped so disagreement can't produce an absurd range.
        widen_factor = min(1 + spread, 1.6)
        low = low / widen_factor
        high = high * widen_factor
        # Make sure every individual signal's value actually falls inside the shown range
        all_values_inr = [np.expm1(v) for v in log_values]
        low = min(low, min(all_values_inr) * 0.95)
        high = max(high, max(all_values_inr) * 1.05)
    low = max(low, 50_000)  # floor: never show a non-sensical near-zero lower bound

    overall_confidence = 'high' if (company_known and title_known) else (
        'medium' if len(signals) >= 3 else 'low'
    )

    return {
        'point_estimate_inr': blended_point,
        'range_low_inr': low,
        'range_high_inr': high,
        'confidence': overall_confidence,
        'disagreement_detected': disagreement,
        'signals_used': signals,
        'industry_guessed': industry_used,
        'web_search_result': web_result,
        'base_model_result': base_result,
    }
