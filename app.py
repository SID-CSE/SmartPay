"""
SmartPay India -- Complete Prediction System
Predict -> Contribute real data -> Model learns from it -> Transparent about unknowns.

Run: streamlit run app.py
"""
import json
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np

from data_store import DataStore, hash_submitter
from prediction_engine import PredictionEngine
from reconciliation_engine import reconcile
import retrain_pipeline

BASE_DIR = Path(__file__).resolve().parent
ADMIN_PASSWORD = os.environ.get("SMARTPAY_ADMIN_PASSWORD", "changeme123")

st.set_page_config(page_title="SmartPay India | Complete Prediction System", page_icon="\U0001F4BC",
                    layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: #0b1220; }
[data-testid="stHeader"] { background: rgba(11, 18, 32, 0.85); }
.hero h1 { color: #f4f8ff; font-size: 2.3rem; margin: 0; letter-spacing: -1px; }
.hero p { color: #9fb1c9; font-size: 1.0rem; margin-top: .35rem; }
.result-box { background: linear-gradient(135deg, #1f4f77, #17304d); border: 1px solid #50b8ff;
              border-radius: 14px; padding: 1.4rem; text-align: center; color: white; font-size: 1.6rem; font-weight: 700; }
.range-sub { font-size: 1rem; color: #cfe8ff; font-weight: 400; margin-top: .3rem; }
.warn-box { background: #3a301f; border: 1px solid #ffb020; border-radius: 12px; padding: .9rem 1.1rem; color: #ffe1a8; }
.info-box { background: #16283f; border: 1px solid #2f8ef6; border-radius: 12px; padding: .9rem 1.1rem; color: #cfe8ff; }
div[data-testid="stMetric"] { background: #132238; border: 1px solid #243753; border-radius: 12px; padding: .8rem; }
.badge-high { background:#1f3a2a; color:#7cf0bf; padding:2px 10px; border-radius:10px; font-size:.8rem; }
.badge-medium { background:#3a301f; color:#ffd166; padding:2px 10px; border-radius:10px; font-size:.8rem; }
.badge-low { background:#3a1f1f; color:#ff8b8b; padding:2px 10px; border-radius:10px; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_store():
    return DataStore()


@st.cache_resource
def get_engine(_store, artifact_version=0):
    return PredictionEngine(data_store=_store)


store = get_store()
# artifact_version busts the cache after a retrain (see admin tab)
engine = get_engine(store, st.session_state.get('artifact_version', 0))

st.markdown("""
<div class="hero">
  <h1>\U0001F4BC SmartPay India \u2014 Complete Prediction System</h1>
  <p>Predicts salary, learns from real contributions, and is honest when it doesn't know something.</p>
</div>
""", unsafe_allow_html=True)

meta = engine.metadata
st.markdown(
    f"""<div class='info-box'>
    Model R\u00b2: <strong>{meta.get('test_r2',0):.3f}</strong> &nbsp;|&nbsp;
    Trained on <strong>{meta.get('n_training_rows',0):,}</strong> postings &nbsp;|&nbsp;
    Knows <strong>{meta.get('known_companies_count',0):,}</strong> companies, <strong>{meta.get('known_titles_count',0):,}</strong> titles &nbsp;|&nbsp;
    Community-verified contributions so far: <strong>{store.count_usable_contributions()}</strong>
    </div>""",
    unsafe_allow_html=True,
)

tab_predict, tab_contribute, tab_admin, tab_quality, tab_about = st.tabs(
    ["\U0001F50D Predict", "\u270D\ufe0f Contribute Real Data", "\U0001F510 Admin: Retrain", "\U0001F4CA Model Quality", "\u2139\ufe0f About"]
)

# ============================================================================
# TAB 1: PREDICT
# ============================================================================
with tab_predict:
    st.subheader("Get a Salary Estimate")
    col1, col2 = st.columns(2)
    with col1:
        p_title = st.text_input("Job Title", "Software Engineer", key="p_title")
        p_company = st.text_input("Company Name", "", key="p_company",
                                    help="Leave blank if unknown -- we'll use role/experience/location only.")
        c1, c2 = st.columns(2)
        p_exp_min = c1.number_input("Min experience (yrs)", 0, 40, 3, key="p_expmin")
        p_exp_max = c2.number_input("Max experience (yrs)", 0, 40, 6, key="p_expmax")
    with col2:
        p_location = st.text_input("Location", "Bengaluru", key="p_location")
        p_skills = st.text_area("Skills (comma-separated)", "python, sql, aws", key="p_skills")
        has_rating = st.checkbox("I know the company's rating", value=False, key="p_hasrating")
        p_rating = st.slider("Company rating", 1.0, 5.0, 3.9, 0.1, key="p_rating") if has_rating else None
        p_reviews = st.number_input("Review count", 0, 200000, 500, key="p_reviews") if has_rating else 0

    if st.button("\U0001F680 Predict Salary", type="primary", use_container_width=True):
        profile = {
            'title': p_title, 'company_name': p_company or 'Unknown Company',
            'experience_min': p_exp_min, 'experience_max': max(p_exp_max, p_exp_min),
            'location': p_location, 'skills': p_skills,
            'company_rating': p_rating, 'company_reviews': p_reviews,
        }
        with st.spinner("Reconciling model, web search, and other signals..."):
            result = reconcile(profile, engine, use_web_search=True)
        st.session_state['last_result'] = result
        st.session_state['last_profile'] = profile

    if 'last_result' in st.session_state:
        r = st.session_state['last_result']
        lpa = lambda v: v / 100000
        st.markdown(
            f"""<div class='result-box'>Rs {lpa(r['range_low_inr']):.1f} - {lpa(r['range_high_inr']):.1f} Lacs PA
            <div class='range-sub'>Reconciled estimate: Rs {lpa(r['point_estimate_inr']):.1f} LPA</div></div>""",
            unsafe_allow_html=True)

        badge_class = {'high':'badge-high','medium':'badge-medium','low':'badge-low'}[r['confidence']]
        st.markdown(f"<br><span class='{badge_class}'>Confidence: {r['confidence'].upper()}</span> "
                    f"&nbsp; <span style='color:#9fb1c9'>({len(r['signals_used'])} signal(s) combined)</span>",
                    unsafe_allow_html=True)

        if r['disagreement_detected']:
            st.markdown(
                "<div class='warn-box'>\u26a0\ufe0f Our signals disagreed noticeably on this one "
                "(e.g. the model's baseline estimate and a fuzzy-matched or web-sourced figure pointed "
                "to different numbers) -- the range above has been widened to reflect that extra "
                "uncertainty honestly, rather than averaging the disagreement away.</div>",
                unsafe_allow_html=True)

        with st.expander("\U0001F50E How we calculated this \u2014 every signal that went in"):
            for s in r['signals_used']:
                weight_pct = s['weight'] / sum(x['weight'] for x in r['signals_used']) * 100
                st.write(f"**{s['source'].replace('_',' ').title()}** (weight: {weight_pct:.0f}%) \u2014 "
                         f"Rs {lpa(float(np.expm1(s['log_value']))):.1f} LPA \u2014 {s['detail']}")
            if r.get('industry_guessed') and r['industry_guessed'] != 'Other/Unknown':
                st.caption(f"Company classified as: {r['industry_guessed']} (guessed from company name)")
            if r.get('web_search_result') and not r['web_search_result'].get('available'):
                st.caption(f"Web search: {r['web_search_result']['message']}")

        st.divider()
        st.caption("Think this estimate is off? Help improve it \u2192 go to **Contribute Real Data**.")

# ============================================================================
# TAB 2: CONTRIBUTE
# ============================================================================
with tab_contribute:
    st.subheader("Submit a Real Salary Data Point")
    st.caption(
        "This directly improves the model. New submissions are sanity-checked automatically, then "
        "immediately blended into predictions for that company/title. An admin can trigger a full "
        "model retrain periodically to bake verified contributions permanently into the core model."
    )

    with st.form("contribute_form"):
        c1, c2 = st.columns(2)
        with c1:
            f_company = st.text_input("Company Name*")
            f_title = st.text_input("Job Title*")
            f_exp = st.number_input("Years of experience (yours, at this role)*", 0.0, 40.0, 3.0, 0.5)
        with c2:
            f_location = st.text_input("Location*", "Bengaluru")
            f_skills = st.text_input("Key skills (comma-separated)")
            f_remote = st.selectbox("Work mode", ["On-site", "Hybrid", "Remote"])
        f_salary_lpa = st.number_input("Actual annual salary (Lacs PA)*", 0.5, 500.0, 10.0, 0.5)
        f_email = st.text_input("Email (optional, only used for spam-prevention hashing, never stored/shown)")
        submitted = st.form_submit_button("Submit Contribution", type="primary", use_container_width=True)

    if submitted:
        if not (f_company and f_title and f_location):
            st.error("Company, Title, and Location are required.")
        else:
            actual_salary_inr = f_salary_lpa * 100000
            profile_for_check = {
                'title': f_title, 'company_name': f_company, 'experience_min': f_exp, 'experience_max': f_exp,
                'location': f_location, 'skills': f_skills, 'company_rating': None, 'company_reviews': 0,
            }
            predicted = engine.predict(profile_for_check)
            status = store.add_contribution(
                company_name=f_company, title=f_title, experience_years=f_exp, location=f_location,
                skills=f_skills, remote_work=f_remote, actual_salary=actual_salary_inr,
                model_predicted_at_submit=predicted['point_estimate_inr'],
                submitter_hash=hash_submitter(f_email) if f_email else None,
            )
            if status == 'verified':
                store.refresh_lookup_overrides()  # <-- actually apply the blend, immediately
                st.success(
                    f"\u2705 Thank you! Your submission passed our sanity check and is already blended into "
                    f"predictions for '{f_company}' / '{f_title}'."
                )
            else:
                st.warning(
                    f"\u26a0\ufe0f Submitted, but this differs a lot from our current estimate "
                    f"(predicted Rs {predicted['point_estimate_inr']/100000:.1f}L vs your Rs {f_salary_lpa:.1f}L), "
                    "so it's flagged for manual review before being used. Thank you regardless!"
                )
            get_engine.clear()  # refresh engine so blended lookups reflect the new submission immediately
            st.session_state['artifact_version'] = st.session_state.get('artifact_version', 0) + 1

    st.divider()
    st.caption(f"Total contributions so far: {len(store.get_contributions())} "
               f"({store.count_usable_contributions()} verified and already influencing predictions)")

# ============================================================================
# TAB 3: ADMIN
# ============================================================================
with tab_admin:
    st.subheader("Admin: Review & Retrain")
    pwd = st.text_input("Admin password", type="password")
    if pwd != ADMIN_PASSWORD:
        st.info("Enter the admin password to review contributions and trigger a full retrain.")
    else:
        all_contribs = store.get_contributions()
        st.write(f"**{len(all_contribs)} total contributions** "
                 f"({sum(1 for c in all_contribs if c['status']=='verified')} verified, "
                 f"{sum(1 for c in all_contribs if c['status']=='flagged_review')} flagged for review)")

        flagged = [c for c in all_contribs if c['status'] == 'flagged_review']
        if flagged:
            st.markdown("#### Flagged for Review")
            for c in flagged:
                cols = st.columns([3, 1, 1])
                cols[0].write(f"{c['title']} @ {c['company_name']} ({c['location']}) \u2014 "
                               f"claimed Rs {c['actual_salary']/100000:.1f}L vs model's "
                               f"Rs {(c['model_predicted_at_submit'] or 0)/100000:.1f}L")
                if cols[1].button("Approve", key=f"appr_{c['id']}"):
                    store.update_contribution_status(c['id'], 'verified')
                    store.refresh_lookup_overrides()  # apply blend immediately on manual approval too
                    get_engine.clear()
                    st.session_state['artifact_version'] = st.session_state.get('artifact_version', 0) + 1
                    st.rerun()
                if cols[2].button("Reject", key=f"rej_{c['id']}"):
                    store.delete_contribution(c['id'])
                    st.rerun()

        st.divider()
        st.markdown("#### Live Web Search Status (Tavily)")
        from external_lookup import _get_api_key, try_external_lookup
        key = _get_api_key()
        if key:
            st.success(f"\u2705 Tavily API key detected ({key[:8]}...). Live lookup is active for unknown companies.")
        else:
            st.warning(
                "\u26a0\ufe0f No Tavily API key found. Get a free one (1,000 searches/month, no card) at "
                "https://app.tavily.com, then set `TAVILY_API_KEY` as an environment variable or in "
                "`.streamlit/secrets.toml`."
            )
        test_col1, test_col2 = st.columns([2, 1])
        test_query_profile = test_col1.text_input("Test lookup -- company name", "Zoho Corporation")
        if test_col2.button("Run test search", use_container_width=True):
            with st.spinner("Searching..."):
                test_result = try_external_lookup({
                    'title': 'Software Engineer', 'company_name': test_query_profile,
                    'experience_min': 3, 'experience_max': 5, 'location': 'Chennai',
                })
            st.json(test_result)

        st.divider()
        st.markdown("#### Trigger Full Retrain")
        st.caption("Rebuilds the model from [original data + all verified contributions]. Takes ~1-2 minutes.")
        if st.button("\U0001F504 Retrain Now", type="primary"):
            with st.spinner("Retraining..."):
                result = retrain_pipeline.retrain(store)
            if result.get('skipped'):
                st.warning(result['reason'])
            else:
                st.success(f"Retrained on {result['n_training_rows']:,} rows "
                           f"({result['n_contributions_included']} from community). "
                           f"New Test R\u00b2 = {result['test_r2']:.4f}")
                get_engine.clear()
                st.session_state['artifact_version'] = st.session_state.get('artifact_version', 0) + 1

        st.divider()
        st.markdown("#### Model Version History")
        history = store.get_model_history()
        if history:
            st.dataframe(history, use_container_width=True, hide_index=True)
        else:
            st.caption("No retrains yet -- still running the original notebook-trained model.")

# ============================================================================
# TAB 4: MODEL QUALITY
# ============================================================================
with tab_quality:
    st.subheader("Model Quality")
    m = st.columns(4)
    m[0].metric("Test R\u00b2", f"{meta.get('test_r2',0):.4f}")
    m[1].metric("Test MAE", f"Rs {meta.get('test_mae',0)/100000:.2f} L")
    m[2].metric("Test RMSE", f"Rs {meta.get('test_rmse',0)/100000:.2f} L")
    m[3].metric("Quantile coverage", f"{meta.get('quantile_coverage',0)*100:.1f}%")
    st.caption(f"Model version: {meta.get('model_version','v1')} | Training rows: {meta.get('n_training_rows',0):,}")

# ============================================================================
# TAB 5: ABOUT
# ============================================================================
with tab_about:
    st.subheader("How This System Works")
    st.markdown("""
    **1. Predict** \u2014 uses a LightGBM model trained on real Naukri.com postings, with
    company/title/location target-encoding for known entities.

    **2. Contribute** \u2014 submit a real salary data point. It's sanity-checked against the current
    prediction; close matches are auto-verified and *immediately* blended into future predictions for
    that company/title (no retrain needed). Large deviations are queued for admin review instead of
    being silently trusted (basic spam/error protection).

    **3. Admin Retrain** \u2014 periodically, an admin can fully retrain the model on
    [original data + all verified contributions], permanently improving the core model rather than
    just the lookup blend.

    **4. Unknown companies/titles** \u2014 live web search is **intentionally not enabled** in this
    deployment (see `external_lookup.py` for how to wire one in later). When something is unrecognized,
    the app says so explicitly and falls back to a role/experience/location-only estimate rather than
    pretending to know more than it does.
    """)
    st.warning(
        "Deployment note: contributions are stored in a local SQLite file. If deployed to a platform "
        "with an ephemeral filesystem (e.g. Streamlit Community Cloud), this data will NOT persist "
        "across restarts -- swap `data_store.py`'s internals for a real cloud database first."
    )
