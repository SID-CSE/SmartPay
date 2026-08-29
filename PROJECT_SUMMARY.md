# SmartPay India — Complete Project Summary

Everything about this project, in one place: the datasets tested, the model that won and why,
every file and what it does, and exactly how the whole system works end to end.

---

## 1. What This Project Is

**SmartPay India** predicts annual salary (in Lacs Per Annum, India's native convention) for a given
job/candidate profile — title, company, experience, location, skills — using a model trained on real
Naukri.com job postings. It's wrapped in a complete Streamlit application that also:
- Lets users **contribute real salary data**, which improves predictions immediately and permanently
- **Searches the web** for companies it doesn't recognize (via a free API)
- **Reconciles every available signal** (model, web search, fuzzy matches, industry averages) into
  one transparent answer instead of showing disconnected numbers

---

## 2. The Dataset — What Was Used, and What Wasn't (and why)

### 2.1 The dataset actually used: `indian-job-market-dataset-2025.xlsx`

Real job postings scraped from **Naukri.com** (India's largest job portal), provided as the project's
core data source.

| Stage | Rows | What happened |
|---|---|---|
| Raw file | 97,929 | 17 columns: title, jobId, currency, jobUploaded, companyName, tagsAndSkills, experience, salary, location, companyId, ReviewsCount, AggregateRating, jobDescription, minimumSalary, maximumSalary, minimumExperience, maximumExperience |
| After currency + disclosure filter | 33,209 | Kept only `currency == 'INR'` and `minimumSalary > 0` — a critical catch: "Not disclosed" postings are encoded as **`0`, not `NaN`**, which would silently corrupt the target if missed |
| After outlier capping | **32,583** | Capped at the 1st/99th percentile of `salary_mid` (removes ~626 extreme/likely-erroneous rows while preserving the realistic range) |

**Target variable:** `salary_mid = (minimumSalary + maximumSalary) / 2`, trained in `log1p` space
(raw skew ≈19 → log skew ≈0.56, a much better-behaved target for regression).

### 2.2 Datasets tested and rejected — with real, measured reasons

Five other real datasets were tested empirically (not assumed) before settling on the Naukri data:

| Dataset | Real? | Best R² achieved | Why it lost |
|---|---|---|---|
| `Salary_Dataset_DataScienceLovers.csv` (Glassdoor India scrape) | Yes | 0.26–0.29 | **Has no experience/tenure column at all** — and experience alone explains ~48% of salary variance everywhere else tested. No amount of tuning fixes missing information. |
| `indian_tech_jobs_2026.csv` | **Questionable** | 0.48–0.51 | Only 2,768/23,201 rows (12%) disclosed salary; job descriptions were suspiciously truncated to exactly 90 characters or "Not Available"; all rows shared one identical scrape timestamp — fingerprints of partially synthetic data, flagged explicitly rather than used |
| Stack Overflow Developer Survey 2025 — **India only** | Yes | 0.34 | Only 1,050 India respondents with disclosed compensation after filtering — too small a sample |
| Stack Overflow Developer Survey 2025 — **Global** | Yes | 0.57–0.64 | No employer-identity field (anonymous survey) — the single biggest lever (company target encoding) isn't available |
| Original toy/synthetic dataset (first attempt, before real data was provided) | No | ~0.85 (meaningless) | Discarded entirely — unrealistically clean, taught the wrong lessons |

**The pattern that emerged:** every dataset without company-identity signal capped between 0.26–0.64.
The one dataset where company identity could be safely extracted (via target encoding, not raw
one-hot) reached the best real result.

### 2.3 How the winning result was found: target encoding

The Naukri data alone (without using company/title identity) capped at **R² = 0.593**. Testing
confirmed company identity carries real signal (11,006 unique companies, most with 1-2 postings each —
too sparse for one-hot encoding, but well-suited to **smoothed out-of-fold target encoding**, which
shrinks small-sample companies toward the global mean instead of overfitting to them).

Adding company + title + location target encoding (with smoothing factors of 10, 15, and 20 pseudo-samples
respectively) raised the result to **R² = 0.681** — the final, shipped model.

---

## 3. Feature Engineering — Every Feature, and Why

21 raw engineered columns (17 numeric + 4 categorical before encoding, expanding to 48 columns after
one-hot encoding of categoricals):

| Feature | What it captures | India-specific reasoning |
|---|---|---|
| `experience_mid`, `experience_breadth` | Midpoint and width of the posting's required experience range | Strongest single predictor everywhere tested (r≈0.69 alone) |
| `estimated_age`, `age_band` *(age_band kept, raw age dropped)* | Career-stage proxy: `22 + experience_mid` (India's typical graduation age) | **Never a real demographic field** — deliberately excluded from the final feature set as a raw number since it's mathematically redundant with experience; `age_band` (5 bins) kept for non-linear breakpoints. Used only to study aggregate role-pricing patterns, never to score an individual. |
| `experience_post_plateau` | `max(0, experience_mid − 14)` | Tests the widely-discussed Indian IT "senior plateau" hypothesis explicitly — found to be a weak signal at the *posting* level (see notebook Section 7 for the honest discussion of why) |
| `seniority_score`, `is_management_title` | Regex-parsed from job title (Intern=0 → CXO=9) | Reduces thousands of free-text titles to one interpretable ordinal signal |
| `is_metro`, `is_it_hub`, `is_remote`, `is_hybrid`, `num_locations_listed`, `primary_location_bucketed` | Curated Tier-1 city and IT-hub lists (Bengaluru, Hyderabad, Pune, Chennai, Gurugram, etc.) | Captures India's real geographic pay premium; confirmed empirically (+lift for metro/hub postings) |
| `skills_count`, `high_value_skill_count`, `has_leadership_skill` | Raw skill count (found to be a platform display artifact, low signal) vs. a curated list of 2025 high-value skills (Python, AWS, ML, Cloud, etc. — much stronger signal, confirmed by correlation) | |
| `has_rating`, `rating_bucket`, `reviews_count_log`, `company_size_proxy` | Company reputation signals with a missing-indicator pattern (absent rating ≠ rating of 0) | Found a genuinely counter-intuitive result: highly-rated companies show *lower* median salary — likely large-scale bulk junior hiring at big IT-services firms — reported honestly rather than forced into the expected narrative |
| `posting_age_days`, `description_word_count`, `mentions_qualification` | Parsed posting recency and light text signals from the job description | Included for completeness; honestly weak predictors, noted as such |
| `company_enc`, `title_enc`, `location_enc` | **Out-of-fold smoothed target encoding** — the single biggest lever, adding ~0.09 R² | Encodes real company/title/location pay-scale identity without the overfitting risk of raw one-hot on high-cardinality text |

**Explicit leakage exclusions:** `minimumSalary`, `maximumSalary`, `salary` (define the target — never
usable as features), raw `companyName`/`title`/`location`/`tagsAndSkills` text (replaced by engineered
versions above), `jobId`/`companyId` (identifiers, zero signal), `currency` (constant after filtering).

---

## 4. Model Selection — What Was Tried, What Won

**8 base models + 3 tuned variants**, evaluated with 5-fold cross-validation and 5 metrics (R², RMSE,
MAE, MAPE, MedAE) on a stratified 80/20 train/test split (stratified by salary decile, so both sets
represent the full salary spectrum):

| Model | Test R² |
|---|---|
| Linear Regression / Ridge / Lasso | ~0.47 |
| Random Forest | ~0.59 (but largest overfit gap: Train 0.75 vs Test 0.59) |
| Extra Trees | ~0.59 |
| Gradient Boosting | ~0.58 |
| XGBoost | ~0.59 (much smaller overfit gap than Random Forest) |
| LightGBM | ~0.59 |
| **LightGBM (Tuned) + target encoding** | **0.681** ✅ **final model** |

**Why LightGBM won:** best accuracy-to-overfitting ratio among tree ensembles, fastest training, and
handles the target-encoded high-cardinality features cleanly. Hyperparameters tuned via
`RandomizedSearchCV` (15 iterations, 3-fold CV): `n_estimators=500, num_leaves=90, learning_rate=0.03,
subsample=0.8, colsample_bytree=0.6, min_child_samples=20`.

**Feature importance** was computed **three ways** deliberately — split-count (LightGBM's default,
found to be *misleading*, biased toward continuous features), gain-based, and permutation importance
on the held-out test set. Gain-based and permutation importance agreed with each other and with the
raw EDA correlations; split-count did not — a genuine methodological lesson documented in the notebook.

**Quantile range models:** two additional `GradientBoostingRegressor` models (quantile loss,
α=0.10 and α=0.90) provide a calibrated `[P10, P90]` salary *range* rather than a single number —
verified at **79.3% empirical coverage** against an 80% target, i.e. genuinely well-calibrated, not
just plausible-looking.

### Final Model Metrics

| Metric | Value |
|---|---|
| Test R² | **0.6812** |
| Test RMSE | ₹4,02,130 |
| Test MAE | ₹2,08,878 |
| Train R² | 0.7907 (train/test gap is normal and expected, not alarming) |
| Quantile coverage | 79.3% (target 80%) |
| Training rows | 32,583 |
| Known companies | 11,006 |
| Known titles | 20,275 |

---

## 5. Fairness / Limitation Check (done honestly, not skipped)

Segmented error analysis across career-stage groups revealed a real, disclosed limitation: **absolute
error grows substantially for senior profiles** (MAE ≈₹86K for 0-2yr roles vs. ≈₹7.3L+ for 15+yr
roles). This is partly scale (senior salaries are numerically larger) but also genuine — senior/executive
compensation depends on negotiation and specialization that isn't observable in a job posting. This is
surfaced explicitly to the end user in the app (a visible low-confidence warning for senior profiles),
not hidden behind a single aggregate R² number.

---

## 6. Complete File Reference

```
SmartPay_India/
├── README.md                                    Top-level project overview (this journey, summarized)
├── notebook/
│   └── employee_salary_prediction_INDIA.ipynb   Full training pipeline — see below
└── app/
    ├── README.md                                Full technical docs: architecture, setup, deployment
    ├── app.py                                   Main Streamlit application — RUN THIS FILE
    ├── reconciliation_engine.py                 Combines all signals into one final answer
    ├── prediction_engine.py                     Feature engineering + model inference + fuzzy matching
    ├── data_store.py                            SQLite persistence layer
    ├── retrain_pipeline.py                      Full model retrain incorporating user contributions
    ├── external_lookup.py                       Live web search via Tavily (free tier)
    ├── train_production_model.py                One-time script that built artifacts/ (rebuild reference)
    ├── requirements.txt                         Python dependencies
    ├── artifacts/                               Trained model + lookup tables (ready to use)
    │   ├── model.pkl                            The LightGBM model (4.1 MB)
    │   ├── preprocessor.pkl                     Fitted sklearn ColumnTransformer (imputers/scalers/encoders)
    │   ├── quantile_p10.pkl / quantile_p90.pkl  Range prediction models
    │   ├── company_lookup.pkl                   11,006 companies → target-encoded value (344 KB)
    │   ├── title_lookup.pkl                     20,275 titles → target-encoded value (1.0 MB)
    │   ├── location_lookup.pkl                  Location → target-encoded value
    │   ├── industry_lookup.pkl                  7 industry categories → average salary
    │   ├── industry_keywords.pkl                Keyword lists used to guess industry from company name
    │   └── metadata.json                        All metrics, feature lists, smoothing params, versioning
    └── data/
        └── smartpay_community.db                SQLite DB for user contributions (starts empty)
```

### 6.1 Notebook — `employee_salary_prediction_INDIA.ipynb`

85 cells across **18 sections**, fully executed with **zero errors**: business problem framing →
environment setup (every library choice justified) → data loading → data quality audit (with charts
driving each filtering decision) → target transformation → feature engineering (the India-specific
core) → leakage audit (explicit table of every column's fate) → EDA → stratified train/test split →
preprocessing pipeline → 8-model benchmark → hyperparameter tuning → deep-dive evaluation (residuals,
worst-error inspection) → 3-way feature importance → segmented fairness analysis → quantile range
models → end-to-end inference demo → deployment artifact export → conclusions/ethics/limitations.

### 6.2 Application files — usage detail

| File | What it does | Key functions/classes |
|---|---|---|
| **`app.py`** | The Streamlit UI. 5 tabs: **Predict** (enter a profile, get a reconciled answer), **Contribute** (submit real salary data), **Admin: Retrain** (password-protected — review flagged submissions, trigger full retrain, check web search status), **Model Quality** (metrics dashboard), **About** | Run with `streamlit run app.py` |
| **`reconciliation_engine.py`** | Combines every signal into one weighted answer. See §7 below for full detail | `reconcile(profile, engine, use_web_search)` |
| **`prediction_engine.py`** | Loads artifacts, engineers features from a raw profile (mirrors the notebook exactly), checks known/unknown status, does fuzzy title matching | `PredictionEngine` class; `.predict_raw()`, `.fuzzy_match_title()`, `.guess_industry()` |
| **`data_store.py`** | SQLite abstraction — contributions, model version history, lookup overrides | `DataStore` class; swap this file's internals to move to a cloud DB later |
| **`retrain_pipeline.py`** | Rebuilds the entire model from [original 32,583 rows + all verified user contributions] | `retrain(data_store)` — run via the Admin tab or standalone |
| **`external_lookup.py`** | Live web search for unknown companies via Tavily | `try_external_lookup(profile)` |
| **`train_production_model.py`** | The original script that produced `artifacts/` — reference only, not needed for normal use | Run standalone to rebuild from scratch |

---

## 7. How The Whole System Works — Step by Step

### 7.1 A prediction request, end to end

1. User enters a profile in the **Predict** tab (title, company, experience, location, skills, optional rating).
2. `reconciliation_engine.reconcile()` is called, which:
   - Calls `PredictionEngine.predict_raw()` — engineers all 21 features exactly as the notebook did,
     checks the trained lookup tables for the company/title, applies the preprocessor + LightGBM model,
     and gets a point estimate + [P10,P90] range.
   - **If company and title are both recognized** → that's the primary signal (weight 30), done.
   - **If not recognized:**
     - Tries **fuzzy title matching** (weight 15) via `difflib` against 20,275 known titles.
     - Tries **live web search** via Tavily (weight 12), targeting salary-specific sites
       (`ambitionbox.com`, `glassdoor.co.in`, `naukri.com`, `payscale.com`) and parsing LPA/Lakh
       figures out of the results with a regex layer (handles "10-15 LPA", "6L-9L", single figures, etc.).
     - Guesses the company's **industry** from its name (weight 8) and uses that industry's real
       average salary (computed from the training data) as a fallback signal.
     - Always keeps the **role/experience/location-only baseline** (weight 5) in the mix.
   - All available signals are **blended in log-salary space**, weighted by trust tier.
   - If signals disagree substantially, the shown range is **widened proportionally** rather than
     silently averaging the disagreement away, and a warning is shown.
3. The final answer displays: a reconciled range + point estimate, a confidence badge, and an
   expandable **"How we calculated this"** breakdown listing every signal that contributed and its weight.

### 7.2 A user contribution, end to end

1. User submits a real salary in the **Contribute** tab (company, title, experience, location, actual salary).
2. The submission is compared against what the model would have predicted for that exact profile.
3. **Within 0.3x-3x of the prediction** → automatically `verified`, and immediately blended into the
   lookup tables (`DataStore.refresh_lookup_overrides()`) — the very next prediction for that
   company/title reflects it, no retrain needed.
4. **Outside that range** → flagged for manual review in the Admin tab (protects against typos, joke
   submissions, unit confusion) rather than being silently trusted.

### 7.3 A full retrain, end to end

1. Admin clicks **"Retrain Now"** (password-protected).
2. `retrain_pipeline.retrain()` combines the original 32,583-row training data with all verified
   community contributions, re-engineers every feature from scratch, rebuilds the target-encoding
   lookup tables (now including any new companies/titles from contributions), and refits the LightGBM
   model + quantile models.
3. New artifacts are saved to a timestamped versioned folder, then **atomically swapped** into the live
   `artifacts/` directory — and the version is logged in the database's model history table.

---

## 8. Setup & Running the Project

```bash
cd SmartPay_India/app
pip install -r requirements.txt
streamlit run app.py
```

Works immediately with the pre-trained model — no setup required for basic prediction.

**Optional: enable live web search** (free, ~2 minutes):
1. Sign up at [app.tavily.com](https://app.tavily.com) (no credit card, 1,000 free searches/month)
2. `export TAVILY_API_KEY="tvly-..."` (or add to `.streamlit/secrets.toml` for cloud deployment)

**Optional: change the admin password** (default is `changeme123`):
```bash
export SMARTPAY_ADMIN_PASSWORD="your-real-password"
```

**⚠️ Deployment note:** contributions are stored in a local SQLite file. This works for local/self-hosted
use but will **not persist** on platforms with ephemeral filesystems (e.g. Streamlit Community Cloud) —
see `app/README.md` for how to swap in a cloud database.

---

## 9. Known Limitations (stated plainly, not buried)

- Predicts an advertised **role-level compensation band** from job postings, not a verified individual paycheck.
- Only ~34% of scraped postings disclosed salary — a real selection-bias limitation in the source data.
- Accuracy is meaningfully lower for senior/10+ year experience profiles (surfaced explicitly to users).
- `age_band` is a derived experience proxy, **never a real demographic field** — must not be used for
  individual age-based hiring decisions.
- The contribution sanity-gate (0.3x-3x check) is heuristic, not foolproof against determined bad actors.
- Web search results are single-source and shown as one signal among several, not verified ground truth.
