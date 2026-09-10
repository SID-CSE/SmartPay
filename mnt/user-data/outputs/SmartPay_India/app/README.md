# SmartPay India — App Technical Documentation

Full documentation for the application layer. See the top-level `README.md` for the project overview
and the from-scratch training procedure.

---

## Consistency by construction

Every file in this app that needs feature-engineering rules (`prediction_engine.py`,
`retrain_pipeline.py`, `train_model.py`) imports them from **`feature_engineering.py`** — there is
exactly one definition of "what counts as a metro city," "how seniority is scored from a job title,"
"what the age-band boundaries are," etc. This is the direct fix for an earlier version of this project
where the training notebook and the live app quietly diverged and produced incompatible models.

## File Reference

| File | What it does |
|---|---|
| `app.py` | Streamlit UI — 5 tabs: Predict, Contribute, Admin: Retrain, Model Quality, About |
| `feature_engineering.py` | **Single source of truth** for every feature rule, target encoding, and industry guesser |
| `train_model.py` | Standalone training script: raw `.xlsx` in, `artifacts/` + `cleaned_training_data.csv` out |
| `job_search.py` | Searches the web (Tavily) for real, currently-posted similar job listings -- distinct from `external_lookup.py`'s salary figure search |
| `prediction_engine.py` | Loads `artifacts/`, engineers features from a raw profile, checks known/unknown, fuzzy title matching |
| `reconciliation_engine.py` | Combines model + fuzzy match + web search + industry fallback into one weighted answer |
| `data_store.py` | SQLite persistence — contributions, model version history, lookup overrides |
| `retrain_pipeline.py` | Full retrain on `[cleaned_training_data.csv + verified contributions]` |
| `external_lookup.py` | Live web search via Tavily (free tier) |

## The Reconciliation Engine — combining every signal into ONE answer

| Signal | Trust weight | When it's used |
|---|---|---|
| Community-verified exact match | 40 (highest) | Real reported data exists for this exact company+title |
| Trained model (company + title both known) | 30 | The validated ~0.72 R² model recognizes both |
| Fuzzy-matched title | 15 | e.g. "Sr Dev" matches known title "Senior Developer" (via `difflib`) |
| Live web search (Tavily) | 12 | Company is unrecognized; a parseable figure was found online |
| Industry-similar average | 8 | Company name suggests an industry (e.g. "...Bank Ltd") with a known average |
| Role/experience/location baseline | 5 (always included) | The floor signal — always available |

Signals are blended in log-salary space. When they disagree substantially, the final range is
**widened proportionally rather than silently averaged away**, and a warning is shown. Every
prediction includes an expandable "How we calculated this" breakdown — not a black box.

## New Feature: Similar Jobs Search

`job_search.py` searches the web for real, currently-posted job listings similar to a profile (title + location + experience), targeting naukri.com, linkedin.com/jobs, indeed.co.in, and glassdoor.co.in. Uses the same Tavily API key as the salary web-lookup feature below -- no separate signup needed. Shown as a distinct "Similar Jobs Currently Posted" section in the Predict tab, never blended into the salary estimate itself (market comparables, not model input).

## Setup: Live Web Search (optional, free)

1. Sign up at [app.tavily.com](https://app.tavily.com) — no card needed, 1,000 free searches/month
2. `export TAVILY_API_KEY="tvly-..."` (or add to `.streamlit/secrets.toml` for cloud deployment)
3. Done — auto-detected. Without a key, unknown companies just skip straight to the other signals.

## Setup: Admin Password

Default is `changeme123`. Change it:
```bash
export SMARTPAY_ADMIN_PASSWORD="your-real-password"
```

## Retraining

Two ways, both consistent because both use `feature_engineering.py`:

- **From the Admin tab** — click "Retrain Now" after some contributions have accumulated. Reads
  `data/cleaned_training_data.csv` (produced whenever `train_model.py` or the notebook runs) plus all
  verified contributions, retrains, and atomically swaps the new model into `artifacts/`.
- **From scratch** — `python train_model.py` (see top-level README).

## ⚠️ Deployment Note

Contributions and the cleaned training data are stored as local files (SQLite + CSV). This works for
local/self-hosted use. On platforms with an ephemeral filesystem (e.g. Streamlit Community Cloud),
these will **not persist** across restarts — swap `data_store.py`'s internals for a real cloud database
(Postgres via Supabase/Neon is the easiest drop-in); the rest of the app only calls `DataStore`'s public
methods, so this is a contained change to one file.

## Current Model

- **Test R²: ~0.72** (target-encoded LightGBM + TF-IDF text features on skills and optional job description)
- Trained on 32,583 real, cleaned Naukri.com postings
- Knows 11,006 companies and 20,275 title variants out of the box
- Feature journey: 0.59 (base features) -> 0.68 (+ company/title/location target encoding) -> 0.72 (+ TF-IDF text signal, bigrams, retuned hyperparameters)
- Tested and rejected along the way: CatBoost native categoricals (0.656), simple/weighted multi-model ensembles (0.681, no better than plain LightGBM) -- reported honestly, not hidden
- Exact-string matching means close name variants aren't automatically unified — fuzzy title
  matching and the industry fallback exist specifically to soften this for unknown cases

## Known Limitations

- Predicts an advertised role-level compensation band, not a verified individual paycheck.
- Accuracy is meaningfully lower for senior/10+ year profiles (surfaced explicitly to users).
- `age_band` is a derived experience proxy, never a real demographic field.
- The contribution sanity-gate (0.3x-3x check) is heuristic, not foolproof against bad actors.
- Web search results are single-source and shown as one signal among several, not verified ground truth.
