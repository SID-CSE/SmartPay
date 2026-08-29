# SmartPay India — Complete Prediction System

Predict → Contribute real data → Model learns from it → Honest about what it doesn't know.

Tested end-to-end (real Streamlit server launched, health-checked, zero errors) before delivery.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Default admin password is `changeme123` — **change this** by setting the environment variable before running:

```bash
export SMARTPAY_ADMIN_PASSWORD="your-real-password"
streamlit run app.py
```

---

## Architecture — how the 3 pieces actually fit together

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   Predict   │────▶│  prediction_      │────▶│   artifacts/       │
│   tab       │     │  engine.py        │     │   (model, lookups) │
└─────────────┘     └──────────────────┘     └───────────────────┘
                            │
                            │ checks: is company/title known?
                            ▼
                     ┌──────────────────┐
                     │  data_store.py    │◀──── Contribute tab
                     │  (SQLite)         │      (new submissions)
                     └──────────────────┘
                            │
                            │ admin clicks "Retrain Now"
                            ▼
                     ┌──────────────────┐
                     │ retrain_pipeline  │────▶ new artifacts_versions/v_TIMESTAMP/
                     │ .py               │      (atomically swapped into artifacts/)
                     └──────────────────┘
```

### Two speeds of "learning from user data" — by design, not accident

| | Immediate blend | Full retrain |
|---|---|---|
| **When** | The moment a submission passes the sanity check | Admin clicks "Retrain Now" |
| **What happens** | That company/title's average nudges toward the new data point | Entire model refit from scratch on base data + all verified contributions |
| **Cost** | Instant, free | ~1-2 minutes (rebuilds encodings + refits LightGBM + quantile models) |
| **Why both exist** | Refitting a gradient-boosted model on every single form submission would be wasteful and would let one bad/joke submission destabilize predictions before it has corroborating data points. The immediate blend gives instant feedback; the periodic retrain bakes verified data permanently into the core model. |

### What happens for an unknown company/title

**Live web search is now active**, powered by [Tavily](https://tavily.com) — chosen because, as of
August 2026, it's the most viable genuinely-free option left: Bing's Search API was retired
(Aug 2025), Brave's free tier was withdrawn (late 2025), and Google Custom Search is closed to new
signups and shutting down entirely (Jan 2027). Tavily gives **1,000 free searches/month, recurring,
no credit card required**, and returns clean synthesized answers instead of raw HTML to parse.

**Setup (2 minutes, do this before running the app):**
1. Sign up at [app.tavily.com](https://app.tavily.com) (email or Google/GitHub, no card)
2. Copy your API key (starts with `tvly-`)
3. Set it one of two ways:
   ```bash
   export TAVILY_API_KEY="tvly-..."          # local/self-hosted
   ```
   or add to `.streamlit/secrets.toml`:
   ```toml
   TAVILY_API_KEY = "tvly-..."                # Streamlit Cloud
   ```
4. Done — no code changes needed. The app auto-detects the key.

**What happens without a key configured:** everything still works — predictions for unknown
companies just fall back to the role/experience/location-only estimate with an honest message,
exactly as before. Nothing breaks if you skip this setup.

**How it decides what to search:** queries target salary-specific sites directly
(`site:ambitionbox.com OR site:glassdoor.co.in OR site:naukri.com OR site:payscale.com`) rather than
open web search, since generic results rarely contain a clean salary figure. A regex layer then
extracts LPA/Lakh figures from Tavily's synthesized answer or result snippets — tested against 9
real-world phrasing patterns (ranges, single figures, per-number units like "6L-9L") including a
false-positive guard so it doesn't mistake an unrelated rupee figure (like a course fee) for a salary.

**Important: the web result is shown separately, not blended into the model's prediction.** A single
web search result is not verified data the way a community contribution is — showing it alongside
the model estimate (rather than silently merging the two) keeps the provenance honest.

Check search status and run a test lookup anytime in the **Admin tab**.

### Sanity gating on contributions (basic spam/error protection)

When someone submits a real salary, it's compared to what the model would have predicted for that
exact profile:
- Within **0.3x–3x** of the prediction → auto-`verified`, blended immediately.
- Outside that range → `flagged_review`, held for an admin to approve or reject in the Admin tab.

This isn't foolproof (a determined bad actor could still submit plausible-looking fake data), but it
catches the obvious cases (typos, joke submissions, unit confusion) without needing manual review of
every single contribution.

---

## ⚠️ Deployment note — read before deploying anywhere but your own machine

**Contributions are stored in a local SQLite file (`data/smartpay_community.db`).** This works great
for local use or a self-hosted server with a persistent disk. It will **NOT** work correctly on
**Streamlit Community Cloud** or similar platforms with ephemeral filesystems — the database (and any
retrained model versions) will vanish on every restart/redeploy.

If you deploy there, swap `data_store.py`'s internals for a real cloud database (Postgres via
Supabase/Neon is the easiest drop-in) — the rest of the app only calls `DataStore`'s public methods,
so this is a contained change to one file.

---

## The Reconciliation Engine — combining every signal into ONE answer

Rather than showing the model's prediction and a web search result as two separate numbers,
`reconciliation_engine.py` gathers **every available signal** for a profile and blends them into
a single answer, weighted by how much each source should actually be trusted:

| Signal | Trust weight | When it's used |
|---|---|---|
| Community-verified exact match | 40 (highest) | Real reported data exists for this exact company+title |
| Trained model (company + title both known) | 30 | The validated ~0.68 R² model recognizes both |
| Fuzzy-matched title | 15 | e.g. "Sr Dev" matches known title "Senior Developer" (via `difflib`) |
| Live web search (Tavily) | 12 | Company is unrecognized; a parseable figure was found online |
| Industry-similar average | 8 | Company name suggests an industry (e.g. "...Bank Ltd" → Finance/Banking) with a known average |
| Role/experience/location baseline | 5 (always included) | The floor signal — always available even when nothing else is |

**Signals are blended in log-salary space** (consistent with how the whole system is trained), and
weights are a deliberate, documented judgment call — not fitted — reflecting how much statistical
backing each source actually has (thousands of training rows vs. a single web search hit).

**When signals disagree substantially** (e.g. the baseline estimate and a web search result point to
very different numbers), the final range is **widened proportionally rather than silently averaged
away** — an honest reconciliation flags conflict instead of hiding it. Every prediction shows an
expandable "How we calculated this" breakdown listing every signal that contributed and its weight —
this is deliberately not a black box.

### Two additional signals I added beyond what was asked for

1. **Fuzzy title matching** (`prediction_engine.py`'s `fuzzy_match_title`) — an exact-string company/title
   lookup misses obvious near-matches ("Sr Software Developer" vs. the trained data's "Sr. Software
   Developer"). Tested against the real 20,275 known titles — correctly matches abbreviations, typos,
   and reordered phrasing before falling back to the generic baseline.
2. **Industry-guessed company fallback** (`artifacts/industry_lookup.pkl`) — when a company is
   completely unknown, guessing its industry from keywords in the name (e.g. "Bank", "Hospital",
   "Technologies") and borrowing that industry's real average (computed from the actual 32,583-row
   training set) is a much better fallback than the pure global mean. Real computed averages range from
   ₹12.87-13.42 (log scale) across 7 industry buckets, each with hundreds-to-thousands of supporting rows.

---



| File | Purpose |
|---|---|
| `app.py` | Main Streamlit app (5 tabs: Predict, Contribute, Admin, Model Quality, About) |
| `reconciliation_engine.py` | Combines model + web search + fuzzy match + industry fallback into one weighted answer |
| `prediction_engine.py` | Feature engineering + known/unknown detection + fuzzy title matching + blended inference |
| `data_store.py` | SQLite persistence layer (contributions, model version history, lookup overrides) |
| `retrain_pipeline.py` | Full batch retrain incorporating verified contributions |
| `external_lookup.py` | Live web search via Tavily (free tier, see setup above) |
| `train_production_model.py` | One-time script that produced the initial `artifacts/` (re-run only if you want to rebuild from raw scratch) |
| `artifacts/` | Current live model: preprocessor, LightGBM model, quantile range models, company/title/location/industry lookup tables, metadata |
| `data/smartpay_community.db` | SQLite database (starts empty) |

## Current Model

- **Test R²: 0.681** (company/title/location target-encoded LightGBM — the strongest result across
  every dataset and configuration tested)
- Trained on 32,583 real, cleaned Naukri.com job postings
- Knows 11,006 companies and 20,275 title variants out of the box; everything else falls back to
  role/experience/location-only estimation with an honest low-confidence flag

## Known Limitations (carried over from the training notebook — still true here)

- Predicts an advertised **role-level compensation band**, not a verified individual paycheck.
- Accuracy is meaningfully lower for senior/10+ year profiles (wider uncertainty range reflects this honestly).
- `age_band` is a derived experience proxy, never a real demographic field — must not be used for
  individual age-based decisions.
- The sanity-gate on contributions is heuristic, not a guarantee against bad data — periodically
  review the Admin tab's flagged queue.
