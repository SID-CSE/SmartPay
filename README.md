# SmartPay India — Complete Salary Prediction Application

A real, end-to-end salary prediction system: trained on real Naukri.com job postings, wrapped in a
complete web application that predicts, learns from user contributions, searches the web for
unknowns and for similar jobs, and reconciles every signal into one transparent answer.

**Ready to upload straight into VS Code and run.** `.vscode/` configs are included (interpreter path,
debug configs, recommended extensions) so it works immediately, not just technically opens.

---

## Part 1 — How It Works

### 1.1 The Model

Trained on **32,583 real, cleaned Naukri.com job postings** (from an original 97,929 — filtered to
disclosed INR salaries, outlier-capped). Predicts an annual salary **range** (not just a point estimate)
in Lakhs Per Annum, India's native convention.

**How the final model was chosen** — every step measured, not assumed:

| Stage | Test R² | What happened |
|---|---|---|
| Structured features only (experience, location tier, skills count, etc.) | 0.593 | Baseline: 8 models compared (Linear/Ridge/Lasso/Random Forest/Extra Trees/Gradient Boosting/XGBoost/LightGBM), LightGBM wins |
| + Company/title/location target encoding | 0.681 | The single biggest lever: out-of-fold smoothed encoding captures real pay-scale identity from 11,006 companies without overfitting to the ~2 postings/company median |
| + TF-IDF text features (skills + optional job description, bigrams) | **0.7215** | Final. Cross-validated stable (3-fold CV std = 0.0047) |

**Tested and honestly rejected along the way** (not hidden):
- CatBoost with native categorical handling: **0.656** — worse. Data too sparse per-company for its
  ordered encoding to beat manual target encoding.
- Multi-model ensembles (LightGBM+XGBoost+CatBoost, weighted or averaged): best result **0.681** —
  tied with plain LightGBM alone, no real gain.

**Final model:** LightGBM, tuned via `RandomizedSearchCV`, on target-encoded structured features +
TF-IDF text signal. **Test R² = 0.72, Test MAE ≈ ₹1.93 Lakh.**

### 1.2 The Application (not just a model in a notebook)

```
┌─────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│   Predict   │────▶│ reconciliation_engine │────▶│  One final answer   │
│   (a form)  │     │ (combines 6 signals)  │     │  + confidence badge │
└─────────────┘     └──────────────────────┘     │  + full breakdown   │
                              │                    └────────────────────┘
                     ┌────────┴────────┬──────────────┬─────────────┐
                     ▼                 ▼               ▼             ▼
              Trained model    Fuzzy title match   Web search    Industry
              (known company)  (difflib)           (Tavily)      fallback
```

**Reconciliation signal weights** (blended in log-salary space; disagreement widens the range honestly
instead of averaging it away):

| Signal | Weight | When used |
|---|---|---|
| Community-verified exact match | 40 | A user already reported real salary for this exact company+title |
| Trained model (company + title both known) | 30 | Default path for recognized entries |
| Fuzzy-matched title | 15 | e.g. "Sr Dev" → "Senior Developer" |
| Live web search (Tavily) | 12 | Company unrecognized; found a parseable figure online |
| Industry-similar average | 8 | Company name suggests an industry (e.g. "...Bank Ltd") |
| Role/experience/location baseline | 5 | Always included as the floor signal |

**Two more application layers, beyond the model:**

- **Contribute real data** → auto-sanity-checked against the model's own prediction → close matches
  blend in *instantly* (next prediction reflects it, no retrain needed); outliers get queued for admin
  review instead of silently trusted.
- **Similar Jobs search** (`job_search.py`) → a *different* live web search from the salary lookup —
  finds real, currently-posted listings (Naukri, LinkedIn, Indeed, Glassdoor) as market comparables,
  shown separately, never blended into the salary number.

### 1.3 One Codebase, No Drift

`feature_engineering.py` is the single place every rule lives (age bands, seniority scoring, city
tiers, target encoding, TF-IDF, industry guessing). The notebook, `train_model.py`, and
`retrain_pipeline.py` **all import this same file** — they cannot silently compute a feature
differently from each other. (An earlier version of this project didn't do this, and the notebook and
the live app ended up as two incompatible pipelines. Fixed by construction now, not by promise.)

---

## Part 2 — Project Structure

```
SmartPay_India/
├── README.md                 <- you are here
├── VSCODE_SETUP.md            <- VS Code specifics: debug configs, kernels, troubleshooting
├── .vscode/                   <- ready-to-use workspace config (interpreter, debug, extensions)
├── data/raw/                  <- the raw dataset (already included, nothing to download)
├── notebook/
│   ├── employee_salary_prediction_INDIA.ipynb   <- full pipeline, executed, 0 errors, every step explained
│   └── requirements.txt
└── app/
    ├── app.py                  <- run this: streamlit run app.py
    ├── feature_engineering.py  <- single source of truth for every feature rule
    ├── train_model.py          <- raw .xlsx in, artifacts/ out (~2-4 min)
    ├── prediction_engine.py    <- core inference
    ├── reconciliation_engine.py<- combines model + web search + fuzzy match + industry fallback
    ├── job_search.py           <- NEW: finds real similar job listings (separate from salary lookup)
    ├── external_lookup.py      <- salary web-lookup for unknown companies
    ├── data_store.py           <- SQLite: contributions, model version history
    ├── retrain_pipeline.py     <- full retrain on [base data + verified contributions]
    ├── requirements.txt
    ├── artifacts/               <- trained model, ready to use immediately
    └── data/                    <- cleaned training data + community database (starts empty)
```

---

## Part 3 — Run It in VS Code (step by step)

### 3.1 Open and Set Up

```
File → Open Folder... → select this SmartPay_India/ folder (the whole thing, not just app/)
```

VS Code will prompt to install recommended extensions (Python, Pylance, Jupyter, Python Debugger) —
accept the prompt, or install manually from the Extensions panel (`Ctrl+Shift+X`).

Open a terminal in VS Code (`` Ctrl+` ``) and create the virtual environment:

```bash
cd app
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

*(Windows: `venv\Scripts\pip install -r requirements.txt`)*

Then select this interpreter: `Ctrl+Shift+P` → **"Python: Select Interpreter"** → pick
`app/venv/bin/python3` (VS Code usually detects it automatically).

### 3.2 Run the App (the model already included works immediately — no training required first)

**Terminal:**
```bash
./venv/bin/streamlit run app.py
```

**Or with the debugger** (breakpoints work): press `F5`, or open Run & Debug (`Ctrl+Shift+D`) and pick
**"Streamlit: Run App"** from the dropdown (already configured in `.vscode/launch.json`).

Your browser opens to the app. Try the **Predict** tab with a known company (e.g. "Infosys",
"Software Engineer", Bengaluru) to see a high-confidence result, then try a made-up company name to see
the honest low-confidence fallback kick in.

### 3.3 Check the Model Yourself — Score, Selection, Everything

Three ways to verify the model without taking anything on faith:

1. **In the app** — open the **Model Quality** tab: shows Test R², MAE, RMSE, and quantile calibration
   live, pulled directly from `artifacts/metadata.json`.
2. **Directly in a terminal:**
   ```bash
   cat app/artifacts/metadata.json
   ```
   Shows the exact test R², MAE, training row count, known companies/titles, and model version string.
3. **Retrain from scratch and watch it happen:**
   ```bash
   cd app
   ./venv/bin/python3 train_model.py
   ```
   Prints every stage live: raw row count → cleaning → feature engineering → target encoding → TF-IDF
   fitting → LightGBM training → final Test R². Takes 2-4 minutes. Or use the **"Python: Train Model
   From Scratch"** debug config to step through it.
4. **See the full model-selection process** (all 8 models compared, tuning, feature importance,
   residual analysis, fairness check) — open the notebook (see 3.4). This is where you can watch
   LightGBM actually win against 7 other candidates, not just take the final choice on faith.

### 3.4 Run the Notebook

Open `notebook/employee_salary_prediction_INDIA.ipynb` directly in VS Code (the Jupyter extension
renders it natively). Pick the `app/venv` kernel when prompted (needs `ipykernel`, already in
`app/requirements.txt`). Click **Run All**, or step through with `Shift+Enter`.

Takes 15-20 minutes end-to-end — it's the full pipeline: data audit → feature engineering → leakage
audit → EDA → 8-model benchmark → TF-IDF → hyperparameter tuning → evaluation → feature importance →
fairness analysis → quantile models → saves directly into `app/artifacts/` (no manual copying).

### 3.5 Optional Enhancements

**Live web search** (salary lookup for unknown companies + similar-jobs search), free:
1. Sign up at [app.tavily.com](https://app.tavily.com) — no card, 1,000 free searches/month
2. `export TAVILY_API_KEY="tvly-..."` before running the app (or add to `.streamlit/secrets.toml`)

**Admin panel** (review contributions, trigger retrains) — default password `changeme123`, change via:
```bash
export SMARTPAY_ADMIN_PASSWORD="your-real-password"
```

---

## Part 4 — Building Your Own Application On This

- **Add a new page/feature** → new `st.tabs()` entry in `app.py`
- **Add a new prediction signal** → follow the pattern in `reconciliation_engine.py`: gather the
  signal, assign a trust weight, append to the `signals` list
- **Change a feature rule** → edit `feature_engineering.py` once; the app, retrain pipeline, and
  notebook all stay consistent automatically
- **Move beyond SQLite** → `data_store.py`'s public methods are the only contract other files rely on;
  swap its internals for a real database without touching anything else

---

## Known Limitations (stated plainly)

- Predicts an advertised **role-level compensation band** from job postings, not a verified individual
  paycheck.
- Only ~34% of scraped postings disclosed salary — a real selection-bias limitation in the source data.
- Accuracy is meaningfully lower for senior/10+ year experience profiles (surfaced explicitly in the app).
- `age_band` is a derived experience proxy, **never a real demographic field** — never usable for
  individual age-based decisions.
- Exact-string company/title matching means close variants (e.g. "TCS" vs "Tata Consultancy Services")
  aren't automatically unified — this is exactly why fuzzy title matching and the industry fallback exist.

**Next:** read `VSCODE_SETUP.md` for IDE-specific troubleshooting, or `app/README.md` for deeper
technical documentation on the reconciliation engine and deployment notes.
