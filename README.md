# SmartPay India — Complete Salary Prediction Project

A real, honestly-evaluated, end-to-end salary prediction system: trained on real Naukri.com job
postings, wrapped in a complete web application that predicts, learns from user contributions,
searches the web for unknowns, and reconciles every signal into one transparent answer.

---

## Project Structure

```
SmartPay_India/
├── README.md                          <- you are here
├── notebook/
│   └── employee_salary_prediction_INDIA.ipynb   <- full training pipeline (18 sections, executed, 0 errors)
└── app/
    ├── app.py                         <- run this: streamlit run app.py
    ├── reconciliation_engine.py       <- combines model + web search + fuzzy match + industry fallback
    ├── prediction_engine.py           <- core model inference + feature engineering
    ├── data_store.py                  <- SQLite persistence (contributions, model versions)
    ├── retrain_pipeline.py            <- full model retrain on community data
    ├── external_lookup.py             <- live web search via Tavily (free tier)
    ├── train_production_model.py      <- one-time script that built artifacts/
    ├── artifacts/                     <- trained model, lookups, metadata (ready to use)
    ├── data/                          <- SQLite DB (starts empty)
    ├── requirements.txt
    └── README.md                      <- full technical documentation, read this before deploying
```

## Quick Start

```bash
cd SmartPay_India/app
pip install -r requirements.txt
streamlit run app.py
```

That's it — the app works immediately with the pre-trained model. Web search and community
contributions are optional enhancements layered on top (see `app/README.md` for setup).

## The Journey (why this project looks the way it does)

1. **Started with a generic toy dataset** — quickly replaced once real data was provided, because a
   clean synthetic dataset teaches the wrong lessons about what real salary prediction looks like.
2. **Built the full pipeline on real Naukri.com data** (98K postings) — honestly confronted real data
   quality problems (0-encoded "not disclosed" salaries, mixed currencies, extreme outliers) rather
   than hiding them.
3. **Got R²=0.59, and was asked "can we get to 0.90?"** — tested that question empirically rather than
   guessing: quantified that experience alone explains ~48% of variance, and that a genuine ceiling
   exists without company-identity signal.
4. **Tested 5 additional real datasets** (Glassdoor-India, a second Naukri-style scrape, the Stack
   Overflow Developer Survey 2025 both India-only and global) — none beat the original data, and the
   process itself proved something: when Linear Regression and XGBoost land within 0.03-0.05 R² of
   each other, that's a real information ceiling, not a modeling gap.
5. **Fixed the actual gap** — proper out-of-fold target encoding of company/title/location took the
   real result from 0.59 to **0.68**, the strongest defensible number across everything tested.
6. **Built a complete application around it** — predict, contribute, retrain, search the web, and
   reconcile every signal into one honest answer — catching and fixing two real bugs along the way
   (an "instant blend" feature that was UI-only until traced through the actual code path, and a
   disagreement-widening formula that could produce a nonsensical ₹0 lower bound).

## Final Model Performance

| Metric | Value |
|---|---|
| Test R² | **0.681** |
| Test MAE | ~₹2.09 Lakh |
| Trained on | 32,583 real, cleaned Naukri.com postings |
| Known companies | 11,006 |
| Known titles | 20,275 |
| Quantile range coverage | ~79% (target 80%) |

## Read Next

- **`app/README.md`** — full technical documentation: architecture, the reconciliation engine's
  signal-weighting logic, web search setup, deployment notes, and known limitations.
- **`notebook/employee_salary_prediction_INDIA.ipynb`** — the complete, executed training pipeline
  with every decision justified in place (why each library, why each feature, why each modeling choice).
