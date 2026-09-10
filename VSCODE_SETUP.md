# Running SmartPay India in VS Code

Yes — this is a standard Python project, so it works cleanly in VS Code. This file covers the
VS Code-specific setup; see the main `README.md` for what the project actually does.

---

## 1. Open the Project

```
File -> Open Folder... -> select SmartPay_India/
```

Open the **whole `SmartPay_India/` folder**, not just `app/` — the `.vscode/` config, the notebook, and
the raw dataset all live at this level, and the app's code expects to find the dataset via a relative
path (`../data/raw/...`) from inside `app/`.

## 2. Install Recommended Extensions

VS Code will prompt you automatically (a small popup, bottom-right) because of `.vscode/extensions.json`.
If it doesn't, install manually from the Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`):

- **Python** (ms-python.python) — required
- **Pylance** (ms-python.vscode-pylance) — required, gives you IntelliSense/autocomplete
- **Jupyter** (ms-toolsai.jupyter) — required to run the notebook inside VS Code
- **Python Debugger** (ms-python.debugpy) — required for the debug configs below

## 3. Create the Virtual Environment (once)

Open a terminal in VS Code (`` Ctrl+` `` / `` Cmd+` ``):

```bash
cd app
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

(Use `venv\Scripts\pip` instead of `./venv/bin/pip` on Windows.)

## 4. Select the Interpreter

`Ctrl+Shift+P` / `Cmd+Shift+P` -> type **"Python: Select Interpreter"** -> choose the one at
`app/venv/bin/python3` (VS Code usually finds it automatically since `.vscode/settings.json` already
points at it — you may just need to confirm it once).

Do this **before** opening the notebook, so Jupyter cells run against the same environment.

## 5. Run the App

**Option A — Terminal** (simplest):
```bash
cd app
./venv/bin/streamlit run app.py
```

**Option B — Debugger** (lets you set breakpoints in `app.py`, `prediction_engine.py`, etc.):
Press `F5`, or open the Run and Debug panel (`Ctrl+Shift+D`) and pick **"Streamlit: Run App"** from the
dropdown — this is already configured in `.vscode/launch.json`. Streamlit opens in your browser as
normal; breakpoints in any imported module will pause execution in VS Code.

## 6. Run/Retrain the Model From VS Code

Two more configs are ready in the Run and Debug dropdown:
- **"Python: Train Model From Scratch"** — runs `train_model.py`, ~2-4 minutes, regenerates `artifacts/`
- **"Python: Retrain Pipeline (standalone)"** — runs `retrain_pipeline.py` directly (normally triggered
  from the app's Admin tab instead, but useful for debugging the retrain logic in isolation)

## 7. Run the Notebook

Open `notebook/employee_salary_prediction_INDIA.ipynb` directly in VS Code — the Jupyter extension
renders it natively, no separate `jupyter notebook` server needed. Click **"Run All"** at the top, or
run cells one at a time with `Shift+Enter`. First run will prompt you to pick a kernel — choose the
`app/venv` interpreter from step 4 (you'll need `ipykernel` too: `./venv/bin/pip install ipykernel`
inside `app/`, or install it in a separate `notebook/venv` if you prefer keeping notebook and app
dependencies fully separate — see `notebook/requirements.txt`).

Takes 15-20 minutes end to end (it trains and compares 8 models before landing on the final tuned one)
and writes directly into `app/artifacts/` — no manual copying between notebook and app.

## 8. Building Your Own Application on Top of This

Some natural extension points, since you mentioned building your own application:

- **`app.py`** is the UI layer — add new `st.tabs()` entries here for new pages/features.
- **`reconciliation_engine.py`** is where signals get combined — add a new signal source (another
  data provider, a different model) by following the existing pattern: gather it, assign a trust
  weight, append to the `signals` list.
- **`feature_engineering.py`** is the single source of truth for feature rules — change something here
  and it's automatically consistent across the app, the retrain pipeline, and the notebook (they all
  import from this file). This is deliberate — see the top-level README for why.
- **`data_store.py`** currently uses SQLite — swap its internals for a real database if you outgrow a
  single-file store; nothing else in the app needs to change since everything else only calls its
  public methods.

## Troubleshooting

**"Import could not be resolved" (Pylance warnings on `feature_engineering`, etc.)**
Make sure the interpreter selected in step 4 is the `app/venv` one — Pylance resolves imports based on
whichever interpreter is active, and `.vscode/settings.json` already tells it to look inside `app/` for
local modules.

**Notebook kernel doesn't show up**
Run `./venv/bin/pip install ipykernel jupyter` inside `app/` (or your notebook-specific venv), then
reload the VS Code window (`Ctrl+Shift+P` -> "Developer: Reload Window").

**Streamlit debug config doesn't stop at breakpoints**
Confirm `"justMyCode": true` isn't hiding the file you're debugging — Streamlit's own internals are
skipped by design, but breakpoints in this project's own files (`app.py`, `prediction_engine.py`, etc.)
should work as-is.
