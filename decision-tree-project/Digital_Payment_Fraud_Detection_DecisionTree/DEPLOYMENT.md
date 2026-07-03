# 🚀 Deployment Guide — Streamlit Community Cloud

This project is deploy-ready. This guide covers the exact steps, plus a fix
for the most common failure mode: `ModuleNotFoundError` on a freshly deployed
app.

---

## ✅ Correct Repository Layout

Streamlit Cloud installs dependencies from **`requirements.txt`** found in
the **same directory as the app's "Main file path."** If your repo nests this
project inside a subfolder (e.g. `decisiontree/`), `requirements.txt` and
`runtime.txt` **must live in that same subfolder**, not just the repo root.

```
your-repo/                          <- GitHub repo root
└── decisiontree/                   <- or wherever you place this project
    ├── data/paysim_sample_15000.csv
    ├── notebook/Fraud_Detection_DecisionTree.ipynb
    ├── model/
    │   ├── decision_tree.pkl
    │   ├── label_encoder.pkl
    │   ├── feature_columns.pkl
    │   ├── model_metadata.json
    │   └── tree_rules.txt
    ├── assets/*.png
    ├── app.py
    ├── train_pipeline.py
    ├── requirements.txt            <- MUST be next to app.py
    ├── runtime.txt                 <- MUST be next to app.py
    ├── style.css
    ├── README.md
    └── DEPLOYMENT.md
```

## 🩹 Fixing `ModuleNotFoundError: ... sklearn.metrics ...`

This error means the deployed environment never installed `scikit-learn` (or
another package) — it is **not** a bug in `app.py`. Root causes, in order of
likelihood:

1. **`requirements.txt` isn't where Streamlit Cloud expects it.**
   In your app's **Settings → General**, check **"Main file path."** If it's
   set to `decisiontree/app.py`, Streamlit looks for `decisiontree/requirements.txt`.
   Fix: move/copy `requirements.txt` (and `runtime.txt`) into that same folder.

2. **Stale build cache after fixing #1.**
   Go to **Manage app (bottom-right) → ⋮ menu → Reboot app**. Streamlit Cloud
   caches the environment; a reboot forces a clean `pip install`.

3. **A typo or bad pin in `requirements.txt`.**
   Check **Manage app → Logs** — the real `pip install` error is printed there
   even though the in-app traceback is redacted.

4. **Version mismatch between training and deployment scikit-learn.**
   `model/decision_tree.pkl` was pickled with `scikit-learn==1.8.0`. This
   repo's `requirements.txt` pins that exact version to avoid unpickling
   errors. If you retrain locally with a different scikit-learn version,
   update the pin to match.

`app.py` in this repo already **detects missing packages at import time** and
shows a clear in-app message (instead of a raw traceback) pointing back to
this checklist — so you'll know immediately if #1–#3 recur.

## 📝 Step-by-Step: Deploy from Scratch

1. Push this project folder to a GitHub repository (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **"New app"** → select your repo/branch.
4. Set **"Main file path"** to the path of `app.py` (e.g. `app.py` if the
   project is at the repo root, or `decisiontree/app.py` if nested).
5. Click **Deploy**. Streamlit installs `requirements.txt` from the *same
   directory* as the main file automatically.
6. If the first build fails, check **Manage app → Logs**, fix the issue, then
   **Reboot app** (a plain "rerun" does not reinstall dependencies).

## 🔁 Retraining Before Deploy (optional)

Model artifacts are already committed under `/model`, so the app works
out-of-the-box without retraining. To regenerate them (e.g. after changing
features):

```bash
pip install -r requirements.txt
python3 train_pipeline.py
```

This overwrites `/model/*.pkl`, `/model/model_metadata.json`, and the PNGs in
`/assets`.

## 🖥️ Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

App should be reachable at `http://localhost:8501`.

## 🧪 Health Check

Streamlit Cloud (and most platforms) can probe `/_stcore/health` on the
deployed app to confirm it's live — useful if you wire this into external
uptime monitoring.
