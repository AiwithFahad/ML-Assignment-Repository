"""
AI-Powered Real-Time Digital Payment Fraud Detection System
=============================================================
Streamlit UI for a Decision-Tree-only fraud detection model trained on the
PaySim dataset.

Run:  streamlit run app.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard AI | Decision Tree Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent


def load_css():
    css_path = ROOT / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


load_css()

# --------------------------------------------------------------------------
# LOAD ARTIFACTS (cached)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    with open(ROOT / "model" / "decision_tree.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ROOT / "model" / "label_encoder.pkl", "rb") as f:
        type_encoder = pickle.load(f)
    with open(ROOT / "model" / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open(ROOT / "model" / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, type_encoder, feature_columns, metadata


@st.cache_data
def load_sample_data():
    return pd.read_csv(ROOT / "data" / "paysim_sample_15000.csv")


try:
    model, type_encoder, FEATURE_COLUMNS, metadata = load_model_artifacts()
    ARTIFACTS_OK = True
except FileNotFoundError:
    ARTIFACTS_OK = False

# --------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------------------------------
st.sidebar.markdown(
    "<div class='sidebar-brand'>🛡️ <span>FraudGuard AI</span></div>",
    unsafe_allow_html=True,
)
st.sidebar.caption("Decision Tree Fraud Detection · PaySim")

PAGES = [
    "🏠 Home",
    "🔮 Fraud Prediction",
    "📊 Dataset Explorer",
    "🌳 Decision Tree Visualizer",
    "📈 Model Performance",
    "🎯 Feature Importance",
    "🧠 AI Insights",
    "ℹ️ About Project",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

if not ARTIFACTS_OK:
    st.error(
        "Model artifacts not found. Run `python3 train_pipeline.py` first to "
        "generate `/model/*.pkl` and `/model/model_metadata.json`."
    )
    st.stop()


def engineer_features(row: dict) -> pd.DataFrame:
    """Turn a raw transaction dict into the model's engineered feature row."""
    amount = row["amount"]
    oldbalanceOrg = row["oldbalanceOrg"]
    newbalanceOrig = row["newbalanceOrig"]
    oldbalanceDest = row["oldbalanceDest"]
    newbalanceDest = row["newbalanceDest"]

    features = {
        "step": row["step"],
        "type_encoded": type_encoder.transform([row["type"]])[0],
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balance_diff_orig": oldbalanceOrg - newbalanceOrig,
        "balance_diff_dest": newbalanceDest - oldbalanceDest,
        "amount_to_balance_ratio": amount / (oldbalanceOrg + 1),
        "orig_zero_balance_flag": int(newbalanceOrig == 0),
        "dest_zero_balance_flag": int(oldbalanceDest == 0),
        "suspicious_transaction_flag": int(
            oldbalanceOrg == amount and newbalanceOrig == 0
        ),
        "large_transaction_flag": int(amount > 200000),
        "orig_balance_error": oldbalanceOrg - amount - newbalanceOrig,
        "dest_balance_error": oldbalanceDest + amount - newbalanceDest,
        "isFlaggedFraud": int(amount > 200000 and row["type"] == "TRANSFER"),
    }
    return pd.DataFrame([features])[FEATURE_COLUMNS]


# ==========================================================================
# PAGE: HOME
# ==========================================================================
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">⚡ Real-Time · Explainable · Decision-Tree-Only</div>
            <h1>AI-Powered Digital Payment<br><span class="gradient-text">Fraud Detection</span></h1>
            <p>An enterprise-grade fraud screening system built entirely on a single,
            fully explainable Decision Tree Classifier — trained on the PaySim
            simulated mobile-money dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m = metadata["metrics"]
    kpis = [
        ("🎯 Accuracy", f"{m['Accuracy']*100:.1f}%"),
        ("🔍 Recall", f"{m['Recall']*100:.1f}%"),
        ("✅ Precision", f"{m['Precision']*100:.1f}%"),
        ("📈 ROC-AUC", f"{m['ROC-AUC']*100:.1f}%"),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, kpis):
        col.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### 🧭 How It Works")
    steps = st.columns(4)
    workflow = [
        ("1️⃣", "Transaction In", "Raw transaction data (type, amount, balances) enters the pipeline."),
        ("2️⃣", "Feature Engineering", "8 fraud-indicator features are derived in real time."),
        ("3️⃣", "Decision Tree Scoring", "A tuned Decision Tree Classifier scores fraud probability."),
        ("4️⃣", "Risk Decision", "Legitimate / Fraudulent verdict with confidence & explanation."),
    ]
    for col, (icon, title, desc) in zip(steps, workflow):
        col.markdown(
            f"""<div class="glass-card"><div class="wf-icon">{icon}</div>
            <div class="wf-title">{title}</div><p class="wf-desc">{desc}</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("### 💼 Business Impact")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        """<div class="glass-card"><h4>🏦 For Banks</h4>
        <p>Screen transfers and cash-outs in real time before funds leave the network.</p></div>""",
        unsafe_allow_html=True,
    )
    c2.markdown(
        """<div class="glass-card"><h4>📱 For Mobile Wallets</h4>
        <p>Flag account-draining patterns the moment a balance is zeroed out.</p></div>""",
        unsafe_allow_html=True,
    )
    c3.markdown(
        """<div class="glass-card"><h4>🛒 For E-Commerce</h4>
        <p>Reduce chargeback exposure with explainable, auditable fraud rules.</p></div>""",
        unsafe_allow_html=True,
    )

# ==========================================================================
# PAGE: FRAUD PREDICTION
# ==========================================================================
elif page == "🔮 Fraud Prediction":
    st.markdown("## 🔮 Real-Time Fraud Prediction")
    st.caption("Enter transaction details to get an instant fraud risk assessment.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            step = st.number_input("Time Step (hour)", min_value=1, max_value=744, value=10)
            tx_type = st.selectbox("Transaction Type", list(type_encoder.classes_), index=1)
            amount = st.number_input("Amount", min_value=0.0, value=5000.0, step=100.0)
        with c2:
            oldbalanceOrg = st.number_input("Sender Balance Before", min_value=0.0, value=10000.0, step=100.0)
            newbalanceOrig = st.number_input("Sender Balance After", min_value=0.0, value=5000.0, step=100.0)
        with c3:
            oldbalanceDest = st.number_input("Receiver Balance Before", min_value=0.0, value=0.0, step=100.0)
            newbalanceDest = st.number_input("Receiver Balance After", min_value=0.0, value=5000.0, step=100.0)

        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

    if submitted:
        row = dict(
            step=step, type=tx_type, amount=amount,
            oldbalanceOrg=oldbalanceOrg, newbalanceOrig=newbalanceOrig,
            oldbalanceDest=oldbalanceDest, newbalanceDest=newbalanceDest,
        )
        X_input = engineer_features(row)
        proba = model.predict_proba(X_input)[0][1]
        pred = int(proba >= 0.5)

        st.markdown("---")
        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if pred == 1:
                st.markdown(
                    f"""<div class="result-card fraud">
                    <div class="result-icon">🚨</div>
                    <div class="result-title">FRAUDULENT TRANSACTION</div>
                    <div class="result-conf">Confidence: {proba*100:.1f}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="result-card legit">
                    <div class="result-icon">✅</div>
                    <div class="result-title">LEGITIMATE TRANSACTION</div>
                    <div class="result-conf">Confidence: {(1-proba)*100:.1f}%</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            risk_level = "🔴 High" if proba > 0.7 else ("🟠 Medium" if proba > 0.3 else "🟢 Low")
            st.markdown(f"**Risk Level:** {risk_level}")

            reasons = []
            if row["oldbalanceOrg"] == row["amount"] and row["newbalanceOrig"] == 0:
                reasons.append("Sender account fully drained to zero (classic fraud signature).")
            if row["amount"] > 200000:
                reasons.append("Transaction amount exceeds the large-transaction threshold.")
            if tx_type in ("TRANSFER", "CASH_OUT"):
                reasons.append(f"Transaction type '{tx_type}' has the highest historical fraud rate.")
            if not reasons:
                reasons.append("No strong individual fraud signatures detected; verdict driven by combined feature pattern.")
            st.markdown("**Prediction Explanation:**")
            for r in reasons:
                st.markdown(f"- {r}")

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": "Fraud Risk Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc2626" if pred else "#059669"},
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, 70], "color": "#fef9c3"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                },
            ))
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download Prediction Report (CSV)",
            data=pd.DataFrame([{**row, "fraud_probability": proba, "prediction": "Fraud" if pred else "Legitimate"}]).to_csv(index=False),
            file_name="fraud_prediction_report.csv",
            mime="text/csv",
        )

# ==========================================================================
# PAGE: DATASET EXPLORER
# ==========================================================================
elif page == "📊 Dataset Explorer":
    st.markdown("## 📊 Dataset Explorer")
    df = load_sample_data()
    st.caption(f"Reproducible sample: {len(df):,} transactions · fraud ratio {df['isFraud'].mean()*100:.1f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraud Cases", f"{df['isFraud'].sum():,}")
    c3.metric("Legitimate Cases", f"{(df['isFraud']==0).sum():,}")
    c4.metric("Transaction Types", df["type"].nunique())

    tab1, tab2, tab3 = st.tabs(["Fraud Distribution", "Transaction Types", "Raw Data"])
    with tab1:
        fig = px.pie(df, names=df["isFraud"].map({0: "Legitimate", 1: "Fraud"}),
                     color_discrete_sequence=["#2563eb", "#dc2626"], hole=0.45)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        rate = df.groupby("type")["isFraud"].mean().reset_index()
        fig = px.bar(rate, x="type", y="isFraud", color="isFraud",
                     color_continuous_scale="Reds", labels={"isFraud": "Fraud Rate"})
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.dataframe(df.head(500), use_container_width=True, height=400)
        st.download_button("⬇️ Download Sample CSV", df.to_csv(index=False),
                            "paysim_sample_15000.csv", "text/csv")

# ==========================================================================
# PAGE: DECISION TREE VISUALIZER
# ==========================================================================
elif page == "🌳 Decision Tree Visualizer":
    st.markdown("## 🌳 Decision Tree Visualizer")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tree Depth", metadata["tree_depth"])
    c2.metric("Leaf Nodes", metadata["tree_leaves"])
    c3.metric("Features Used", len(FEATURE_COLUMNS))

    tree_img = ROOT / "assets" / "tree_visualization.png"
    if tree_img.exists():
        st.image(str(tree_img), caption="Decision Tree structure (first 3 levels)", use_container_width=True)

    st.markdown("### 📜 Extracted Decision Rules")
    rules_path = ROOT / "model" / "tree_rules.txt"
    if rules_path.exists():
        st.code(rules_path.read_text(), language="text")
    else:
        st.info("Run train_pipeline.py to generate tree_rules.txt")

    st.markdown("### 🎛️ Best Hyperparameters (from GridSearchCV)")
    st.json(metadata["best_params"])

# ==========================================================================
# PAGE: MODEL PERFORMANCE
# ==========================================================================
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Dashboard")
    m = metadata["metrics"]
    cols = st.columns(5)
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    for col, label in zip(cols, labels):
        col.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
            <div class="kpi-value">{m[label]*100:.1f}%</div></div>""",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        cm_img = ROOT / "assets" / "confusion_matrix.png"
        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion Matrix", use_container_width=True)
    with c2:
        roc_img = ROOT / "assets" / "roc_curve.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="ROC Curve", use_container_width=True)

    pr_img = ROOT / "assets" / "precision_recall_curve.png"
    if pr_img.exists():
        st.image(str(pr_img), caption="Precision-Recall Curve", use_container_width=True)

    st.markdown("### 🎛️ Best Hyperparameters")
    st.json(metadata["best_params"])

    with st.expander("💰 Financial Impact of Errors"):
        st.markdown(
            "- **False Positive** (legit flagged as fraud): customer friction, "
            "support cost — recoverable.\n"
            "- **False Negative** (fraud missed): direct monetary loss, "
            "irreversible — this is why the model is tuned to prioritize **Recall**."
        )

# ==========================================================================
# PAGE: FEATURE IMPORTANCE
# ==========================================================================
elif page == "🎯 Feature Importance":
    st.markdown("## 🎯 Feature Importance Analysis")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    fig = px.bar(
        x=importances.values, y=importances.index, orientation="h",
        color=importances.values, color_continuous_scale="Viridis",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔎 Interpretation")
    top3 = importances.head(3)
    for feat, val in top3.items():
        st.markdown(f"- **`{feat}`** (importance {val:.3f}): a top driver of the tree's fraud splits.")

# ==========================================================================
# PAGE: AI INSIGHTS
# ==========================================================================
elif page == "🧠 AI Insights":
    st.markdown("## 🧠 AI Insights & Fraud Rules")
    st.markdown(
        """
        The Decision Tree learned interpretable, auditable rules directly from
        the data. A few of the strongest patterns:
        """
    )
    st.markdown(
        """
        <div class="glass-card">
        <b>Rule 1 — Account Drain Signature</b><br>
        <code>IF suspicious_transaction_flag == 1 → Fraud</code><br>
        Sender's entire balance was withdrawn in one transaction — the single
        strongest fraud signal in the dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="glass-card">
        <b>Rule 2 — Destination Ledger Anomaly</b><br>
        <code>IF dest_balance_error is large → Fraud</code><br>
        The receiver's before/after balances don't reconcile with the transferred
        amount — a synthetic-data marker strongly associated with fraud.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="glass-card">
        <b>Rule 3 — High-Risk Channel</b><br>
        <code>IF type in [TRANSFER, CASH_OUT] AND amount is large → Elevated Risk</code><br>
        These channels move money out of the ecosystem and carry the highest
        historical fraud rates.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================================================
# PAGE: ABOUT
# ==========================================================================
elif page == "ℹ️ About Project":
    st.markdown("## ℹ️ About This Project")
    st.markdown(
        f"""
        **Project:** AI-Powered Real-Time Digital Payment Fraud Detection System
        **Algorithm:** Decision Tree Classifier (exclusively — no ensembles)
        **Dataset:** PaySim — Synthetic Financial Datasets for Fraud Detection (Kaggle)
        **Sample size used:** {metadata.get('sample_size', 15000):,} transactions
        **Model params:** `{metadata['best_params']}`

        ### 🎯 Why Decision Trees?
        Decision Trees are fully transparent — every prediction can be traced
        back to a human-readable sequence of `IF/THEN` rules, which matters for
        regulated financial use cases where model decisions must be explainable
        and auditable.

        ### 🏗️ Tech Stack
        Python · scikit-learn · pandas · Streamlit · Plotly · Matplotlib/Seaborn

        ### 👤 Ideal For
        BSCS Final Year Projects · University Competitions · Job Portfolios ·
        GitHub Showcases · FinTech Demos · ML Learning Resource
        """
    )

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
    Built with 🛡️ using a single, fully-explainable Decision Tree · PaySim Dataset · 2026
    </div>
    """,
    unsafe_allow_html=True,
)
