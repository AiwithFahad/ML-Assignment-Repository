"""
Mobile Price Category Prediction - Streamlit Application
===========================================================
A production-quality Streamlit app that predicts a mobile phone's price
category (Low / Medium / High / Very High Cost) using a trained
K-Nearest Neighbors (KNN) model.

Run with:  streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Mobile Price Category Predictor | KNN",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "knn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")
EVAL_PATH = os.path.join(BASE_DIR, "model", "eval_summary.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "mobile_price.csv")
CM_IMAGE_PATH = os.path.join(BASE_DIR, "images", "confusion_matrix.png")
K_IMAGE_PATH = os.path.join(BASE_DIR, "images", "k_vs_accuracy.png")

PRICE_LABELS = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
PRICE_COLORS = {0: "#2DD4BF", 1: "#7C5CFC", 2: "#FBBF24", 3: "#FB6584"}
PRICE_ICONS = {0: "💚", 1: "💜", 2: "💛", 3: "❤️"}

# ----------------------------------------------------------------------
# CUSTOM CSS — "Circuit Board" dark tech theme
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-deep: #0B0F19;
    --bg-surface: #131826;
    --bg-card: #161C2C;
    --border-glass: rgba(124, 92, 252, 0.18);
    --accent-violet: #7C5CFC;
    --accent-teal: #2DD4BF;
    --accent-amber: #FBBF24;
    --accent-rose: #FB6584;
    --text-primary: #E8EAF3;
    --text-muted: #8B93AB;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(124,92,252,0.10) 0%, transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(45,212,191,0.08) 0%, transparent 40%),
        var(--bg-deep);
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}

/* ---------- Hero ---------- */
.hero-wrap {
    position: relative;
    border-radius: 22px;
    padding: 3.2rem 3rem;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg, #171227 0%, #1B1032 45%, #0E1E2E 100%);
    border: 1px solid var(--border-glass);
    overflow: hidden;
}
.hero-wrap::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(var(--border-glass) 1px, transparent 1px),
        linear-gradient(90deg, var(--border-glass) 1px, transparent 1px);
    background-size: 42px 42px;
    opacity: 0.35;
    mask-image: radial-gradient(ellipse at 70% 30%, black 10%, transparent 70%);
}
.hero-eyebrow {
    position: relative;
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    color: var(--accent-teal);
    background: rgba(45, 212, 191, 0.08);
    border: 1px solid rgba(45, 212, 191, 0.3);
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 1.1rem;
}
.hero-title {
    position: relative;
    font-size: 2.9rem;
    font-weight: 700;
    line-height: 1.08;
    margin: 0 0 0.7rem 0;
    background: linear-gradient(90deg, #F3F0FF 10%, #7C5CFC 55%, #2DD4BF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    position: relative;
    font-size: 1.08rem;
    color: var(--text-muted);
    max-width: 640px;
    line-height: 1.6;
}

/* ---------- Cards ---------- */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
}
.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(124, 92, 252, 0.55);
    box-shadow: 0 10px 30px rgba(124, 92, 252, 0.15);
}
.metric-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--accent-teal);
}
.metric-label {
    color: var(--text-muted);
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* ---------- Prediction result card ---------- */
.result-card {
    border-radius: 20px;
    padding: 2.2rem 2rem;
    text-align: center;
    border: 1px solid var(--border-glass);
    background: linear-gradient(160deg, var(--bg-card) 0%, #1A1030 100%);
    animation: fadeInUp 0.5s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.5rem;
    padding: 0.5rem 1.4rem;
    border-radius: 999px;
    margin: 0.6rem 0 1rem 0;
}

/* ---------- Section divider ---------- */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1.6rem 0 0.8rem 0;
    padding-left: 0.7rem;
    border-left: 4px solid var(--accent-violet);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--bg-surface);
    border-right: 1px solid var(--border-glass);
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Grotesk', sans-serif;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, var(--accent-violet), #5A3FE0);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    box-shadow: 0 6px 20px rgba(124, 92, 252, 0.4);
    transform: translateY(-2px);
}

/* Footer */
.footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid var(--border-glass);
    margin-top: 2.5rem;
}
hr { border-color: var(--border-glass) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained KNN model...")
def load_model():
    """Load the trained KNN model from disk."""
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading feature scaler...")
def load_scaler():
    """Load the fitted StandardScaler from disk."""
    return joblib.load(SCALER_PATH)


@st.cache_resource(show_spinner=False)
def load_feature_columns():
    """Load the ordered list of feature columns the model expects."""
    return joblib.load(FEATURES_PATH)


@st.cache_data(show_spinner=False)
def load_eval_summary():
    """Load the saved evaluation metrics/summary JSON."""
    with open(EVAL_PATH, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    """Load the raw training dataset used for EDA / dataset page."""
    return pd.read_csv(DATA_PATH)


def artifacts_available():
    """Check whether all required model artifacts exist on disk."""
    return all(
        os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH, EVAL_PATH]
    )


# ----------------------------------------------------------------------
# PREDICTION HELPER
# ----------------------------------------------------------------------
def predict_price_category(input_dict: dict):
    """
    Predict the price category for a single phone specification.

    Parameters
    ----------
    input_dict : dict
        Dictionary of feature_name -> value covering all model features.

    Returns
    -------
    tuple(int, np.ndarray)
        Predicted class label and the class probability array.
    """
    model = load_model()
    scaler = load_scaler()
    feature_columns = load_feature_columns()

    input_df = pd.DataFrame([input_dict])[feature_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    return int(prediction), probabilities


# ----------------------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------------------
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_probabilities" not in st.session_state:
    st.session_state.last_probabilities = None
if "last_inputs" not in st.session_state:
    st.session_state.last_inputs = None


# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1.2rem 0;">
            <div style="font-size:2.2rem;">📶</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.15rem;">
                Mobile Price KNN
            </div>
            <div style="color:#8B93AB; font-size:0.8rem;">v1.0 &middot; KNN Classifier</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "🗂️ Dataset", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.8rem; color:#8B93AB; line-height:1.6;">
        Built with <b>Streamlit</b> &amp; <b>Scikit-learn</b><br>
        Algorithm: <b>K-Nearest Neighbors</b><br>
        Dataset: Mobile Price Classification
        </div>
        """,
        unsafe_allow_html=True,
    )

if not artifacts_available():
    st.error(
        "Model artifacts not found. Please run the training notebook "
        "(`notebook/Mobile_Price_KNN.ipynb`) first to generate "
        "`model/knn_model.pkl`, `model/scaler.pkl`, `model/feature_columns.pkl`, "
        "and `model/eval_summary.json`."
    )
    st.stop()

eval_summary = load_eval_summary()
feature_columns = load_feature_columns()

# ========================================================================
# PAGE: HOME
# ========================================================================
if page == "🏠 Home":
    st.markdown(
        f"""
        <div class="hero-wrap">
            <span class="hero-eyebrow">K-NEAREST NEIGHBORS &middot; MULTI-CLASS CLASSIFICATION</span>
            <h1 class="hero-title">Predict a Phone's Price Category<br>From Its Specs Alone</h1>
            <p class="hero-subtitle">
                A production-ready machine learning app that classifies a mobile phone into
                <b>Low</b>, <b>Medium</b>, <b>High</b>, or <b>Very High</b> cost tiers using
                20 hardware specifications and a tuned K-Nearest Neighbors model
                (K = {eval_summary['best_k']}).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Test Accuracy", f"{eval_summary['accuracy']*100:.1f}%"),
        ("Best K", str(eval_summary["best_k"])),
        ("Training Samples", f"{eval_summary['train_size']}"),
        ("Features Used", f"{len(feature_columns)}"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-num">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    steps = [
        ("📥", "1. Enter Specs", "Provide battery, RAM, camera, display and connectivity details for a phone."),
        ("📐", "2. Distance Voting", "The model scales inputs and finds the K most similar phones in the training data."),
        ("🏷️", "3. Get a Category", "The majority price category among those neighbors becomes the prediction."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], steps):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div style="font-size:1.8rem;">{icon}</div>
                    <div style="font-weight:600; margin:0.4rem 0;">{title}</div>
                    <div style="color:#8B93AB; font-size:0.9rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Price Categories</div>', unsafe_allow_html=True)
    cats = st.columns(4)
    for col, cls in zip(cats, [0, 1, 2, 3]):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center; border-color:{PRICE_COLORS[cls]}55;">
                    <div style="font-size:1.6rem;">{PRICE_ICONS[cls]}</div>
                    <div style="font-weight:700; color:{PRICE_COLORS[cls]}; margin-top:0.3rem;">
                        {PRICE_LABELS[cls]}
                    </div>
                    <div style="color:#8B93AB; font-size:0.8rem;">Class {cls}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Developer Information</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <b>Project:</b> Mobile Price Category Prediction Using KNN (BSCS Final Assignment)<br>
            <b>Tech Stack:</b> Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit<br>
            <b>Repository:</b> Deployment-ready for GitHub &amp; Streamlit Cloud
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========================================================================
# PAGE: PREDICTION
# ========================================================================
elif page == "🔮 Prediction":
    st.markdown('<div class="section-title">📥 Enter Phone Specifications</div>', unsafe_allow_html=True)
    st.caption("Adjust the sliders and inputs below to describe the phone, then click Predict.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🔋 Power & Performance**")
            battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1200, step=10)
            clock_speed = st.number_input("Processor Clock Speed (GHz)", 0.5, 3.0, 1.5, step=0.1)
            n_cores = st.slider("Number of Processor Cores", 1, 8, 4)
            int_memory = st.slider("Internal Memory (GB)", 2, 64, 32)
            ram = st.slider("RAM (MB)", 256, 4000, 2000, step=10)
            m_dep = st.number_input("Mobile Depth / Thickness (cm)", 0.1, 1.0, 0.5, step=0.1)
            mobile_wt = st.slider("Mobile Weight (grams)", 80, 200, 140)

        with col2:
            st.markdown("**📷 Camera & Display**")
            fc = st.slider("Front Camera (MP)", 0, 20, 5)
            pc = st.slider("Primary Camera (MP)", 0, 20, 10)
            px_height = st.slider("Pixel Resolution Height", 0, 1960, 800)
            px_width = st.slider("Pixel Resolution Width", 500, 2000, 1200)
            sc_h = st.slider("Screen Height (cm)", 5, 19, 12)
            sc_w = st.slider("Screen Width (cm)", 0, 18, 7)
            talk_time = st.slider("Talk Time (hours)", 2, 20, 10)

        with col3:
            st.markdown("**📶 Connectivity Features**")
            blue = st.selectbox("Bluetooth", ["Yes", "No"], index=0)
            dual_sim = st.selectbox("Dual SIM Support", ["Yes", "No"], index=0)
            four_g = st.selectbox("4G Support", ["Yes", "No"], index=0)
            three_g = st.selectbox("3G Support", ["Yes", "No"], index=0)
            touch_screen = st.selectbox("Touch Screen", ["Yes", "No"], index=0)
            wifi = st.selectbox("WiFi", ["Yes", "No"], index=0)

        col_predict, col_reset = st.columns([3, 1])
        with col_predict:
            submitted = st.form_submit_button("🔮 Predict Price Category", use_container_width=True)
        with col_reset:
            reset_clicked = st.form_submit_button("♻️ Reset Form", use_container_width=True)

    def yn(value):
        return 1 if value == "Yes" else 0

    if reset_clicked:
        st.session_state.last_prediction = None
        st.session_state.last_probabilities = None
        st.session_state.last_inputs = None
        st.rerun()

    if submitted:
        input_dict = {
            "battery_power": battery_power,
            "blue": yn(blue),
            "clock_speed": clock_speed,
            "dual_sim": yn(dual_sim),
            "fc": fc,
            "four_g": yn(four_g),
            "int_memory": int_memory,
            "m_dep": m_dep,
            "mobile_wt": mobile_wt,
            "n_cores": n_cores,
            "pc": pc,
            "px_height": px_height,
            "px_width": px_width,
            "ram": ram,
            "sc_h": sc_h,
            "sc_w": sc_w,
            "talk_time": talk_time,
            "three_g": yn(three_g),
            "touch_screen": yn(touch_screen),
            "wifi": yn(wifi),
        }
        pred_class, probabilities = predict_price_category(input_dict)
        st.session_state.last_prediction = pred_class
        st.session_state.last_probabilities = probabilities
        st.session_state.last_inputs = input_dict

    if st.session_state.last_prediction is not None:
        pred_class = st.session_state.last_prediction
        probabilities = st.session_state.last_probabilities
        confidence = probabilities[pred_class] * 100
        color = PRICE_COLORS[pred_class]
        icon = PRICE_ICONS[pred_class]
        label = PRICE_LABELS[pred_class]

        st.markdown('<div class="section-title">🏷️ Prediction Result</div>', unsafe_allow_html=True)
        res_col, chart_col = st.columns([1, 1.2])

        with res_col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size:2.4rem;">{icon}</div>
                    <div style="color:#8B93AB; font-size:0.9rem; margin-top:0.3rem;">PREDICTED PRICE CATEGORY</div>
                    <div class="badge" style="background:{color}22; color:{color}; border:1px solid {color}66;">
                        {label}
                    </div>
                    <div style="color:#8B93AB;">Confidence: <b style="color:{color};">{confidence:.1f}%</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success(f"Prediction complete — classified as **{label}** with {confidence:.1f}% confidence.")

            result_df = pd.DataFrame(
                {
                    "Feature": list(st.session_state.last_inputs.keys()),
                    "Value": list(st.session_state.last_inputs.values()),
                }
            )
            result_df.loc[len(result_df)] = ["predicted_price_category", label]
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Prediction Result (CSV)",
                data=csv_bytes,
                file_name="mobile_price_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with chart_col:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[PRICE_LABELS[i] for i in range(4)],
                        y=[p * 100 for p in probabilities],
                        marker_color=[PRICE_COLORS[i] for i in range(4)],
                        text=[f"{p*100:.1f}%" for p in probabilities],
                        textposition="outside",
                    )
                ]
            )
            fig.update_layout(
                title="Class Probability Breakdown",
                yaxis_title="Probability (%)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#E8EAF3"),
                yaxis=dict(range=[0, 100]),
                height=380,
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# PAGE: MODEL PERFORMANCE
# ========================================================================
elif page == "📊 Model Performance":
    st.markdown('<div class="section-title">📊 Model Performance Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    perf_metrics = [
        ("Accuracy", f"{eval_summary['accuracy']*100:.2f}%"),
        ("Precision (weighted)", f"{eval_summary['precision']*100:.2f}%"),
        ("Recall (weighted)", f"{eval_summary['recall']*100:.2f}%"),
        ("F1 Score (weighted)", f"{eval_summary['f1_score']*100:.2f}%"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], perf_metrics):
        with col:
            st.markdown(
                f"""<div class="glass-card"><div class="metric-num">{value}</div>
                <div class="metric-label">{label}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="glass-card">
        <b>Best K (Hyperparameter):</b> {eval_summary['best_k']}<br>
        <b>Train / Test Split:</b> {eval_summary['train_size']} train / {eval_summary['test_size']} test (80/20)<br>
        <b>Distance Metric:</b> Euclidean<br>
        <b>Dataset Shape:</b> {eval_summary['dataset_shape'][0]} rows &times; {eval_summary['dataset_shape'][1]} columns
        </div>
        """, unsafe_allow_html=True)
    with c2:
        k_df = pd.DataFrame({
            "K": eval_summary["k_values"],
            "Accuracy": [round(a, 4) for a in eval_summary["k_accuracies"]],
        })
        fig_k = go.Figure(
            data=go.Scatter(
                x=k_df["K"], y=k_df["Accuracy"] * 100, mode="lines+markers",
                line=dict(color="#7C5CFC", width=3), marker=dict(size=9, color="#2DD4BF"),
            )
        )
        fig_k.update_layout(
            title="K vs Accuracy", xaxis_title="K", yaxis_title="Accuracy (%)",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#E8EAF3"), height=280, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_k, use_container_width=True)

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    if os.path.exists(CM_IMAGE_PATH):
        st.image(CM_IMAGE_PATH, use_container_width=False, width=520)
    else:
        cm = np.array(eval_summary["confusion_matrix"])
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Low", "Medium", "High", "Very High"], y=["Low", "Medium", "High", "Very High"],
            colorscale="Purples", text=cm, texttemplate="%{text}",
        ))
        fig_cm.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(eval_summary["classification_report"]).T
    report_df = report_df.round(3)
    st.dataframe(report_df, use_container_width=True)

    st.markdown('<div class="section-title">Evaluation Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="glass-card">
        The final model uses <b>K = {eval_summary['best_k']}</b> neighbors with Euclidean distance,
        chosen via a hyperparameter sweep over K &isin; {eval_summary['k_values']}.
        It was trained on {eval_summary['train_size']} samples and evaluated on
        {eval_summary['test_size']} held-out samples, achieving
        <b>{eval_summary['accuracy']*100:.2f}% accuracy</b>. Most misclassifications occur between
        adjacent price tiers (e.g., Medium vs. High), which is expected for a continuous
        underlying price spectrum discretized into four bins.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ========================================================================
# PAGE: DATASET
# ========================================================================
elif page == "🗂️ Dataset":
    st.markdown('<div class="section-title">🗂️ Dataset Overview</div>', unsafe_allow_html=True)
    df = load_dataset()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="glass-card"><div class="metric-num">{df.shape[0]}</div>
        <div class="metric-label">Rows</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="glass-card"><div class="metric-num">{df.shape[1]}</div>
        <div class="metric-label">Columns</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="glass-card"><div class="metric-num">{int(df.isnull().sum().sum())}</div>
        <div class="metric-label">Missing Values</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().T.round(2), use_container_width=True)

    st.markdown('<div class="section-title">Feature Descriptions</div>', unsafe_allow_html=True)
    feature_desc = {
        "battery_power": "Total energy a battery can store, in mAh",
        "blue": "Has Bluetooth or not (1 = Yes, 0 = No)",
        "clock_speed": "Speed at which the microprocessor executes instructions (GHz)",
        "dual_sim": "Has dual SIM support or not",
        "fc": "Front camera megapixels",
        "four_g": "Has 4G or not",
        "int_memory": "Internal memory in gigabytes",
        "m_dep": "Mobile depth / thickness in cm",
        "mobile_wt": "Weight of the mobile phone in grams",
        "n_cores": "Number of processor cores",
        "pc": "Primary (rear) camera megapixels",
        "px_height": "Pixel resolution height",
        "px_width": "Pixel resolution width",
        "ram": "Random Access Memory in megabytes",
        "sc_h": "Screen height in cm",
        "sc_w": "Screen width in cm",
        "talk_time": "Longest time a single battery charge lasts on a call (hours)",
        "three_g": "Has 3G or not",
        "touch_screen": "Has touch screen or not",
        "wifi": "Has WiFi or not",
        "price_range": "Target: 0=Low, 1=Medium, 2=High, 3=Very High cost",
    }
    desc_df = pd.DataFrame(list(feature_desc.items()), columns=["Feature", "Description"])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

# ========================================================================
# PAGE: ABOUT
# ========================================================================
elif page == "ℹ️ About":
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
        <h4>Problem</h4>
        <p>Manufacturers and retailers need a fast, explainable way to estimate what price tier a
        phone belongs to based purely on its hardware specifications.</p>

        <h4>Dataset</h4>
        <p>The <b>Mobile Price Classification</b> dataset (Kaggle) contains 2,000 phone records with
        20 numerical hardware features and a 4-class target: Low, Medium, High, and Very High cost.</p>

        <h4>Algorithm</h4>
        <p><b>K-Nearest Neighbors (KNN)</b> — a distance-based, non-parametric, lazy-learning
        classifier. Predictions are made by majority vote among the K closest training samples,
        measured with Euclidean distance on standardized features.</p>

        <h4>Workflow</h4>
        <p>Data Cleaning &rarr; Exploratory Data Analysis &rarr; Feature Scaling
        &rarr; Train/Test Split (80/20) &rarr; Hyperparameter Tuning (K search)
        &rarr; Model Evaluation &rarr; Model Persistence (Joblib) &rarr; Streamlit Deployment.</p>

        <h4>Business Applications</h4>
        <p>E-commerce price-tier auto-tagging, competitive benchmarking for new devices, and a
        quick fair-price sanity check for consumers.</p>

        <h4>Limitations</h4>
        <p>KNN treats all 20 features equally in distance calculations even though several are weak
        predictors, and the model can be slower at prediction time on very large datasets. The
        dataset is synthetically curated rather than pulled from live market data.</p>

        <h4>Future Work</h4>
        <p>Feature selection / PCA, distance-weighted KNN, cross-validated metric selection, and
        training on real-world market pricing data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit &amp; Scikit-learn &nbsp;|&nbsp;
        Mobile Price Category Prediction &mdash; KNN Project &nbsp;|&nbsp;
        © 2026 BSCS Final Project
    </div>
    """,
    unsafe_allow_html=True,
)
