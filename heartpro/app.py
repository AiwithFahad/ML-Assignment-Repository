"""
Heart Disease Prediction — Streamlit App
------------------------------------------
Interactive UI for predicting 10-year risk of Coronary Heart Disease (CHD)
using a trained Logistic Regression model (Framingham Heart Study dataset).

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# ----------------------------------------------------------------------------
# Load model & scaler
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run `Heart_Disease_Prediction.ipynb` "
        "first to generate `logistic_model.pkl` and `scaler.pkl`."
    )
    st.stop()

FEATURE_ORDER = [
    "male", "age", "education", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
]

DEFAULTS = {
    "male": 1, "age": 50, "education": 2, "currentSmoker": 0, "cigsPerDay": 0,
    "BPMeds": 0, "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0,
    "totChol": 200, "sysBP": 120, "diaBP": 80, "BMI": 25.0, "heartRate": 75,
    "glucose": 90
}

if "reset" not in st.session_state:
    st.session_state.reset = False

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("❤️ Heart Disease Risk Predictor")
st.caption("10-Year Coronary Heart Disease (CHD) risk estimation — Framingham Heart Study model")
st.write(
    "Enter the patient's clinical and lifestyle information below, then click "
    "**Predict** to estimate their 10-year risk of developing coronary heart disease."
)
st.divider()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100,
                           value=DEFAULTS["age"], step=1)
    sex = st.selectbox("Sex", options=["Male", "Female"], index=0)
    education = st.selectbox(
        "Education Level",
        options=[1, 2, 3, 4],
        index=1,
        format_func=lambda x: {1: "Some High School", 2: "High School Graduate",
                                3: "Some College", 4: "College Graduate"}[x]
    )
    current_smoker = st.selectbox("Current Smoker?", options=["No", "Yes"], index=0)
    cigs_per_day = st.number_input("Cigarettes Per Day", min_value=0, max_value=80,
                                    value=DEFAULTS["cigsPerDay"], step=1)
    bp_meds = st.selectbox("On Blood Pressure Medication?", options=["No", "Yes"], index=0)
    prevalent_stroke = st.selectbox("History of Stroke?", options=["No", "Yes"], index=0)
    prevalent_hyp = st.selectbox("Hypertension?", options=["No", "Yes"], index=0)

with col2:
    diabetes = st.selectbox("Diabetes?", options=["No", "Yes"], index=0)
    tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600,
                                value=DEFAULTS["totChol"], step=1)
    sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=300,
                              value=DEFAULTS["sysBP"], step=1)
    dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=200,
                              value=DEFAULTS["diaBP"], step=1)
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0,
                           value=float(DEFAULTS["BMI"]), step=0.1)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200,
                                  value=DEFAULTS["heartRate"], step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400,
                               value=DEFAULTS["glucose"], step=1)

st.divider()

# ----------------------------------------------------------------------------
# Buttons
# ----------------------------------------------------------------------------
btn_col1, btn_col2 = st.columns(2)
predict_clicked = btn_col1.button("🔍 Predict", use_container_width=True, type="primary")
clear_clicked = btn_col2.button("🧹 Clear Inputs", use_container_width=True)

if clear_clicked:
    st.rerun()

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def encode_yes_no(val):
    return 1 if val == "Yes" else 0

if predict_clicked:
    input_dict = {
        "male": 1 if sex == "Male" else 0,
        "age": age,
        "education": education,
        "currentSmoker": encode_yes_no(current_smoker),
        "cigsPerDay": cigs_per_day,
        "BPMeds": encode_yes_no(bp_meds),
        "prevalentStroke": encode_yes_no(prevalent_stroke),
        "prevalentHyp": encode_yes_no(prevalent_hyp),
        "diabetes": encode_yes_no(diabetes),
        "totChol": tot_chol,
        "sysBP": sys_bp,
        "diaBP": dia_bp,
        "BMI": bmi,
        "heartRate": heart_rate,
        "glucose": glucose,
    }

    input_array = np.array([[input_dict[col] for col in FEATURE_ORDER]])
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    risk_pct = probability * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High-Risk Patient** — Heart Disease Risk Detected")
    else:
        st.success(f"✅ **Low-Risk Patient** — No Significant Heart Disease Risk Detected")

    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Predicted Class", "CHD Risk (1)" if prediction == 1 else "No CHD Risk (0)")
    res_col2.metric("Risk Probability", f"{risk_pct:.1f}%")

    st.progress(min(int(risk_pct), 100))

    if risk_pct >= 50:
        st.warning(
            "This patient's estimated 10-year CHD risk is elevated. "
            "Consider consulting a cardiologist for further clinical evaluation."
        )
    else:
        st.info(
            "This patient's estimated 10-year CHD risk is relatively low. "
            "Maintaining a healthy lifestyle is still recommended."
        )

st.divider()
st.caption(
    "⚕️ Disclaimer: This tool is for educational and portfolio purposes only and is "
    "**not** a substitute for professional medical advice, diagnosis, or treatment."
)
