# ❤️ Heart Disease Prediction Using Logistic Regression

A complete, end-to-end Machine Learning **classification** project that predicts a patient's **10-year risk of Coronary Heart Disease (CHD)** using **Logistic Regression**, built on the **Framingham Heart Study** dataset.

---

## 📌 Project Overview

Cardiovascular disease is one of the leading causes of death worldwide. This project builds a data-driven clinical decision-support tool that estimates a patient's likelihood of developing coronary heart disease within the next 10 years, based on demographic, behavioral, and clinical risk factors.

The project is intentionally kept to **3 core files** for a clean, professional, beginner-friendly GitHub portfolio piece:

| File | Purpose |
|---|---|
| `Heart_Disease_Prediction.ipynb` | Full ML workflow: cleaning → EDA → modeling → evaluation |
| `app.py` | Interactive Streamlit web app for live predictions |
| `README.md` | Project documentation (this file) |

---

## ❓ Problem Statement

> Given a patient's demographic information (age, sex, education), behavioral habits (smoking), and clinical measurements (blood pressure, cholesterol, glucose, BMI, heart rate), predict whether the patient is at risk of developing coronary heart disease within the next 10 years.

This is a **binary classification problem**:
- `0` → No 10-year CHD risk
- `1` → 10-year CHD risk present

---

## 📊 Dataset Information

- **Source:** [Framingham Heart Study — Kaggle](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)
- **Records:** 4,238 patients
- **Features:** 15 input features + 1 target (`TenYearCHD`)

| Feature | Description |
|---|---|
| `male` | Sex (1 = Male, 0 = Female) |
| `age` | Age in years |
| `education` | Education level (1–4) |
| `currentSmoker` | Current smoking status |
| `cigsPerDay` | Cigarettes smoked per day |
| `BPMeds` | On blood pressure medication |
| `prevalentStroke` | History of stroke |
| `prevalentHyp` | Hypertensive |
| `diabetes` | Diabetic status |
| `totChol` | Total cholesterol (mg/dL) |
| `sysBP` | Systolic blood pressure |
| `diaBP` | Diastolic blood pressure |
| `BMI` | Body Mass Index |
| `heartRate` | Heart rate (bpm) |
| `glucose` | Glucose level (mg/dL) |
| `TenYearCHD` | **Target** — 10-year CHD risk (0/1) |

To download the dataset programmatically:
```python
import kagglehub
path = kagglehub.dataset_download("dileep070/heart-disease-prediction-using-logistic-regression")
print(path)
```

---

## 🛠 Technologies Used

- Python
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

---

## 📁 Project Structure

```
Heart_Disease_Prediction/
│
├── Heart_Disease_Prediction.ipynb     # Complete ML pipeline
├── app.py                              # Streamlit user interface
├── README.md                           # Project documentation
│
├── logistic_model.pkl                  # Trained Logistic Regression model
├── scaler.pkl                          # Fitted StandardScaler
├── requirements.txt                    # Project dependencies
└── .gitignore
```

---

## 🧹 Data Cleaning

- Checked and removed duplicate rows.
- Handled missing values (`education`, `cigsPerDay`, `BPMeds`, `totChol`, `BMI`, `heartRate`, `glucose`) using **median imputation**, preserving robustness against skewed clinical distributions.
- Verified data types and consistency after cleaning.

---

## 🔍 Exploratory Data Analysis

The notebook includes:
- **Univariate analysis** — histograms & boxplots for `age`, `BMI`, `totChol`, `glucose`, `heartRate`, `sysBP`.
- **Target analysis** — class distribution shows a clear imbalance (~85% No CHD vs. ~15% CHD).
- **Bivariate analysis** — Age, smoking, diabetes, BMI, cholesterol, and blood pressure plotted against CHD outcome.
- **Correlation analysis** — Full correlation heatmap; `age`, `sysBP`, `prevalentHyp`, and `diaBP` show the strongest positive correlation with CHD risk.

---

## 🧪 Outlier Detection & Handling

Outliers in `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, and `glucose` were identified via boxplots and **capped at the 1st/99th percentile**, preserving genuine high-risk clinical cases while limiting the influence of extreme/erroneous values.

---

## ⚙️ Feature Engineering

- Built the feature matrix `X` (15 input features) and target vector `y` (`TenYearCHD`).
- Added optional exploratory features `age_group` and `bmi_category` for deeper EDA insight (not used in final modeling, since raw numeric values already capture this signal for Logistic Regression).

---

## 📏 Feature Scaling

All numeric features were standardized using **`StandardScaler`** (zero mean, unit variance). The scaler was fit **only on the training set** to prevent data leakage, then applied to both train and test sets, and finally saved as `scaler.pkl`.

---

## 🤖 Model Training

- **Algorithm:** Logistic Regression (`class_weight='balanced'` to address class imbalance)
- **Split:** 80% train / 20% test, stratified on the target, `random_state=42`

---

## 📈 Model Evaluation

| Metric | Score |
|---|---|
| Accuracy | **0.672** |
| Precision (CHD class) | **0.253** |
| Recall (CHD class) | **0.589** |
| F1 Score (CHD class) | **0.354** |
| ROC-AUC | **0.699** |

> Because the dataset is imbalanced (~85% negative class), accuracy alone is not a reliable measure of performance. Using `class_weight='balanced'`, the model trades some precision for substantially higher **recall**, meaning it correctly flags a much larger share of genuinely at-risk patients — an appropriate trade-off for a medical screening context, where missing a true positive is costlier than a false alarm.

**Generated visuals (see `/screenshots` once exported from the notebook):**
- Confusion Matrix
- ROC Curve (AUC = 0.699)
- Feature Coefficient Plot
- Correlation Heatmap

---

## 🧠 Results & Model Interpretation

Logistic Regression coefficients (on standardized features) reveal the key clinical risk drivers:

- **Age** is the strongest positive predictor — risk increases steadily with age.
- **Systolic blood pressure**, **hypertension**, and **history of stroke** are strong positive contributors.
- **Cigarettes per day** and **diabetes** also increase predicted risk.
- **Sex (male)** is moderately associated with higher CHD risk, consistent with established cardiovascular epidemiology.

These findings align closely with real-world clinical understanding of cardiovascular risk factors, lending interpretability and trust to the model.

---

## 🌐 Streamlit Application

`app.py` provides an interactive UI where a user enters a patient's clinical details (age, sex, smoking habits, blood pressure, cholesterol, BMI, glucose, etc.) and receives:

- **Prediction** — Low-Risk or High-Risk classification
- **Risk Probability** — predicted CHD probability as a percentage
- **Visual risk indicator** with contextual guidance

### Run the app locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** `app.py` loads `logistic_model.pkl` and `scaler.pkl`, which are produced by running `Heart_Disease_Prediction.ipynb`. Run the notebook first if these files are not present.

---

## 📸 Screenshots

After running the notebook, the following visuals are generated and saved as PNG files in the project directory:

- `target_distribution.png` — Class distribution
- `univariate_histograms.png` — Feature distributions
- `boxplots.png` — Outlier detection
- `bivariate_analysis.png` — Feature vs. target relationships
- `correlation_heatmap.png` — Correlation matrix
- `confusion_matrix.png` — Model confusion matrix
- `roc_curve.png` — ROC curve
- `feature_coefficients.png` — Feature importance

*(Add Streamlit app screenshots here once deployed/run locally.)*

---

## 🚀 Future Improvements

- Experiment with additional algorithms (Random Forest, XGBoost, SVM) and compare performance.
- Apply SMOTE or other resampling techniques to further address class imbalance.
- Add hyperparameter tuning (GridSearchCV / Optuna) for the Logistic Regression model.
- Deploy the Streamlit app to Streamlit Community Cloud for public access.
- Add SHAP-based explainability for individual predictions.
- Incorporate cross-validation for more robust performance estimates.

---

## ⚕️ Disclaimer

This project is built for **educational and portfolio purposes only**. It is **not** a certified medical diagnostic tool and should not be used as a substitute for professional medical advice.

---

## 📬 Author

Built as a complete, production-style Machine Learning Classification portfolio project using the Framingham Heart Study dataset.
