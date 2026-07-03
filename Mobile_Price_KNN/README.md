# 📱 Mobile Price Category Prediction Using K-Nearest Neighbors (KNN)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-KNN-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Deployment--Ready-brightgreen)

An end-to-end, production-quality machine learning project that predicts a mobile phone's
**price category** (Low / Medium / High / Very High Cost) from its hardware specifications,
using a **K-Nearest Neighbors (KNN)** classifier — complete with a full Jupyter Notebook
analysis and a polished, deployable **Streamlit** web application.

---

## 📌 Project Overview

Manufacturers and retailers need a fast, explainable way to estimate what price tier a phone
belongs to, based purely on its specifications. This project builds that estimator using KNN
— a simple, interpretable, distance-based classification algorithm — and wraps it in an
interactive dashboard.

## 🧩 Problem Statement

Given 20 hardware specifications of a mobile phone (battery power, RAM, camera resolution,
screen size, connectivity features, etc.), predict which of four price categories the phone
belongs to:

| Class | Category |
|:-----:|----------|
| 0 | Low Cost |
| 1 | Medium Cost |
| 2 | High Cost |
| 3 | Very High Cost |

## 📊 Dataset

**Source:** [Mobile Price Classification — Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

- 2,000 records, 20 numerical features, 1 target column (`price_range`)
- No missing values, no duplicate rows, perfectly balanced classes (500 samples per class)

```python
import kagglehub
path = kagglehub.dataset_download("iabhishekofficial/mobile-price-classification")
```

## ✨ Features

- 📓 **Extremely detailed Jupyter Notebook** — EDA, cleaning, feature scaling, hyperparameter
  tuning, evaluation, and manual predictions, all with markdown explanations.
- 🔮 **Interactive Streamlit app** — sliders/inputs for every feature, instant predictions with
  confidence scores and probability breakdowns.
- 📊 **Model Performance dashboard** — accuracy, precision, recall, F1, confusion matrix,
  classification report, and the K-vs-Accuracy tuning curve.
- 🎨 **Premium dark UI** — glassmorphism cards, gradient hero, custom CSS, smooth hover
  animations.
- 💾 **Downloadable predictions**, form reset, and cached model loading for performance.

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn (KNeighborsClassifier, StandardScaler) |
| Model Persistence | Joblib |
| Web App | Streamlit, HTML/CSS |
| Notebook | Jupyter |

## 📁 Folder Structure

```
Mobile_Price_KNN/
│
├── data/
│   └── mobile_price.csv          # Training dataset
│
├── notebook/
│   └── Mobile_Price_KNN.ipynb    # Full EDA + training notebook (executed, with outputs)
│
├── model/
│   ├── knn_model.pkl             # Trained KNN classifier
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── feature_columns.pkl       # Ordered feature list expected by the model
│   └── eval_summary.json         # Saved metrics consumed by the Streamlit app
│
├── images/
│   ├── confusion_matrix.png      # Generated confusion matrix
│   └── k_vs_accuracy.png         # Generated K-tuning curve
│
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore
└── LICENSE
```

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Mobile_Price_KNN.git
cd Mobile_Price_KNN

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## ▶️ How to Run

### Run the Streamlit App
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`.

### Explore / Retrain via the Notebook
```bash
jupyter notebook notebook/Mobile_Price_KNN.ipynb
```
Running the notebook end-to-end regenerates `model/knn_model.pkl`, `model/scaler.pkl`, and the
evaluation images used by the app.

## 🖼️ Screenshots

> _Add screenshots of your running app here before publishing to GitHub._

| Home | Prediction | Model Performance |
|------|------------|--------------------|
| `images/screenshot_home.png` | `images/screenshot_prediction.png` | `images/screenshot_performance.png` |

## 📈 Model Performance

The model was tuned over K = {1, 3, 5, 7, 9, 11, 15}. Final results on the held-out 20% test set:

| Metric | Score |
|--------|-------|
| Best K | 15 |
| Accuracy | ~57.3% |
| Precision (weighted) | ~58.3% |
| Recall (weighted) | ~57.3% |
| F1 Score (weighted) | ~57.6% |

> Exact numbers are regenerated every time the notebook is re-run and saved to
> `model/eval_summary.json`, which the Streamlit **Model Performance** page reads live.
> Most misclassifications occur between **adjacent** price tiers (e.g., Medium vs. High),
> which is expected behavior for a continuous price spectrum discretized into four bins.

## 🚀 Future Improvements

- Feature selection / PCA to reduce noise from weakly-correlated features
- Distance-weighted KNN (`weights="distance"`)
- Cross-validated search over distance metrics (Manhattan, Minkowski)
- Training on real-world market pricing data instead of a synthetic dataset

## 👤 Author

**BSCS Final Year Project**
Built as a portfolio-ready, deployment-ready machine learning application.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
