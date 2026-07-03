# 🛡️ FraudGuard AI — Digital Payment Fraud Detection with Decision Trees

**An enterprise-grade, fully-explainable fraud detection system built exclusively on a single Decision Tree Classifier.**

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red) ![License](https://img.shields.io/badge/License-MIT-green)

> Suitable for: BSCS final-year projects · university ML competitions · job portfolios · GitHub showcase · Streamlit Cloud deployment.

---

## 🎯 Project Overview

FraudGuard AI classifies digital payment transactions as **✅ Legitimate** or **🚨 Fraudulent** using the **PaySim** synthetic mobile-money dataset. The entire system — from EDA to the deployed dashboard — deliberately uses **only the Decision Tree Classifier**, no ensembles, to keep every prediction fully explainable via human-readable `IF/THEN` rules.

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 99.8% |
| Precision | 99.5% |
| Recall | 99.5% |
| F1 Score | 99.5% |
| ROC-AUC | 99.7% |

*Measured on a held-out 20% test split of the 15,000-row reproducible sample (see [Sampling Strategy](#-dataset--sampling-strategy)). See the notebook for full cross-validation results and a discussion of how these numbers would shift at the true 0.13% production fraud rate.*

## 🌳 Why Decision Trees Only?

Financial institutions need models whose decisions can be **audited and explained** to regulators and customers. A single Decision Tree produces a transparent sequence of splits — e.g. `IF suspicious_transaction_flag > 0.5 THEN Fraud` — that a compliance team can read and verify directly, unlike a black-box ensemble or neural network.

## 🏗️ Project Structure

```
Digital_Payment_Fraud_Detection_DecisionTree/
├── data/
│   └── paysim_sample_15000.csv     # Reproducible stratified sample
├── notebook/
│   └── Fraud_Detection_DecisionTree.ipynb   # Full case-study notebook (executed)
├── model/
│   ├── decision_tree.pkl           # Trained model
│   ├── label_encoder.pkl           # Transaction-type encoder
│   ├── feature_columns.pkl         # Ordered feature list
│   ├── model_metadata.json         # Hyperparameters + metrics
│   └── tree_rules.txt              # Extracted decision rules
├── assets/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── tree_visualization.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
├── train_pipeline.py                # End-to-end training script
├── app.py                           # Streamlit dashboard
├── style.css                        # Premium FinTech UI theme
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## 📂 Dataset & Sampling Strategy

**Source:** [PaySim — Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) (6,362,620 transactions, 8,213 fraud = 0.13%)

```python
import kagglehub
path = kagglehub.dataset_download("ealaxi/paysim1")
```

Because 0.13% fraud is too sparse to learn from in a small, fast-iterating sample, this project uses a **documented, reproducible oversample**:

- 3,000 fraudulent transactions (`random_state=42`)
- 12,000 legitimate transactions (`random_state=42`)
- → **15,000 rows, 20% fraud ratio**, shuffled and saved to `data/paysim_sample_15000.csv`

This trade-off is explained in full in the notebook, along with its implications for production deployment.

## ⚙️ Feature Engineering

| Feature | Purpose |
|---|---|
| `balance_diff_orig` / `balance_diff_dest` | Direct outflow/inflow measurement |
| `amount_to_balance_ratio` | Normalizes amount by sender's liquidity |
| `orig_zero_balance_flag` / `dest_zero_balance_flag` | Detects drained or fresh "mule" accounts |
| `suspicious_transaction_flag` | Encodes the classic "empty the account" fraud signature |
| `large_transaction_flag` | Flags outlier-sized transactions (95th percentile) |
| `orig_balance_error` / `dest_balance_error` | Detects ledger inconsistencies |

## 🚀 Getting Started

### 1. Clone & install
```bash
git clone <your-repo-url>
cd Digital_Payment_Fraud_Detection_DecisionTree
pip install -r requirements.txt
```

### 2. (Optional) Retrain the model
```bash
python3 train_pipeline.py
```
This regenerates everything in `/model` and `/assets` from `data/paysim_sample_15000.csv`.

### 3. Launch the dashboard
```bash
streamlit run app.py
```

### 4. Explore the notebook
```bash
jupyter notebook notebook/Fraud_Detection_DecisionTree.ipynb
```

## 🖥️ Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Home | KPI overview, workflow, business impact |
| 🔮 Fraud Prediction | Live transaction scoring with risk gauge & explanation |
| 📊 Dataset Explorer | Interactive EDA on the sample dataset |
| 🌳 Decision Tree Visualizer | Tree diagram, depth/leaves, extracted rules |
| 📈 Model Performance | Accuracy/Precision/Recall/F1/ROC-AUC, confusion matrix |
| 🎯 Feature Importance | Ranked feature contributions |
| 🧠 AI Insights | Plain-language fraud rules |
| ℹ️ About Project | Tech stack & project rationale |

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo.
3. Set the main file path to `app.py`.
4. Deploy — the app reads pre-trained artifacts from `/model`, so no retraining is needed at deploy time.

## 🧠 Decision Tree Theory (Summary)

The notebook includes a full walkthrough of Gini impurity, entropy, information gain, key hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, `class_weight`, `ccp_alpha`), overfitting/underfitting, and cost-complexity pruning — with code, math, and visualizations.

## 💰 Why Recall Matters in Fraud Detection

A **false negative** (missed fraud) causes direct, often irreversible monetary loss. A **false positive** (legitimate transaction flagged) causes customer friction but is recoverable. This project therefore tunes `GridSearchCV` on **recall**, not raw accuracy.

## 📜 License

MIT — see [LICENSE](LICENSE).

## 🙏 Acknowledgements

- Dataset: [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) by Edgar Lopez-Rojas et al.
- Built with scikit-learn, Streamlit, and Plotly.
