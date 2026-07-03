"""
Fraud_Detection_DecisionTree - Training Pipeline
==================================================
Loads the PaySim sample, engineers fraud-detection features, trains a
tuned Decision Tree Classifier, evaluates it, and persists all model
artifacts (model, encoders, feature list, metadata) plus diagnostic
plots used by the notebook / README / Streamlit app.

Run:  python3 train_pipeline.py
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "paysim_sample_15000.csv"
MODEL_DIR = ROOT / "model"
ASSET_DIR = ROOT / "assets"
MODEL_DIR.mkdir(exist_ok=True)
ASSET_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded sample: {df.shape}")

# --------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------------
df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df["orig_zero_balance_flag"] = (df["newbalanceOrig"] == 0).astype(int)
df["dest_zero_balance_flag"] = (df["oldbalanceDest"] == 0).astype(int)
df["suspicious_transaction_flag"] = (
    (df["oldbalanceOrg"] == df["amount"]) & (df["newbalanceOrig"] == 0)
).astype(int)
df["large_transaction_flag"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
df["orig_balance_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
df["dest_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

# Encode categorical transaction type
type_encoder = LabelEncoder()
df["type_encoded"] = type_encoder.fit_transform(df["type"])

FEATURE_COLUMNS = [
    "step", "type_encoded", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "amount_to_balance_ratio",
    "orig_zero_balance_flag", "dest_zero_balance_flag",
    "suspicious_transaction_flag", "large_transaction_flag",
    "orig_balance_error", "dest_balance_error",
    "isFlaggedFraud",
]

X = df[FEATURE_COLUMNS]
y = df["isFraud"]

# --------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# --------------------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (GridSearchCV, Decision Tree ONLY)
# --------------------------------------------------------------------------
param_grid = {
    "max_depth": [4, 5, 6, 7, 8, 10],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"],
    "class_weight": [None, "balanced"],
}

base_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    base_tree, param_grid, cv=5, scoring="recall", n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best params:", best_params)

model = grid_search.best_estimator_

# Cost complexity pruning check (informational)
path = base_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# --------------------------------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="recall")
metrics["cv_recall_mean"] = float(cv_scores.mean())
metrics["cv_recall_std"] = float(cv_scores.std())

print(json.dumps(metrics, indent=2))
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# --------------------------------------------------------------------------
# 6. PLOTS
# --------------------------------------------------------------------------
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.title("Confusion Matrix - Decision Tree")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(ASSET_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# Feature Importance
importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(ASSET_DIR / "feature_importance.png", dpi=150)
plt.close()

# Tree Visualization (depth-limited for readability)
plt.figure(figsize=(22, 12))
plot_tree(
    model, max_depth=3, feature_names=FEATURE_COLUMNS,
    class_names=["Legit", "Fraud"], filled=True, rounded=True, fontsize=8
)
plt.title("Decision Tree Structure (first 3 levels)")
plt.tight_layout()
plt.savefig(ASSET_DIR / "tree_visualization.png", dpi=150)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}", color="#2563eb")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(ASSET_DIR / "roc_curve.png", dpi=150)
plt.close()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(rec, prec, color="#059669")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig(ASSET_DIR / "precision_recall_curve.png", dpi=150)
plt.close()

# --------------------------------------------------------------------------
# 7. RULE EXTRACTION (top of tree, text form)
# --------------------------------------------------------------------------
tree_rules_text = export_text(model, feature_names=FEATURE_COLUMNS, max_depth=4)
with open(MODEL_DIR / "tree_rules.txt", "w") as f:
    f.write(tree_rules_text)

# --------------------------------------------------------------------------
# 8. SAVE ARTIFACTS
# --------------------------------------------------------------------------
with open(MODEL_DIR / "decision_tree.pkl", "wb") as f:
    pickle.dump(model, f)

with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(type_encoder, f)

with open(MODEL_DIR / "feature_columns.pkl", "wb") as f:
    pickle.dump(FEATURE_COLUMNS, f)

metadata = {
    "model_type": "DecisionTreeClassifier",
    "random_state": RANDOM_STATE,
    "best_params": best_params,
    "n_features": len(FEATURE_COLUMNS),
    "feature_columns": FEATURE_COLUMNS,
    "transaction_types": list(type_encoder.classes_),
    "tree_depth": int(model.get_depth()),
    "tree_leaves": int(model.get_n_leaves()),
    "metrics": metrics,
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "sample_size": int(len(df)),
    "fraud_ratio_in_sample": float(y.mean()),
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nAll artifacts saved to /model and /assets")
print(json.dumps(metadata, indent=2))
