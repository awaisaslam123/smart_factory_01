"""
Phase 3: Machine Learning - XGBoost Delivery Delay Predictor
=============================================================
Trains a binary classification model to predict late_delivery_risk.
Uses a full Scikit-Learn Pipeline with ColumnTransformer preprocessing.
Evaluates with Accuracy, Precision, Recall, F1, AUC-ROC.
Saves the pipeline and feature importance plots.

Run: python pipeline/train.py
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_PATH     = BASE_DIR / "data" / "processed" / "ml_features.csv"
MODELS_DIR    = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  SUPPLY CHAIN ML TRAINING PIPELINE")
print("=" * 60)

# ── Step 1: Load Data ──────────────────────────────────────────────
print("\n[STEP 1] Loading ML feature set...")
df = pd.read_csv(DATA_PATH)
print(f"  ✅ Loaded {len(df):,} records × {df.shape[1]} features")
print(f"  Class distribution:\n{df['late_delivery_risk'].value_counts(normalize=True).round(3)}")

# ── Step 2: Define Features ────────────────────────────────────────
TARGET = "late_delivery_risk"

CATEGORICAL_FEATURES = [
    "shipping_mode", "order_region", "market", "category_name",
    "department_name", "customer_segment", "payment_type",
]
NUMERIC_FEATURES = [
    "days_for_shipment_scheduled", "order_item_quantity",
    "order_item_discount_rate", "order_item_product_price",
    "sales", "revenue_per_unit", "is_weekend_order",
    "has_discount", "order_hour", "order_month",
    "market_risk_score", "region_risk_score", "mode_risk_score",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

# Encode categoricals as strings to avoid issues
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].astype(str)

print(f"\n  Features: {len(ALL_FEATURES)} total")
print(f"  → Categorical: {len(CATEGORICAL_FEATURES)}")
print(f"  → Numeric:     {len(NUMERIC_FEATURES)}")

# ── Step 3: Train/Test Split ───────────────────────────────────────
print("\n[STEP 2] Splitting dataset (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"  Train late rate: {y_train.mean():.1%} | Test late rate: {y_test.mean():.1%}")

# ── Step 4: Build Preprocessing Pipeline ──────────────────────────
print("\n[STEP 3] Building Scikit-Learn preprocessing pipeline...")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ── Step 5: XGBoost Classifier ────────────────────────────────────
print("\n[STEP 4] Training XGBoost classifier...")

# Calculate class weight for imbalanced data
neg_count  = (y_train == 0).sum()
pos_count  = (y_train == 1).sum()
scale_weight = round(neg_count / pos_count, 2)
print(f"  scale_pos_weight = {scale_weight} (handles class imbalance)")

xgb_clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

# Full pipeline: preprocess → SMOTE → XGBoost
# Note: use imblearn Pipeline for SMOTE
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   xgb_clf),
])

# Preprocess first, then SMOTE on transformed data
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# Apply SMOTE on processed training data
print("  Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)
print(f"  After SMOTE: {len(X_train_sm):,} training samples")

# Train XGBoost on SMOTE data
print("  Training XGBoost...")
xgb_clf.fit(
    X_train_sm, y_train_sm,
    eval_set=[(X_test_proc, y_test)],
    verbose=False,
)

# Save the full sklearn pipeline (preprocessor + classifier)
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   xgb_clf),
])

# ── Step 6: Evaluate ───────────────────────────────────────────────
print("\n[STEP 5] Evaluating model performance...")

y_pred      = xgb_clf.predict(X_test_proc)
y_pred_prob = xgb_clf.predict_proba(X_test_proc)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc_roc   = roc_auc_score(y_test, y_pred_prob)

print(f"\n  📊 MODEL PERFORMANCE METRICS:")
print(f"  {'Accuracy':<18}: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  {'Precision':<18}: {precision:.4f}")
print(f"  {'Recall':<18}: {recall:.4f}")
print(f"  {'F1-Score':<18}: {f1:.4f}")
print(f"  {'AUC-ROC':<18}: {auc_roc:.4f}")

print("\n  📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["On Time", "Late"]))

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
    "Score":  [accuracy, precision, recall, f1, auc_roc],
})
metrics_df.to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)

# ── Step 7: Generate Plots ─────────────────────────────────────────
print("\n[STEP 6] Generating evaluation plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#1a1d2e")

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["On Time", "Late"],
    yticklabels=["On Time", "Late"],
    ax=axes[0], linewidths=0.5,
)
axes[0].set_title("Confusion Matrix", color="white", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted", color="#aaa")
axes[0].set_ylabel("Actual", color="#aaa")
axes[0].tick_params(colors="white")

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color="#00d4ff", lw=2, label=f"AUC = {auc_roc:.3f}")
axes[1].plot([0, 1], [0, 1], "r--", lw=1, label="Random Classifier")
axes[1].fill_between(fpr, tpr, alpha=0.15, color="#00d4ff")
axes[1].set_xlabel("False Positive Rate", color="#aaa")
axes[1].set_ylabel("True Positive Rate", color="#aaa")
axes[1].set_title("ROC Curve", color="white", fontsize=13, fontweight="bold")
axes[1].legend(facecolor="#1a1d2e", labelcolor="white")
axes[1].tick_params(colors="white")

# Plot 3: Feature Importance (Top 15)
feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
importances   = xgb_clf.feature_importances_
feat_imp_df   = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": importances,
}).sort_values("Importance", ascending=True).tail(15)

colors = ["#00d4ff" if i >= len(feat_imp_df) - 3 else "#4a6fa5"
          for i in range(len(feat_imp_df))]
axes[2].barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
             color=colors, edgecolor="none")
axes[2].set_title("Top 15 Feature Importances", color="white", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Importance Score", color="#aaa")
axes[2].tick_params(colors="white")

plt.tight_layout()
plot_path = MODELS_DIR / "model_evaluation.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
plt.close()
print(f"  ✅ Evaluation plots saved → models/model_evaluation.png")

# ── Step 8: Save Model ─────────────────────────────────────────────
print("\n[STEP 7] Saving trained model...")

# Save preprocessor and classifier separately for flexibility
model_package = {
    "preprocessor":       preprocessor,
    "classifier":         xgb_clf,
    "feature_names":      ALL_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "numeric_features":   NUMERIC_FEATURES,
    "metrics": {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "auc_roc":   round(auc_roc, 4),
    }
}
model_path = MODELS_DIR / "supply_chain_model.joblib"
joblib.dump(model_package, model_path)
print(f"  ✅ Model saved → models/supply_chain_model.joblib")

# ── Step 9: Generate Predictions for Dashboard ────────────────────
print("\n[STEP 8] Generating predictions on full dataset...")

dash_df   = pd.read_csv(PROCESSED_DIR / "dashboard_data.csv")
feat_cols = [c for c in ALL_FEATURES if c in dash_df.columns]

X_dash = dash_df[feat_cols].copy()
for col in CATEGORICAL_FEATURES:
    if col in X_dash.columns:
        X_dash[col] = X_dash[col].astype(str)

# Fill missing features with mode/median
for col in NUMERIC_FEATURES:
    if col in X_dash.columns:
        X_dash[col].fillna(X_dash[col].median(), inplace=True)
for col in CATEGORICAL_FEATURES:
    if col in X_dash.columns:
        X_dash[col].fillna("Unknown", inplace=True)

missing_features = [f for f in ALL_FEATURES if f not in X_dash.columns]
for f in missing_features:
    X_dash[f] = 0

X_dash_proc = preprocessor.transform(X_dash[ALL_FEATURES])
dash_df["predicted_late"]       = xgb_clf.predict(X_dash_proc)
dash_df["delay_probability"]    = xgb_clf.predict_proba(X_dash_proc)[:, 1].round(3)
dash_df["risk_label"] = dash_df["delay_probability"].apply(
    lambda p: "🔴 High Risk" if p > 0.7 else ("🟡 Medium Risk" if p > 0.45 else "🟢 Low Risk")
)

pred_path = PROCESSED_DIR / "predictions.csv"
dash_df.to_csv(pred_path, index=False)
print(f"  ✅ Predictions saved → data/processed/predictions.csv")
print(f"  High Risk orders  : {(dash_df['risk_label']=='🔴 High Risk').sum():,}")
print(f"  Medium Risk orders: {(dash_df['risk_label']=='🟡 Medium Risk').sum():,}")
print(f"  Low Risk orders   : {(dash_df['risk_label']=='🟢 Low Risk').sum():,}")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE ✅")
print("=" * 60)
print(f"  AUC-ROC  : {auc_roc:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  Recall   : {recall:.4f}")
print("\n  Run: streamlit run dashboard/app.py")
