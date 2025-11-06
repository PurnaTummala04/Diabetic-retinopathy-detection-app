# train.py
# Uses only CatBoost, LightGBM, and XGBoost.
# Adds Threshold vs Accuracy and ROC curve plots for each model.

import os
import re
import json
import time
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    roc_curve,
)

from imblearn.combine import SMOTETomek

import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from joblib import dump

RANDOM_STATE = 42
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Features exactly as in your code
image_features = [
    'exudates_count',
    'hemorrhages_count',
    'microaneurysms_count',
    'vessel_tortuosity',
    'macular_thickness'
]
clinical_features = ['fasting_glucose', 'hba1c', 'diabetes_duration']
features = image_features + clinical_features
target_col = "retinal_disorder"


def safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return s if s else "model"


def evaluate_model(model, X_train, y_train, X_val, y_val, name):
    """Train a model, compute metrics, and save plots."""
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_val)[:, 1]

    # --- Find best threshold ---
    thresholds = np.arange(0.0, 1.01, 0.02)
    bal_acc_scores = []
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        bal_acc_scores.append(balanced_accuracy_score(y_val, preds))
    best_idx = np.argmax(bal_acc_scores)
    best_thresh = float(thresholds[best_idx])
    best_bal_acc = float(bal_acc_scores[best_idx])

    # --- Final predictions ---
    y_pred_final = (y_probs >= best_thresh).astype(int)
    roc_auc = roc_auc_score(y_val, y_probs)
    precision, recall, _ = precision_recall_curve(y_val, y_probs)
    pr_auc = auc(recall, precision)

    print(f"\n== {name} ==")
    print(f"Best Threshold: {best_thresh:.2f}")
    print(f"Balanced Accuracy: {best_bal_acc:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(confusion_matrix(y_val, y_pred_final))
    print(classification_report(y_val, y_pred_final, zero_division=0))

    # --- Save Threshold vs Accuracy Plot ---
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, bal_acc_scores, label="Balanced Accuracy", color='blue')
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best Threshold = {best_thresh:.2f}')
    plt.xlabel("Threshold")
    plt.ylabel("Balanced Accuracy")
    plt.title(f"Threshold vs Balanced Accuracy - {name}")
    plt.legend()
    plt.grid(True)
    thr_path = os.path.join(ARTIFACT_DIR, f"{safe_name(name)}_threshold_vs_accuracy.png")
    plt.savefig(thr_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {thr_path}")

    # --- Save ROC Curve Plot ---
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', color='green')
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_path = os.path.join(ARTIFACT_DIR, f"{safe_name(name)}_roc_curve.png")
    plt.savefig(roc_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}")

    metrics = {
        "best_threshold": best_thresh,
        "balanced_accuracy": best_bal_acc,
        "auc_roc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_val, y_pred_final).tolist(),
        "classification_report": classification_report(y_val, y_pred_final, zero_division=0, output_dict=True),
    }
    return y_probs, metrics


def main(data_path: str, test_size: float = 0.2):
    data = pd.read_csv(data_path)
    X = data[features]
    y = data[target_col]
    mask = y.isin([0, 1])
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int).values

    # Clinical scaling
    X.loc[:, clinical_features] = X.loc[:, clinical_features].astype(float) * 0.1

    # SMOTETomek BEFORE splitting (as in your code)
    smote_tomek = SMOTETomek(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

    # StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_resampled), columns=features)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_resampled, stratify=y_resampled, test_size=test_size, random_state=RANDOM_STATE
    )

    # Compute scale_pos_weight
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    # --- Only three models ---
    models = {
        "XGBoost": xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            scale_pos_weight=spw,
            n_jobs=-1,
            tree_method="hist",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=8,
            learning_rate=0.05,
            scale_pos_weight=spw,
            verbose=0,
            random_seed=RANDOM_STATE,
            loss_function="Logloss",
        ),
    }

    # --- Train + Evaluate ---
    probs = {}
    metrics_all = {}
    fitted_models = {}

    for name, model in models.items():
        y_probs, m = evaluate_model(model, X_train, y_train, X_val, y_val, name)
        probs[name] = y_probs
        metrics_all[name] = m
        fitted_models[name] = model

    # --- Ensemble (mean) ---
    ensemble_probs = np.mean(np.column_stack(list(probs.values())), axis=1)
    thresholds = np.arange(0.4, 0.65, 0.05)
    best_bal_acc, best_thresh = 0.0, 0.5
    for thr in thresholds:
        y_pred = (ensemble_probs >= thr).astype(int)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = float(thr)

    y_pred_final = (ensemble_probs >= best_thresh).astype(int)
    roc_auc = roc_auc_score(y_val, ensemble_probs)
    precision, recall, _ = precision_recall_curve(y_val, ensemble_probs)
    prauc = auc(recall, precision)

    print("\n== Ensemble ==")
    print(f"Threshold: {best_thresh:.2f}")
    print(f"Balanced Accuracy: {best_bal_acc:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"PR-AUC: {prauc:.4f}")
    print(confusion_matrix(y_val, y_pred_final))
    print(classification_report(y_val, y_pred_final, zero_division=0))

    ensemble_metrics = {
        "threshold": best_thresh,
        "balanced_accuracy": best_bal_acc,
        "auc_roc": roc_auc,
        "pr_auc": prauc,
        "confusion_matrix": confusion_matrix(y_val, y_pred_final).tolist(),
        "classification_report": classification_report(y_val, y_pred_final, zero_division=0, output_dict=True),
    }

    # --- Save artifacts ---
    dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    model_filenames = {}
    for name, mdl in fitted_models.items():
        fname = f"{safe_name(name)}.pkl"
        dump(mdl, os.path.join(MODEL_DIR, fname))
        model_filenames[name] = fname

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "threshold_grid": list(map(float, thresholds)),
        "ensemble_threshold": float(best_thresh),
        "model_order": list(models.keys()),
        "model_filenames": model_filenames,
        "random_state": RANDOM_STATE,
        "data_path": os.path.abspath(data_path),
    }
    with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    metrics_out = {"per_model": metrics_all, "ensemble": ensemble_metrics}
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\nSaved:")
    print(f"- Scaler: {os.path.join(ARTIFACT_DIR, 'scaler.pkl')}")
    print(f"- Models: {MODEL_DIR}/*.pkl")
    print(f"- Metadata: {os.path.join(ARTIFACT_DIR, 'metadata.json')}")
    print(f"- Metrics: {os.path.join(ARTIFACT_DIR, 'metrics.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to your CSV")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation size (default 0.2)")
    args = parser.parse_args()
    main(args.data, test_size=args.test_size)
