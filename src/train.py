import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from utils import (
    NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL,
    load_csv, save_model, save_metrics, optimal_threshold
)

def plot_confusion(cm, classes, path):
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(4.2, 3.6))
    plt.plot(fpr, tpr, label=f'ROC AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'--', lw=1, color='grey')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def plot_pr(y_true, y_prob, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(4.2, 3.6))
    plt.plot(recall, precision, label=f'AP={ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/Fraud.csv")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", type=str, default="xgb", choices=["xgb"])
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load
    df = load_csv(args.data, require_target=True)

    # Split
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Preprocess
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    # Class imbalance handling (SMOTE) + XGBoost
    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1
    )

    pipe = ImbPipeline(steps=[
        ("pre", pre),
        ("smote", SMOTE(random_state=args.seed)),
        ("clf", clf)
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Predict prob
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Threshold by F1
    th = optimal_threshold(y_test, y_prob, beta=1.0)
    y_pred = (y_prob >= th).astype(int)

    # Metrics
    roc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save artifacts
    save_model(pipe, os.path.join(args.results_dir, "model.joblib"))
    save_metrics({
        "roc_auc": float(roc),
        "pr_auc": float(ap),
        "threshold": float(th),
        "classification_report": report
    }, os.path.join(args.results_dir, "metrics.json"))

    # Plots
    plot_confusion(cm, ["Not Fraud", "Fraud"], os.path.join(args.results_dir, "confusion_matrix.png"))
    plot_roc(y_test, y_prob, os.path.join(args.results_dir, "roc_curve.png"))
    plot_pr(y_test, y_prob, os.path.join(args.results_dir, "pr_curve.png"))

    print(json.dumps({
        "roc_auc": roc, "pr_auc": ap, "threshold": th,
        "confusion_matrix": cm.tolist()
    }, indent=2))

if __name__ == "__main__":
    main()
