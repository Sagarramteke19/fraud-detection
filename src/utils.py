import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Expected schema for the common transactions dataset
NUMERIC_COLS = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest'
]
CATEGORICAL_COLS = ['type']
TARGET_COL = 'isFraud'  # set None if unlabeled scoring-only

def check_columns(df: pd.DataFrame, require_target: bool = True) -> None:
    required = NUMERIC_COLS + CATEGORICAL_COLS + ([TARGET_COL] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Present: {list(df.columns)}")

def load_csv(path: str, require_target: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    check_columns(df, require_target=require_target)
    return df

def save_model(model, path: str) -> None:
    joblib.dump(model, path)

def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None

def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 1.0) -> float:
    """Pick threshold maximizing F-beta (default F1)."""
    ths = np.linspace(0.01, 0.99, 99)
    best_th, best_f = 0.5, -1.0
    for th in ths:
        y_pred = (y_prob >= th).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        if precision + recall == 0:
            fbeta = 0.0
        else:
            b2 = beta**2
            fbeta = (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)
        if fbeta > best_f:
            best_f, best_th = fbeta, th
    return float(best_th)

def save_metrics(metrics: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
