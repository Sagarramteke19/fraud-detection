import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from utils import load_model, NUMERIC_COLS, CATEGORICAL_COLS, TARGET_COL

st.set_page_config(page_title='Fraud Detection Demo', page_icon='🕵️', layout='wide')
st.title('🕵️ Fraud Detection – Batch Scoring')

# Load model + metrics if available
model = load_model("results/model.joblib")
metrics_path = "results/metrics.json"
metrics = None
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

colA, colB, colC = st.columns(3)
with colA:
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics else "—")
with colB:
    st.metric("PR-AUC", f"{metrics['pr_auc']:.3f}" if metrics else "—")
with colC:
    st.metric("Threshold", f"{metrics['threshold']:.2f}" if metrics else "0.50")

st.write("**Expected columns:**", NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL])

uploaded = st.file_uploader('Upload CSV (schema like data/Fraud.csv)', type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview", df.head())
    # Allow unlabeled files too
    missing = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        # If no model, return random scores
        if model is None:
            st.warning("No trained model found at results/model.joblib. Showing random scores.")
            scores = np.random.rand(len(df))
        else:
            scores = model.predict_proba(df[NUMERIC_COLS + CATEGORICAL_COLS])[:, 1]
        out = df.copy()
        out["fraud_score"] = scores
        if metrics:
            th = float(metrics["threshold"])
            out["fraud_flag"] = (out["fraud_score"] >= th).astype(int)
        st.subheader("Top suspicious transactions")
        st.dataframe(out.sort_values("fraud_score", ascending=False).head(25))
        st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"),
                           file_name="scored_transactions.csv")
else:
    st.info("Upload a CSV to begin.")
