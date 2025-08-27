from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from utils import load_model, NUMERIC_COLS, CATEGORICAL_COLS

app = FastAPI(title="Fraud Detection API")
model = load_model("results/model.joblib")

class Tx(BaseModel):
    step: float
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(tx: Tx):
    df = pd.DataFrame([tx.dict()])
    if model is None:
        return {"fraud_score": 0.5, "note": "No trained model found."}
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    score = float(model.predict_proba(X)[:, 1][0])
    return {"fraud_score": score}
