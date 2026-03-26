from fastapi import FastAPI
from app.schema import Transaction
from app.predictor import FraudPredictor

app = FastAPI(
    title="Real-Time Credit Card Fraud Detection System",
    version="1.0"
)

predictor = FraudPredictor()


@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}


@app.post("/predict")
def predict(transaction: Transaction):
    pred, prob = predictor.predict(transaction.features)

    return {
        "fraud_prediction": pred,
        "fraud_probability": prob
    }