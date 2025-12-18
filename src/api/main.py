from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
from src.api.pydantic_models import CustomerTransaction, PredictionResponse

app = FastAPI(title="Credit Risk Scoring API")

# Load the best model from MLflow Registry
MODEL_NAME = "Best_CreditRisk_Model"
MODEL_STAGE = "Production"  # Or "Staging"

model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

@app.get("/")
def root():
    return {"message": "Credit Risk Scoring API is running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: CustomerTransaction):
    # Convert input into dataframe
    input_df = pd.DataFrame([transaction.dict()])

    # Predict probability
    risk_proba = model.predict_proba(input_df)[:, 1][0]
    risk_label = int(risk_proba > 0.5)

    return PredictionResponse(risk_probability=risk_proba, risk_label=risk_label)
