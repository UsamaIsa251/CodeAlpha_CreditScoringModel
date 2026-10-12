from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "best_credit_scoring_model.joblib"

app = FastAPI(
    title="Credit Scoring API",
    description=(
        "An educational API that predicts credit risk using a trained "
        "machine-learning pipeline."
    ),
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


class CreditApplication(BaseModel):
    checking_account: str = Field(pattern=r"^A1[1-4]$")
    duration: int = Field(ge=1, le=72)
    credit_history: str = Field(pattern=r"^A3[0-4]$")
    purpose: str = Field(pattern=r"^A4(0|1|2|3|4|5|6|7|8|9|10)$")
    credit_amount: float = Field(gt=0)
    savings_account: str = Field(pattern=r"^A6[1-5]$")
    employment_since: str = Field(pattern=r"^A7[1-5]$")
    installment_rate: int = Field(ge=1, le=4)
    personal_status_sex: str = Field(pattern=r"^A9[1-5]$")
    other_debtors: str = Field(pattern=r"^A10[1-3]$")
    residence_since: int = Field(ge=1, le=4)
    property: str = Field(pattern=r"^A12[1-4]$")
    age: int = Field(ge=18, le=100)
    other_installment_plans: str = Field(pattern=r"^A14[1-3]$")
    housing: str = Field(pattern=r"^A15[1-3]$")
    existing_credits: int = Field(ge=1, le=4)
    job: str = Field(pattern=r"^A17[1-4]$")
    dependents: int = Field(ge=1, le=2)
    telephone: str = Field(pattern=r"^A19[1-2]$")
    foreign_worker: str = Field(pattern=r"^A20[1-2]$")


@app.get("/")
def home():
    return {
        "message": "Credit Scoring API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(application: CreditApplication):
    try:
        applicant = pd.DataFrame([application.model_dump()])
        prediction = int(model.predict(applicant)[0])
        probabilities = model.predict_proba(applicant)[0]
        bad_probability = float(probabilities[1])

        return {
            "prediction": prediction,
            "predicted_class": (
                "bad_credit" if prediction == 1 else "good_credit"
            ),
            "bad_credit_probability": round(bad_probability, 4),
            "good_credit_probability": round(float(probabilities[0]), 4),
            "model": "Logistic Regression",
            "notice": (
                "Educational estimate only; not a definitive lending decision."
            ),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs for details.",
        ) from exc
