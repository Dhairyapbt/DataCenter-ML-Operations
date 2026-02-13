from fastapi import FastAPI
import joblib
import pandas as pd

from api.schemas import SensorInput

# ----------------------
# Create FastAPI app
# ----------------------
app = FastAPI(
    title="Data Center Failure Prediction API",
    version="1.0.0"
)

# ----------------------
# Load model ONCE
# ----------------------
MODEL_PATH = "models/random_forest.joblib"
model = joblib.load(MODEL_PATH)

# ----------------------
# Health endpoint
# ----------------------
@app.get("/health")
def health():
    return {"status": "OK"}

# ----------------------
# Prediction endpoint
# ----------------------
@app.post("/predict")
def predict_failure(data: SensorInput):

    input_df = pd.DataFrame([data.dict()])
    failure_prob = model.predict_proba(input_df)[0][1]

    risk_level = (
        "LOW" if failure_prob < 0.3 else
        "MEDIUM" if failure_prob < 0.6 else
        "HIGH"
    )

    return {
        "failure_probability": round(float(failure_prob), 3),
        "risk_level": risk_level
    }
