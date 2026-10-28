"""Churn prediction API with validated input."""
import os
from functools import lru_cache
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.train.train import FEATURES

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=500)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    support_calls: int = Field(ge=0, le=50)


class Prediction(BaseModel):
    churn_probability: float


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODEL_PATH)


app = FastAPI(title="AI381 Lab 2 churn API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    X = pd.DataFrame([customer.model_dump()])[FEATURES]
    p = float(get_model().predict_proba(X)[0, 1])
    return Prediction(churn_probability=round(p, 3))