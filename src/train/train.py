"""Train the churn model and save it to disk."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"

DATA_PATH = Path("data/churn_v1.csv")
MODEL_PATH = Path("models/model.joblib")


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": round(f1_score(y_test, proba > 0.5), 3),
        "auc": round(roc_auc_score(y_test, proba), 3),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


if __name__ == "__main__":
    print(train())