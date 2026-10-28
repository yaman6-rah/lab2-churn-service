import joblib
import pytest

VALID = {
    "tenure_months": 5,
    "monthly_charges": 89.9,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "support_calls": 3,
}


def test_health_contract(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_contract(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability"}
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_model_loads(model_path):
    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba")


@pytest.mark.parametrize(
    "bad",
    [
        {**VALID, "contract": "banana"},
        {**VALID, "tenure_months": -1},
        {**VALID, "extra_field": 1},
        {k: v for k, v in VALID.items() if k != "support_calls"},
    ],
)
def test_input_schema_rejects_invalid(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_edge_case_minimum_values(client):
    edge = {
        "tenure_months": 0,
        "monthly_charges": 12.0,
        "contract": "two-year",
        "internet_service": "none",
        "support_calls": 0,
    }
    r = client.post("/predict", json=edge)
    assert r.status_code == 200
    assert 0.0 <= r.json()["churn_probability"] <= 1.0


def test_prediction_is_deterministic(client):
    first = client.post("/predict", json=VALID).json()["churn_probability"]
    second = client.post("/predict", json=VALID).json()["churn_probability"]
    assert first == second == pytest.approx(0.866, abs=0.001)