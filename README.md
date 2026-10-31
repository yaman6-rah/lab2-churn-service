# Churn Prediction API

[![CI](https://github.com/yaman6-rah/lab2-churn-service/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yaman6-rah/lab2-churn-service/actions/workflows/ci.yml)

A containerised FastAPI service that predicts customer churn (logistic regression trained on `data/churn_v1.csv`).
Built, scanned with Trivy and published to GHCR by GitHub Actions on every merge to `main`.

## Run it (one command)

```bash
docker run --rm -p 8000:8000 ghcr.io/yaman6-rah/lab2-churn-service:<commit-sha>
```

Replace `<commit-sha>` with a tag from the [Packages page](https://github.com/yaman6-rah/lab2-churn-service/pkgs/container/lab2-churn-service). Or build and run locally:

```bash
docker build -t churn-api . && docker run --rm -p 8000:8000 churn-api
```

## Try it

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure_months": 5, "monthly_charges": 89.9, "contract": "month-to-month", "internet_service": "fiber", "support_calls": 3}'
```

Expected: `{"status":"ok"}` and `{"churn_probability":0.866}`.
Invalid input (e.g. `"contract": "banana"`) returns HTTP 422.

## Develop

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train.train   # writes models/model.joblib
pytest -v && ruff check .
```

## CI

- **Every pull request:** `ruff` and `pytest`.
- **On `main`:** build image, Trivy scan (fails on HIGH/CRITICAL), push to GHCR tagged with the commit SHA.
- `main` is protected: changes go through a pull request and the `test` check must pass.

