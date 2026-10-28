import pytest
from fastapi.testclient import TestClient

from src.app import main
from src.train.train import train


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    train(model_path=path)
    return path


@pytest.fixture()
def client(model_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", str(model_path))
    main.get_model.cache_clear()
    yield TestClient(main.app)
    main.get_model.cache_clear()