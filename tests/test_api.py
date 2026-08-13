import pytest
from starter.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_read_main(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Prediction API"
    }


def test_predict_less_50K(client):
    response = client.post(
        "/predict",
        json={
            "age": 39,
            "capital-gain": 2174,
            "capital-loss": 0,
            "education": "Bachelors",
            "education-num": 13,
            "fnlgt": 77516,
            "hours-per-week": 40,
            "marital-status": "Never-married",
            "native-country": "United-States",
            "occupation": "Adm-clerical",
            "race": "White",
            "relationship": "Not-in-family",
            "sex": "Male",
            "workclass": "State-gov",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_predict_more_50K(client):
    response = client.post(
        "/predict",
        json={
            "age": 52,
            "capital-gain": 0,
            "capital-loss": 0,
            "education": "HS-grad",
            "education-num": 9,
            "fnlgt": 209642,
            "hours-per-week": 45,
            "marital-status": "Married-civ-spouse",
            "native-country": "United-States",
            "occupation": "Exec-managerial",
            "race": "White",
            "relationship": "Husband",
            "sex": "Male",
            "workclass": "Self-emp-not-inc",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_batch_predict(client):
    response = client.post(
        "/batch_predict",
        json={
            "records": [
                {
                    "age": 39,
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "education": "Bachelors",
                    "education-num": 13,
                    "fnlgt": 77516,
                    "hours-per-week": 40,
                    "marital-status": "Never-married",
                    "native-country": "United-States",
                    "occupation": "Adm-clerical",
                    "race": "White",
                    "relationship": "Not-in-family",
                    "sex": "Male",
                    "workclass": "State-gov",
                },
                {
                    "age": 52,
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "education": "HS-grad",
                    "education-num": 9,
                    "fnlgt": 209642,
                    "hours-per-week": 45,
                    "marital-status": "Married-civ-spouse",
                    "native-country": "United-States",
                    "occupation": "Exec-managerial",
                    "race": "White",
                    "relationship": "Husband",
                    "sex": "Male",
                    "workclass": "Self-emp-not-inc",
                },
            ]
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "predictions": [{"prediction": "<=50K"}, {"prediction": ">50K"}]
    }
