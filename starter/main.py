import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data


class InferenceInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        }
    )
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(alias="capital-gain")
    capital_loss: float = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


class BatchInferenceInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "records": [
                        {
                            "age": 39,
                            "workclass": "State-gov",
                            "fnlgt": 77516,
                            "education": "Bachelors",
                            "education-num": 13,
                            "marital-status": "Never-married",
                            "occupation": "Adm-clerical",
                            "relationship": "Not-in-family",
                            "race": "White",
                            "sex": "Male",
                            "capital-gain": 2174,
                            "capital-loss": 0,
                            "hours-per-week": 40,
                            "native-country": "United-States",
                        },
                        {
                            "age": 52,
                            "workclass": "Self-emp-not-inc",
                            "fnlgt": 209642,
                            "education": "HS-grad",
                            "education-num": 9,
                            "marital-status": "Married-civ-spouse",
                            "occupation": "Exec-managerial",
                            "relationship": "Husband",
                            "race": "White",
                            "sex": "Male",
                            "capital-gain": 0,
                            "capital-loss": 0,
                            "hours-per-week": 45,
                            "native-country": "United-States",
                        },
                    ]
                }
            ]
        }
    )
    records: list[InferenceInput] = Field(..., min_length=1)


class InferenceOutput(BaseModel):
    prediction: str


class BatchInferenceOutput(BaseModel):
    predictions: list[InferenceOutput]


ml_models = {}
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ml_models["lb"] = joblib.load("starter/model/lb.pkl")
    ml_models["encoder"] = joblib.load("starter/model/encoder.pkl")
    ml_models["classifier"] = joblib.load("starter/model/model.pkl")
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Welcome to the Census Income Prediction API"}


@app.post("/predict", response_model=InferenceOutput)
def predict(payload: InferenceInput) -> InferenceOutput:
    record = payload.model_dump(by_alias=True)
    row = pd.DataFrame([record])
    X, _, _, _ = process_data(
        row,
        cat_features,
        label=None,
        training=False,
        encoder=ml_models["encoder"],
        lb=ml_models["lb"],
    )
    prediction = (
        ml_models["lb"]
        .inverse_transform(inference(ml_models["classifier"], X))
        .item()
    )
    return InferenceOutput(prediction=prediction)


@app.post("/batch_predict", response_model=BatchInferenceOutput)
def batch_predict(payload: BatchInferenceInput) -> BatchInferenceOutput:
    records = [record.model_dump(by_alias=True) for record in payload.records]
    rows = pd.DataFrame(records)
    X, _, _, _ = process_data(
        rows,
        cat_features,
        label=None,
        training=False,
        encoder=ml_models["encoder"],
        lb=ml_models["lb"],
    )
    predictions = (
        ml_models["lb"]
        .inverse_transform(inference(ml_models["classifier"], X))
        .tolist()
    )
    return BatchInferenceOutput(
        predictions=[InferenceOutput(prediction=pred) for pred in predictions]
    )
