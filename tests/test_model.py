import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from starter.starter.ml.model import train_model
from starter.starter.ml.model import compute_model_metrics
from starter.starter.ml.model import inference
from sklearn.metrics import precision_recall_fscore_support


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )
    return X, y


@pytest.fixture
def fitted_model(sample_data):
    X_train, y_train = sample_data
    classifier = train_model(X_train, y_train)
    return classifier


def test_train_model(fitted_model):
    assert isinstance(
        fitted_model, RandomForestClassifier
    ), "A RandomForestClassifier is required!"
    check_is_fitted(fitted_model)


def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    preds = np.array([0, 1, 0, 0, 1])
    precision_1, recall_1, f1_1, support = precision_recall_fscore_support(
        y_true, preds, average="binary"
    )
    precision_2, recall_2, f1_2 = compute_model_metrics(y_true, preds)
    assert np.isclose(
        precision_1, precision_2
    ), "Precision is not correctly calculated!"
    assert np.isclose(
        recall_1, recall_2
    ), "Recall is not correctly calculated!"
    assert np.isclose(f1_1, f1_2), "F1 score is not correctly calculated!"


def test_inference(fitted_model, sample_data):
    X, y = sample_data
    preds = inference(fitted_model, X)
    assert len(y) == len(
        preds
    ), "The number of predictions must match the number of observations!"
    unique_values = np.unique(preds)
    assert (
        unique_values.size == 2
    ), "There are TWO labels in binary classification!"
