import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    classification_report,
)


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    return classifier


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def prepare_single_report(
    test_index, preds, entire_dataset, encoder, lb, cat_features
):
    df = pd.concat(
        [
            entire_dataset.iloc[test_index],
            pd.DataFrame(
                lb.inverse_transform(preds),
                columns=["Prediction"],
                index=test_index,
            ),
        ],
        join="inner",
        axis=1,
    )
    performance_report = pd.DataFrame()
    for cat_feature in cat_features:
        for slice_name, slice in df.groupby(cat_feature):
            slice_performance = (
                pd.DataFrame(
                    classification_report(
                        slice["salary"],
                        slice["Prediction"],
                        digits=3,
                        output_dict=True,
                        zero_division=0,
                    )
                )
                .drop(columns=["accuracy", "macro avg", "weighted avg"])
                .transpose()
            )
            slice_performance = pd.concat(
                [slice_performance],
                keys=[(cat_feature, slice_name)],
                names=["feature", "slice"],
            )
            performance_report = pd.concat(
                [performance_report, slice_performance], axis=0
            )
    performance_report = performance_report.reset_index().rename(
        {"level_2": "label"}, axis=1
    )
    return performance_report


def prepare_final_report(reports):
    report = pd.concat(reports, axis=0)
    report = report.groupby(["feature", "slice", "label"]).agg(["mean", "std"])
    with open("slice_output.txt", "w", encoding="utf-8") as outfile:
        print(report.to_string(), file=outfile)
    return report


def prepare_single_slice_report(reports, cat_feature, value):
    report = pd.concat(reports, axis=0)
    report = report.groupby(["feature", "slice", "label"]).agg(["mean", "std"])
    report = report.xs((cat_feature, value), level=["feature", "slice"])
    return report
