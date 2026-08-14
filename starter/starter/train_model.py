# Train, evaluate, and serialize the Census Income classifier.
import joblib
import pandas as pd
from statistics import stdev, mean
from starter.starter.ml.data import process_data
from starter.starter.ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    prepare_single_report,
    prepare_final_report,
)
from sklearn.model_selection import StratifiedKFold


def print_metric(metric_name, metric_values):
    print(
        f"{metric_name} {mean(metric_values):.3f} ± {stdev(metric_values):.3f}"
    )


# Load the cleaned Census Income dataset.
data = pd.read_csv("starter/data/census.csv", delimiter=",", engine="python")
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
# Estimate out-of-sample performance with five-fold stratified
# cross-validation.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = data.drop(columns=["salary"])
y = data["salary"]
precisions, recalls, f1s, reports = [], [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, y_train, encoder, lb = process_data(
        data.iloc[train_index],
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    X_test, y_test, encoder, lb = process_data(
        data.iloc[test_index],
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    reports.append(
        prepare_single_report(
            test_index, preds, data, encoder, lb, cat_features
        )
    )

print("Model Evaluation Results:")
print_metric("Precision", precisions)
print_metric("Recall", recalls)
print_metric("F1 Score", f1s)

# Fit the final model on all available data and save its artifacts.
X, y, encoder, lb = process_data(
    data,
    categorical_features=cat_features,
    label="salary",
    training=True,
)
model = train_model(X, y)
final_report = prepare_final_report(reports)
joblib.dump(model, "starter/model/model.pkl")
joblib.dump(encoder, "starter/model/encoder.pkl")
joblib.dump(lb, "starter/model/lb.pkl")
