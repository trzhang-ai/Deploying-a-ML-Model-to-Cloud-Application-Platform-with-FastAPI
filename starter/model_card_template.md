# Model Card

## Model Details

This project contains a Random Forest binary classification model trained with
scikit-learn 1.9.0. The model uses the default `RandomForestClassifier`
hyperparameters and was trained with Python 3.13.3. The final model is trained
on the complete local training dataset and is saved with its categorical encoder
and label binarizer.

This is an educational course project. It is not presented as a production-ready
or independently validated income-prediction system. The reporting structure is
informed by [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).

## Intended Use

The model predicts whether an individual belongs to the `<=50K` or `>50K`
annual-income category in the Census Income dataset. Its intended use is to
demonstrate a machine-learning training, evaluation, serialization, and
inference workflow.

The model is intended for learning, experimentation, and technical demonstration
by students and developers. It must not be used to make or automate decisions
about a person's employment, credit, insurance, housing, education, taxation,
government benefits, or other high-impact opportunities.

## Training Data

The model uses the cleaned `starter/data/census.csv` file from this project. The
local file contains 32,561 rows, 14 input features, and the binary `salary`
label. The label values are `<=50K` and `>50K`; `>50K` is treated as the positive
class when calculating the reported metrics.

The input contains numerical features and categorical features. The eight
categorical features are `workclass`, `education`, `marital-status`,
`occupation`, `relationship`, `race`, `sex`, and `native-country`. Categorical
features are one-hot encoded, while numerical features are passed to the model
without scaling. Unknown categories at inference time are ignored by the
encoder. Values represented by `?` in the source data are not imputed; they are
handled as categorical values by the current preprocessing pipeline.

For evaluation, the script uses five-fold stratified cross-validation with
shuffling and `random_state=42`. Each fold fits its encoder only on the training
portion and applies that encoder to the held-out portion.

## Evaluation Data

Each evaluation fold is the held-out 20 percent of the local dataset. The five
folds together cover the complete local dataset, but there is no separate
external test set. Therefore, these results estimate performance on data drawn
from the same source and distribution as the training data; they do not establish
performance on a new population or a future time period.

## Metrics

The reported metrics are precision, recall, and F1 score for the positive class
(`>50K`). Precision describes how often positive predictions are correct. Recall
describes how many of the actual positive examples are found. F1 is the harmonic
mean of precision and recall. These metrics are reported instead of relying only
on accuracy because the two income classes are imbalanced.

The values below are the mean and standard deviation across the five stratified
cross-validation folds:

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Precision | 0.735 | 0.007 |
| Recall | 0.627 | 0.010 |
| F1 score | 0.677 | 0.008 |

The implementation uses the model's default class decision rule and does not
perform probability calibration or threshold optimization. Slice-level results
for the categorical features are written to `slice_output.txt`. Those results
include the mean and standard deviation across folds for each slice.

## Factors and Slice Analysis

Performance may vary across education, work class, marital status, occupation,
relationship, race, sex, and native-country slices. The project evaluates each
unique value of these categorical features separately. This disaggregated
analysis is intended to make differences in model behavior visible before any
use of the model.

The slice report is not a fairness certification. Some slices contain few
examples, so their metrics can have large uncertainty. The current analysis is
also limited to one feature at a time; it does not provide intersectional
analysis, such as the joint effect of sex and race.

## Ethical Considerations

The dataset contains demographic and socioeconomic attributes, including race,
sex, marital status, and native country. These attributes can reflect historical
inequalities and measurement choices in the source data. A prediction of an
income category is not a measure of a person's ability, merit, worth, or causal
potential to earn income.

Because the target concerns income and the features include sensitive or
potentially sensitive characteristics, errors may affect groups differently.
The model should not be used for consequential decisions about individuals. Any
research or deployment beyond this educational project would require a review of
data provenance, privacy, legal requirements, fairness, calibration, and human
oversight. The public availability of the dataset should not be interpreted as a
guarantee that it is appropriate for every proposed use.

## Caveats and Recommendations

1. The evaluation uses cross-validation on one historical dataset and has no
   external or temporal validation set. Evaluate on a representative, separately
   governed dataset before considering any real-world use.
2. Slice metrics with small support are unstable. Always report the slice support
   together with the metric and avoid ranking groups based on very small samples.
3. The model uses default Random Forest settings and has not been tuned,
   calibrated, or constrained for fairness. These steps should be investigated
   separately rather than inferred from the aggregate metrics.
4. Monitor performance and group-level error rates after deployment, because
   population characteristics, data collection, and the relationship between
   features and income can change over time.
5. Keep the serialized model, encoder, label binarizer, dependency lock file,
   and preprocessing code versioned together so that predictions can be
   reproduced.
