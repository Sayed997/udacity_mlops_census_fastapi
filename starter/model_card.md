# Model Card

## Model Details

This model is a RandomForestClassifier trained to predict whether an individual earns “<=50K” or “>50K” annually based on demographic and employment-related features from the UCI Census Income dataset.
The model was trained using scikit-learn and deployed for usage using FastAPI.

**Key components**:

**Model type**: RandomForestClassifier

**Preprocessing**: OneHotEncoder for categorical features, LabelBinarizer for the target

**Training pipeline**: Provided in train_model.py

**Serving pipeline**: FastAPI app using process_data() for consistent preprocessing

## Intended Use

The model is intended for:

Educational purposes

Demonstrating ML model deployment

Practicing CI/CD, API testing, and cloud deployment workflows

Not intended for real-world decision-making, hiring, credit scoring, or any high‑stakes applications.

## Training Data
The model was trained on the UCI Adult Census dataset, which includes:

Demographic attributes (age, race, sex, education, etc.)

Employment attributes (workclass, occupation, hours-per-week)

Income label (“<=50K” or “>50K”)

The dataset contains approximately 32,000 rows of data

## Evaluation Data
A held-out test set (20% split) from the same dataset was used for evaluation.

## Metrics
The following metrics were used to evaluate model performance:

Precision

Recall

F1-score

Results: Precision: 0.748314606741573 | Recall: 0.6359007001909611 | Fbeta: 0.6875430144528561

## Ethical Considerations

The dataset contains sensitive/protected attributes (race, sex, marital status).

The model may learn historical biases present in the data.

Predictions should not be used for real-world decisions affecting individuals.

Income prediction models can reinforce socioeconomic inequalities if misused.

## Caveats and Recommendations

Users should avoid deploying this model in production without:

Bias audits

Fairness evaluations

Updated and representative training data

For educational and demonstration purposes, the model is appropriate and effective.
