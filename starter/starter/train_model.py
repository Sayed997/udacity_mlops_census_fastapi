# Script to train machine learning model.

# Add the necessary imports for the starter code.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, compute_model_metrics, inference, compute_slice_metrics

# Add code to load in the data.
# Load cleaned census data
FILE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_DIR, "..", "data", "census.csv")

data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# Data splitting
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

# process data with pipeline
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)


# Train and save a model.
model = train_model(X_train, y_train)

# Predict
preds = inference(model, X_test)

# Evaluation metrics
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# resolve paths so saving can run with consistency
FILE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(FILE_DIR, "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(encoder, f)

LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")

with open(LB_PATH, "wb") as f:
    pickle.dump(lb, f)

SLICE_PATH = os.path.join(MODEL_DIR, "slice_output.txt")

compute_slice_metrics(
    test,
    categorical_features=cat_features,
    model=model,
    encoder=encoder,
    lb=lb,
    output_path=SLICE_PATH
)
