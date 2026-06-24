import pandas as pd
import numpy as np
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, inference, compute_model_metrics
import os

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

# resolve data path for consistency
FILE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_DIR, "data", "census.csv")

def test_process_data_training():

    # Test process_data function outputs
    df = pd.read_csv(DATA_PATH).sample(100)
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    assert X.shape[0] == y.shape[0]
    assert encoder is not None
    assert lb is not None


def test_train_model():

    # Test train_model function outputs
    df = pd.read_csv(DATA_PATH).sample(100)
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert hasattr(model, "predict")

# Using training set for testing just to test model functionality here


def test_inference():

    # Test inference function model outputs
    df = pd.read_csv(DATA_PATH).sample(100)
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)


def test_compute_model_metrics():

    # Test compute model metrics function outputs with dummy data
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
