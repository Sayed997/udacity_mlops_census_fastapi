# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np

from starter.ml.data import process_data
from starter.ml.model import inference


# Load model and encoders
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("model/lb.pkl", "rb") as f:
    lb = pickle.load(f)


# Add pydantic class to prcess API input
class CensusInput(BaseModel):
    age: int
    fnlgt: int
    workclass: str
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    education_num: int = Field(alias="education-num")
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 39,
                "fnlgt": 2147,
                "workclass": "State-gov",
                "education": "Bachelors",
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "education-num": 13,
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


# Categorical feature list
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

app = FastAPI()


@app.get("/")
def read_root() -> dict:

    return {"message": "Welcome to the Census Income prediction API"}


@app.post("/predict")
def predict(input_data: CensusInput) -> dict:

    # Convert Pydantic model -> DataFrame
    data = pd.DataFrame([{
        "age": input_data.age,
        "fnlgt": input_data.fnlgt,
        "workclass": input_data.workclass,
        "education": input_data.education,
        "marital-status": input_data.marital_status,
        "occupation": input_data.occupation,
        "relationship": input_data.relationship,
        "race": input_data.race,
        "sex": input_data.sex,
        "hours-per-week": input_data.hours_per_week,
        "native-country": input_data.native_country,
        "education-num": input_data.education_num,
        "capital-gain": input_data.capital_gain,
        "capital-loss": input_data.capital_loss
    }])

    # Process data using the same encoder and lb as training
    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run inference
    pred = inference(model, X)[0]

    # Convert 0/1 - > "<=50K" or ">50K"
    label = lb.inverse_transform(np.array([pred]))[0]

    return {"prediction": label}
