from fastapi.testclient import TestClient
from starter.main import app


client = TestClient(app)


def test_get_root():

    # simple get request test
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income prediction API"}


# to test the predict endpoint we can test using a high paying and low paying observation
payload_low = {
    "age": 25,
    "fnlgt": 226802,
    "workclass": "Private",
    "education": "HS-grad",
    "marital-status": "Never-married",
    "occupation": "Handlers-cleaners",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "education-num": 9,
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

payload_high = {
    "age": 52,
    "fnlgt": 287927,
    "workclass": "Private",
    "education": "Masters",
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "education-num": 14,
    "capital-gain": 15000,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States"
}


def test_post_predict_low_income():

    # test predict output of low income obervation
    response = client.post("/predict", json=payload_low)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


def test_post_predict_high_income():

    # test predict output of high income obervation
    response = client.post("/predict", json=payload_high)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
