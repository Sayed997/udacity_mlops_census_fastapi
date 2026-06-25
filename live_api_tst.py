import requests

URL = "https://census-income-api-egrx.onrender.com/proxy/8000/predict"

payload = {
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

response = requests.post(URL, json=payload)

print("Status:", response.status_code)
print("Raw Response:", response.text)
