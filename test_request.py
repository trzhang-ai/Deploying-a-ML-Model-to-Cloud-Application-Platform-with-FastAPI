import requests


url = "https://deploying-a-ml-model-to-cloud-fjou.onrender.com/predict"

payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def main():
    response = requests.post(url, json=payload, timeout=120)
    print(f"Status code: {response.status_code}")
    print(response.json())


if __name__ == "__main__":
    main()
