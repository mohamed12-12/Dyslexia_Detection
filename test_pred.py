import requests
import json

with open("data.json") as f:
    data = json.load(f)

response = requests.post("http://127.0.0.1:5000/predict", json=data)
print(response.json())