import requests

url = "http://localhost:8000/v1/completions"
payload = {
    "prompt": "What is the first-line treatment for hypertension?",
    "max_tokens": 100,
    "temperature": 0.5
}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        answer = response.json()["choices"][0]["text"]
        print("Answer:", answer)
    else:
        print("Error:", response.status_code, response.text)
except requests.exceptions.RequestException as e:
    print("Request failed:", e)