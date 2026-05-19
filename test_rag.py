import requests

url = "http://localhost:8000/v1/rag"
payload = {
    "question": "What is the first-line treatment for hypertension?",
    "max_tokens": 150,
    "temperature": 0.5
}

response = requests.post(url, json=payload)
if response.status_code == 200:
    data = response.json()
    print("Answer:", data["answer"])
    print("\nContext used:\n", data["context_used"])
else:
    print("Error:", response.status_code, response.text)