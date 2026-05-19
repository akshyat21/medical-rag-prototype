import requests

url = "http://localhost:8000/v1/local_rag"
payload = {"question": "What is the first-line treatment for hypertension?"}

resp = requests.post(url, json=payload)
if resp.status_code == 200:
    data = resp.json()
    print("Answer:", data["answer"])
    print("\nContext used:\n", data["context_used"])
else:
    print("Error:", resp.status_code, resp.text)