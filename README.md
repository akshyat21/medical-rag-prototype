# Medical RAG Prototype

A **Retrieval-Augmented Generation (RAG)** system that answers medical questions using a curated knowledge base.  
It combines vector search (sentence‑transformers) with a large language model (Groq Llama 3.3) to produce accurate, context‑grounded answers.

---

## 🧠 How It Works

1. **Knowledge Base** – A CSV file (`medical_data.csv`) containing medical Q&A pairs.
2. **Embeddings & Vector Search** – Each question is converted into a dense vector using `all-MiniLM-L6-v2` from `sentence-transformers`. When a user asks a question, its embedding is compared to all stored question embeddings via cosine similarity.
3. **Retrieval** – The top‑k most similar questions are found, and their corresponding answers are retrieved.
4. **Generation** – The retrieved context (answers) is fed into **Groq’s Llama 3.3** (cloud LLM) along with the original question to produce a natural, elaborated answer.

---

## ✨ Features

- **Fully local embedding** – The retrieval step runs on your machine (no external API for embeddings).
- **Cloud LLM** – Uses Groq’s fast Llama 3.3 for high‑quality answer generation.
- **Expandable** – Add more rows to `medical_data.csv` – the vector store is rebuilt at runtime.
- **Lightweight** – No ChromaDB, no protobuf conflicts; uses only `sentence-transformers` and scikit‑learn.

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9 – 3.12 (3.13+ may cause issues)
- A Groq API key (get one at [console.groq.com](https://console.groq.com))

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/akshyat21/medical-rag-prototype.git
   cd medical-rag-prototype

2. **create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate      # Linux/Mac
    venv\Scripts\activate         # Windows

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt

4. **Update API key in .env file, GROQ_API_KEY=your_key_here**

5. **Run the streamlit app**
    ```bash
    streamlit run app.py