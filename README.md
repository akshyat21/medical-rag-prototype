# Medical RAG Prototype

A **Retrieval-Augmented Generation (RAG)** system that answers medical questions using a curated knowledge base.  
It combines vector search (ChromaDB + sentence‑transformers) with a large language model (Groq Llama 3.3) to produce accurate, context‑grounded answers.

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-rag-prototype-mvmcoxoco4mjnjjmi8mvsq.streamlit.app/)

**Live Demo:** [https://medical-rag-prototype-mvmcoxoco4mjnjjmi8mvsq.streamlit.app/](https://medical-rag-prototype-mvmcoxoco4mjnjjmi8mvsq.streamlit.app/)

---

## 🧠 How It Works

1. **Knowledge Base** – A CSV file (`medical_data.csv`) containing medical Q&A pairs.
2. **Embeddings & Vector Store** – Each entry is split into chunks, converted into vector embeddings (using `all-MiniLM-L6-v2`), and stored in ChromaDB.
3. **Retrieval** – When a user asks a question, the system finds the most relevant chunks via similarity search.
4. **Generation** – The retrieved context is fed into **Groq’s Llama 3.3** (cloud LLM) to produce a natural, elaborated answer.  

---

## ✨ Features

- **Live Demo** – Deployed on Streamlit Cloud (uses Groq API key).
- **RAG Pipeline** – Answers are grounded in your own medical data, reducing hallucinations.
- **Expandable** – Add more rows to `medical_data.csv` and rebuild the vector store.
- **Optional Local LLM** – The code also supports running TinyLlama 4‑bit on CPU (fully on‑premise).

---

## 📁 Repository Structure
