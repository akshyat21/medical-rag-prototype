# app.py
# Streamlit dashboard for Medical RAG Assistant – supports both Groq (cloud) and TinyLlama (local CPU)

import streamlit as st
import requests
import time

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("🏥 Medical RAG Assistant")
st.markdown("Ask a medical question, and the AI will answer based on your provided knowledge base (medical_data.csv).")

# Sidebar: About & Backend Selection
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    - **RAG (Retrieval-Augmented Generation)** pipeline.
    - Retrieves relevant context from `medical_data.csv` using ChromaDB embeddings.
    - Two backend options:
        - **Groq (cloud)** – fast, high quality (requires API key)
        - **Local TinyLlama (CPU)** – fully on‑premise, slower but private
    - Context is shown for transparency.
    """)
    st.markdown("---")
    
    backend = st.radio(
        "Select backend:",
        ["Groq (cloud, fast)", "Local TinyLlama (CPU, on‑premise)"],
        index=0
    )
    st.caption("Prototype for Plato Tech – on‑premise medical Q&A")

# Determine endpoint URL
if backend == "Groq (cloud, fast)":
    endpoint_url = "http://localhost:8000/v1/rag"
else:
    endpoint_url = "http://localhost:8000/v1/local_rag"

# Main input
question = st.text_area("Your question:", height=100,
                        placeholder="e.g., What is the first-line treatment for hypertension?")

if st.button("🔍 Get Answer", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                start_time = time.time()
                response = requests.post(
                    endpoint_url,
                    json={"question": question, "max_tokens": 300},
                    timeout=60   # local model may take 20-30 seconds
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("Answer generated!")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader("💡 Answer")
                        st.write(data["answer"])
                    with col2:
                        st.metric("Response time", f"{elapsed:.2f} s")
                    
                    with st.expander("📄 Retrieved Context (used to ground the answer)"):
                        st.write(data["context_used"])
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to the RAG server. Please start `server.py` first (python server.py).")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The local model may be too slow on this machine. Try the Groq backend.")

# Example questions expander
with st.expander("💡 Example questions"):
    st.markdown("""
    - What is the first-line treatment for hypertension?
    - How to manage type 2 diabetes?
    - What is the normal range for blood pressure?
    - What are symptoms of myocardial infarction?
    - What vaccine is given for pneumonia?
    """)

# Optional: show backend status
st.sidebar.markdown("---")
st.sidebar.info(f"🔌 Current backend: {backend}\nEndpoint: {endpoint_url}")