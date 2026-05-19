# app_standalone.py
import streamlit as st
import os
from groq import Groq
from rag_chain import get_relevant_context
from dotenv import load_dotenv

# Load environment variables (for local testing)
load_dotenv()

# Get Groq API key from environment (Streamlit Cloud uses secrets)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Fallback to Streamlit secrets (for cloud deployment)
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        st.error("GROQ_API_KEY not found. Please set it in Streamlit secrets or .env file.")
        st.stop()

client = Groq(api_key=api_key)

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("🏥 Medical RAG Assistant (Live Demo)")
st.markdown("Ask any medical question – the AI will answer based on a curated knowledge base (medical_data.csv).")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    - **RAG (Retrieval-Augmented Generation)** pipeline.
    - Uses ChromaDB + sentence‑transformers for retrieval.
    - Generates answers with **Groq Llama 3.3** (fast, high quality).
    - Context is shown for transparency.
    """)
    st.caption("Prototype for Plato Tech – on‑premise ready (local model can replace Groq).")

# Main input
question = st.text_area("Your question:", height=120,
                        placeholder="e.g., What is the difference between a heart attack and cardiac arrest?")

if st.button("🔍 Get Answer", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant information and generating answer..."):
            try:
                # 1. Retrieve context
                context = get_relevant_context(question)
                if not context.strip():
                    context = "No relevant information found in the knowledge base."

                # 2. Build prompt that asks for elaboration
                augmented_prompt = f"""You are a medical assistant. Use the information below to answer the question. Do NOT copy the text word‑for‑word. Instead, write a short, helpful explanation in your own words. Use bullet points if appropriate.

Relevant medical information:
{context}

Question: {question}

Answer (write clearly, expand slightly, avoid just repeating the context):"""

                # 3. Call Groq
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": augmented_prompt}],
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.choices[0].message.content

                # 4. Display results
                st.success("Answer generated!")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("💡 Answer")
                    st.write(answer)
                with col2:
                    st.metric("Response time", f"{response.usage.total_time:.2f} s" if hasattr(response, 'usage') else "N/A")
                
                with st.expander("📄 Retrieved Context (used to ground the answer)"):
                    st.write(context)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Example questions
with st.expander("💡 Example questions"):
    st.markdown("""
    - What is the first-line treatment for hypertension?
    - How to manage type 2 diabetes?
    - What are the symptoms of a heart attack?
    - What is the difference between a sprain and a strain?
    - How often should women have a Pap smear?
    """)