import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from groq import Groq
from rag_chain import get_relevant_context
from local_model import generate_local

# loading API key from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize Groq client
client = Groq(api_key=api_key)

app = FastAPI(title="Local Groq LLM Wrapper")

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 0.7

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user", "content": request.prompt}],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    answer = response.choices[0].message.content
    return {"choices": [{"text": answer}]}

class RAGRequest(BaseModel):
    question: str
    max_tokens: int = 200
    temperature: float = 0.7

@app.post("/v1/rag")
async def rag(request: RAGRequest):
    context = get_relevant_context(request.question)
    augmented_prompt = f"""You are a helpful medical assistant. Use the following information to answer the question. Do NOT copy the text word‑for‑word. Instead, write a clear, concise explanation in 2-4 bullet points or a short paragraph. If the information is insufficient, say so.

context:
{context}

Question: {request.question}
Answer:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user", "content": augmented_prompt}],
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    answer = response.choices[0].message.content
    return {"answer":answer, "context_used":context}

@app.post("/v1/local_rag")
async def local_rag(request: RAGRequest):
    context = get_relevant_context(request.question)
    augmented_prompt = f"""Answer the question based on the context below. Write a short explanation in your own words, not just repeating the context.
context:
{context}

Question: {request.question}
Answer:"""
    
    answer = generate_local(augmented_prompt, max_new_tokens=request.max_tokens, temperature=request.temperature)
    return {"answer": answer, "context_used": context}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)