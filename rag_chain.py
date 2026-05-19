import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV
df = pd.read_csv("medical_data.csv")
questions = df["question"].tolist()
answers = df["answer"].tolist()

# Load embedding model (same as before, but using transformers)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_texts(texts):
    """Generate embeddings for a list of texts."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling to get sentence embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# Pre‑compute embeddings for all questions (runs once at startup)
question_embeddings = embed_texts(questions)

def get_relevant_context(question: str, k: int = 3) -> str:
    """Return top k relevant answers as a single string."""
    q_emb = embed_texts([question])
    similarities = cosine_similarity(q_emb, question_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    context = "\n\n".join([answers[i] for i in top_indices])
    return context