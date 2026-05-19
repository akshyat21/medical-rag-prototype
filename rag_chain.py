# rag_chain.py
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load CSV with robust quoting – handles commas and quotes in answers
try:
    df = pd.read_csv("medical_data.csv", quoting=csv.QUOTE_ALL)
except Exception as e:
    print(f"Pandas error: {e}. Trying manual fallback...")
    questions = []
    answers = []
    with open("medical_data.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                q = row[0].strip()
                a = row[1].strip()
                if q and a:
                    questions.append(q)
                    answers.append(a)
    df = pd.DataFrame({"question": questions, "answer": answers})

print(f"Loaded {len(df)} Q&A pairs.")

# Load embedding model (downloads once, then caches)
model = SentenceTransformer("all-MiniLM-L6-v2")

questions = df["question"].tolist()
answers = df["answer"].tolist()
question_embeddings = model.encode(questions)

def get_relevant_context(question: str, k: int = 3) -> str:
    """Return top k relevant answers using cosine similarity on embeddings."""
    q_emb = model.encode([question])
    similarities = cosine_similarity(q_emb, question_embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    context = "\n\n".join([answers[i] for i in top_indices])
    return context