import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV
df = pd.read_csv("medical_data.csv")
questions = df["question"].tolist()
answers = df["answer"].tolist()

# Create TF‑IDF vectorizer and fit on all questions
vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

def get_relevant_context(user_question: str, k: int = 3) -> str:
    """Return top k relevant answers using TF‑IDF similarity."""
    # Vectorize the user question
    user_vec = vectorizer.transform([user_question])
    # Compute cosine similarities with all stored questions
    similarities = cosine_similarity(user_vec, question_vectors).flatten()
    # Get indices of top k similarities
    top_indices = np.argsort(similarities)[-k:][::-1]
    # Join the corresponding answers
    context = "\n\n".join([answers[i] for i in top_indices])
    return context