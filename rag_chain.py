from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

loader = CSVLoader(file_path="medical_data.csv", source_column="question")
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings model loaded.")

vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db")
vector_store.persist()
print("Vector store created and persisted to ./chroma_db.")

def get_relevant_context(question: str, k: int = 3) -> str:
    """Return top k relevant chunks as a single string"""
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

if __name__ == "__main__":
    test_q = "What is the first-line treatment for hypertension?"
    context = get_relevant_context(test_q)
    print("Retrieved context:\n", context)