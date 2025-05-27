# Vector DB storage/retrieval (FAISS/Chroma)
from langchain.vectorstores import FAISS
import os

def create_vector_store(chunks, embedding_model, index_path="data/index"):
    vectordb = FAISS.from_documents(chunks, embedding_model)
    vectordb.save_local(index_path)
    return vectordb

def load_vector_store(embedding_model, index_path="data/index"):
    return FAISS.load_local(index_path, embedding_model)
