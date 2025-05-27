# Generate embeddings (TF-IDF/BERT/OpenAI)
from langchain.embeddings import HuggingFaceEmbeddings
# Or OpenAIEmbeddings if you use OpenAI

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
