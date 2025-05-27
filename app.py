import streamlit as st
from document_loader import load_and_split_documents
from embedding_generator import get_embedding_model
from vector_store import create_vector_store, load_vector_store
from agentic_query import get_qa_chain

st.title("📄 Agentic AI Corporate Chat")
embedding_model = get_embedding_model()

# Load Vector Store
if "vectordb" not in st.session_state:
    st.session_state.vectordb = load_vector_store(embedding_model)

qa_chain = get_qa_chain(st.session_state.vectordb)

# User Input
query = st.text_input("Ask a question about corporate documents:")

if query:
    response = qa_chain.run(query)
    st.write("### Answer")
    st.write(response)
