# Load documents and generate chunks
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(folder_path):
    import os
    docs = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(full_path)
        elif file.endswith(".txt"):
            loader = TextLoader(full_path)
        else:
            continue
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks
