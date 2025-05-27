# Answer generation logic (LangChain QA chain)
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def get_qa_chain(vectordb):
    llm = ChatOpenAI(model="gpt-4", temperature=0)  # Replace with LocalAI endpoint if needed
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
    return qa_chain
