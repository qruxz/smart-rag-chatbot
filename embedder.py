import os
import faiss
import pickle
from langchain_ollama import OllamaEmbeddings 
from langchain_community.vectorstores import FAISS 

def embed_and_store(documents, db_path="vectordb/db.faiss"): 
    embeddings = OllamaEmbeddings(model="gemma:2b")
    
    vectorstore = FAISS.from_documents(documents, embedding=embeddings) 
    vectorstore.save_local(db_path)

def load_vectorstore(db_path="vectordb/db.faiss"):
    embeddings = OllamaEmbeddings(model="gemma:2b") 
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
