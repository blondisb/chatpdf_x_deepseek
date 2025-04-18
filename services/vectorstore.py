import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from services.embeddingsGroq import GroqEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Instancia del vector store
groq_vector_store = InMemoryVectorStore(
    embedding=GroqEmbeddings(model=os.environ["EMBEDDING_MODEL"])
)

local_vector_store = InMemoryVectorStore(
    embedding=OllamaEmbeddings(model=os.environ["LOCAL_LLM_MODEL"])
)