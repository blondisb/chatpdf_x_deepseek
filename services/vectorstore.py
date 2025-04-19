import os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from services.embeddingsGroq import GroqEmbeddings
# from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv, find_dotenv
# from transformers import AutoModel
# from langchain.embeddings import OpenAIEmbeddings

load_dotenv(find_dotenv(), override=True)

# embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
# embedding_model = OpenAIEmbeddings()

# Instancia del vector store
groq_vector_store = InMemoryVectorStore(
    # embedding=GroqEmbeddings(model=os.environ["EMBEDDING_MODEL"])
    embedding=OllamaEmbeddings(model=os.environ["EMBEDDING_MODEL"])
    # embedding=embedding_model
)

local_vector_store = InMemoryVectorStore(
    embedding=OllamaEmbeddings(model=os.environ["LOCAL_LLM_MODEL"])
)