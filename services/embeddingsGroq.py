import os
from groq import Groq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

class GroqEmbeddings():
    # def __init__(self, model: str = "nomic-embed-text", api_key: str = None):
    def __init__(self, model: str = os.environ["EMBEDDING_MODEL"], api_key: str = None):
        self.model = model
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding
