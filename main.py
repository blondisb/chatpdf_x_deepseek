import os
import logging
from groq import Groq
from dotenv import load_dotenv, find_dotenv
# from langchain.embeddings import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader


logging.getLogger("pdfplumber").setLevel(logging.ERROR)
load_dotenv(find_dotenv(), override=True)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

template = """
    You are a helpful assistant. Answer the user's question based on the provided context.
    If the question is not related to the context, say "I don't know".
    Context: {context}
    Question: {question}
    Answer:
"""



def basic_test() -> str:
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": "How many r's are in the word strawberry?"
            }
        ],
        temperature=0.6,
        max_completion_tokens=1024,
        top_p=0.95,
        stream=True,
        reasoning_format="raw"
    )

    for chunk in completion:
        print(chunk.choices[0].delta.content or "", end="")


pdf_directory = "pdfs/"
embeddings = OllamaEmbeddings(model="deepseek-r1-distill-llama-70b")
vector_store = InMemoryVectorStore(embeddings)
# print(dir(OllamaLLM))
model=OllamaLLM(model="deepseek-r1-distill-llama-70b")



def upload_pdf(file) -> None:
    """Upload a PDF file to the vector store."""
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path: str):
    return PDFPlumberLoader(file_path).load() # return docs as list of Document objects

def split_text(docs):
    # Split the text into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    # Split the documents into smaller chunks # docs = [doc.page_content for doc in docs]
    return text_splitter.split_documents(docs)

def index_documents(docs):
    # Index the documents into the vector store
    # for doc in docs:
    #     vector_store.add_texts([doc.page_content], [doc.metadata])
    vector_store.add_documents(docs)

def retrieve_documents(query):
    # Retrieve documents from the vector store based on the query
    return vector_store.similarity_search(query)



if __name__ == "__main__":
    # basic_test()
    pdf_name = "ecografia.pdf"
    pdf_name = "gestion_publica.pdf"

    # upload_pdf(pdf_name)

    docs = load_pdf(pdf_directory + pdf_name)
    split_docs = split_text(docs)
    
    print(len(split_docs))
    # print(split_docs)
    # pass
