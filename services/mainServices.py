import os
from groq import Groq
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from services.vectorstore import groq_vector_store, local_vector_store
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

class MainServices():
    def __init__(self):
        # self.db = db
        # self.config = config
        # self.logger = logger
        # self.logger.info("mainServices initialized")

        self.pdf_directory = "pdfs/"
        self.template = """
            You are a helpful assistant. Answer the user's question based on the provided context.
            If the question is not related to the context, say "I don't know".
            Context: {context}
            Question: {question}
            Answer:
        """

        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        self.llm = ChatGroq(
            model=os.environ["LLM_MODEL"], # LLM_MODEL='deepseek-r1-distill-llama-70b'
            temperature=0.0,
            max_tokens=None,
            timeout=30
            # max_retries=2,
            # puedes especificar más parámetros según tu caso
        )

        try:
            self.local_llm=OllamaLLM(model=os.environ["LOCAL_LLM_MODEL"])
        except Exception as e:
            print(f"Error initializing local LLM: {e}")
            self.local_llm = None


    def upload_pdf(self, file) -> None:
        try:
            """Upload a PDF file to the vector store."""
            with open(self.pdf_directory + file.name, "wb") as f:
                f.write(file.getbuffer())
            print("PDF uploaded successfully.")
        except Exception as e:
            print(f"Error uploading PDF: {e}")


    def load_pdf(self, filename: str):
        try: return PDFPlumberLoader(self.pdf_directory + filename).load() # return docs as list of Document objects
        except Exception as e:
            print(f"Error uploading PDF: {e}")
        finally:
            print("PDF loaded successfully.")


    def split_text(self, docs):
        """
        Splits a list of documents into smaller chunks for better processing.

        This function uses a RecursiveCharacterTextSplitter to divide the text 
        into smaller chunks of a specified size, with optional overlap between 
        chunks. It also includes the start index of each chunk for reference.

        Args:
            docs (list): A list of documents to be split. Each document is 
                        expected to have a `page_content` attribute containing 
                        the text to be processed.

        Returns:
            list: A list of smaller text chunks, each represented as a dictionary 
                with keys such as 'text', 'start', and 'end'.
        """
        try:
            # Check if docs is empty or None
            if not docs:
                raise ValueError("No documents to split.")

            # Check if each document has the required attributes
            for doc in docs:
                if not hasattr(doc, 'page_content'):
                    raise AttributeError(f"Document {doc} does not have 'page_content' attribute.")
                if not hasattr(doc, 'metadata'):
                    raise AttributeError(f"Document {doc} does not have 'metadata' attribute.")
                
            # Split the text into smaller chunks for better processing
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True
            )

            # Split the documents into smaller chunks # docs = [doc.page_content for doc in docs]
            return text_splitter.split_documents(docs)
        
        except Exception as e:
            print(f"Error uploading PDF: {e}")
        finally:
            print("Text split successfully.")


    def index_documents(self, docs, model_choice): # Index the documents into the vector store
        # for doc in docs: vector_store.add_texts([doc.page_content], [doc.metadata])
        try: 
            if model_choice == 'Local':
                local_vector_store.add_documents(docs)
            else:
                groq_vector_store.add_documents(docs)

        except Exception as e:
            print(f"Error indexing documents: {e}")
        finally:
            print("Documents indexed successfully.")


    def retrieve_documents(self, query, model_choice):
        # Retrieve documents from the vector store based on the query
        try:
            if model_choice == 'Local':
                return local_vector_store.similarity_search(query)
            else:
                return groq_vector_store.similarity_search(query)
        except Exception as e:
            print(f"Error retrieving documents: {e}")
        finally:
            print("Documents retrieved successfully.")


    def answer_question(self, question, docs, model_choice):
        try:
            # Use the model to answer the question based on the context
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = ChatPromptTemplate.from_template(self.template)
            chain = prompt | self.local_llm if model_choice == 'Local' else prompt | self.llm

            return chain.invoke({"question":question, "context": context})
        except Exception as e:
            print(f"Error answering question: {e}")
        finally:
            print("Question answered successfully.")