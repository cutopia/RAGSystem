import os
import chromadb
from langchain_chroma import Chroma
from chromadb.api import ClientAPI
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

class DBWriter:
    static_db: ClientAPI = None
    static_collection_name = "rules_doc"
    static_ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    @staticmethod
    def get_db_location(pdf_filename: str) -> str:
        if pdf_filename is None:
            raise RuntimeError("pdf_filename is None")
        location = "./" + pdf_filename + "_chroma_langchain_db"
        return location

    @staticmethod
    def does_collection_exist_for_pdf(pdf_filename: str) -> bool:
        return os.path.exists(DBWriter.get_db_location(pdf_filename))

    @staticmethod
    def get_collection_for_pdf(filename):
        db = chromadb.PersistentClient(DBWriter.get_db_location(filename))
        return db.get_or_create_collection(DBWriter.static_collection_name)

    @staticmethod
    def get_retriever_for_pdf(filename) -> VectorStoreRetriever:
        return Chroma(
            persist_directory=DBWriter.get_db_location(filename),
            collection_name=DBWriter.static_collection_name,
            embedding_function=DBWriter.static_ollama_embeddings,
        ).as_retriever()
