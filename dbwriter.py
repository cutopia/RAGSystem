import os
import chromadb
from langchain_chroma import Chroma
from typing import Optional
from chromadb.api import ClientAPI
from chromadb.api.models import Collection
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

class DBWriter:
    static_collection: Optional[Collection] = None
    static_db: ClientAPI = None
    static_collection_name = "rules_doc"
    static_ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    static_pdf_filename:str = None

    @staticmethod
    def is_pdf_assigned():
        return DBWriter.static_pdf_filename is not None

    @staticmethod
    def get_db_location() -> str:
        if DBWriter.static_pdf_filename is None:
            raise RuntimeError("DBWriter.static_pdf_filename is None")
        location = "./" + DBWriter.static_pdf_filename + "_chroma_langchain_db"
        print(location)
        return location

    @staticmethod
    def setup_db(filename):
        print(f"Setting up store for {filename}")
        DBWriter.static_pdf_filename = filename
        DBWriter.static_db = chromadb.PersistentClient(DBWriter.get_db_location())
        DBWriter.static_collection = DBWriter.static_db.get_or_create_collection(DBWriter.static_collection_name)

    @staticmethod
    def db_needs_generated(filename) -> bool:
        if DBWriter.static_pdf_filename != filename:
            return True
        return not os.path.exists(DBWriter.get_db_location())

    @staticmethod
    def get_collection(pdf_filename) -> Optional[Collection]:
        if DBWriter.static_collection is None or DBWriter.static_pdf_filename != pdf_filename:
            DBWriter.setup_db(pdf_filename)
        return DBWriter.static_collection

    @staticmethod
    def get_retriever() -> VectorStoreRetriever:
        return Chroma(
            persist_directory=DBWriter.get_db_location(),
            collection_name=DBWriter.static_collection_name,
            embedding_function=DBWriter.static_ollama_embeddings,
        ).as_retriever()
