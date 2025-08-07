import sys
import os
import chromadb
from langchain_chroma import Chroma
from typing import Optional
from chromadb.api import ClientAPI
from chromadb.api.models import Collection
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

class DBWriter:
    static_idx = 1
    static_collection: Optional[Collection] = None
    static_db: ClientAPI = None
    static_collection_name = "rules_doc"
    static_ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    @staticmethod
    def get_db_location() -> str:
        rules_file_name = sys.argv[1]
        location = "./" + rules_file_name + "_chroma_langchain_db"
        print(location)
        return location

    @staticmethod
    def setup_db():
        print("Setting up DB")
        DBWriter.static_db = chromadb.PersistentClient(DBWriter.get_db_location())
        DBWriter.static_collection = DBWriter.static_db.get_or_create_collection(DBWriter.static_collection_name)

    @staticmethod
    def db_needs_generated() -> bool:
        return not os.path.exists(DBWriter.get_db_location())

    @staticmethod
    def get_collection() -> Collection:
        if DBWriter.static_collection is None:
            DBWriter.setup_db()
        return DBWriter.static_collection

    @staticmethod
    def get_retriever() -> VectorStoreRetriever:
        return Chroma(
            persist_directory=DBWriter.get_db_location(),
            collection_name=DBWriter.static_collection_name,
            embedding_function=DBWriter.static_ollama_embeddings,
        ).as_retriever()

    @staticmethod
    def get_next_index() -> int:
        retval = DBWriter.static_idx
        DBWriter.static_idx += 1
        return retval
