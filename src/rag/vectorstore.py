"""Milvus connection + embedding setup, shared by ingestion and retrieval."""
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus

from src.config import settings

COLLECTION_NAME = "research_documents"


def get_vectorstore() -> Milvus:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=settings.ollama_base_url)
    return Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": settings.milvus_host, "port": settings.milvus_port},
        auto_id=True,
    )