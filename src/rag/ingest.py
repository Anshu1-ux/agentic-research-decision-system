"""Ingest documents into Milvus. Run standalone: python -m src.rag.ingest"""
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.vectorstore import get_vectorstore


def ingest_text_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectorstore = get_vectorstore()
    vectorstore.add_texts(chunks, metadatas=[{"source": path}] * len(chunks))

    print(f"Ingested {len(chunks)} chunks from {path} into Milvus.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag.ingest <path-to-text-file>")
        sys.exit(1)
    ingest_text_file(sys.argv[1])