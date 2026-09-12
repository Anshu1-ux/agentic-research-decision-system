"""RAG retrieval tool used by the researcher agent."""
from typing import List

from src.rag.vectorstore import get_vectorstore


def retrieve(query: str, k: int = 4) -> List[str]:
    """Return the top-k most relevant chunks from Milvus for this query."""
    try:
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs] or ["No relevant documents found in RAG store."]
    except Exception as exc:  # noqa: BLE001
        return [f"RAG retrieval unavailable ({exc}). Falling back to web search only."]