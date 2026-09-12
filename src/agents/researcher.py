"""Researcher agent: combines RAG retrieval (Milvus) with live web search (Tavily)."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch

from src.config import settings
from src.llm_factory import get_llm
from src.rag.retrieval import retrieve
from src.state import ResearchState

SYSTEM_PROMPT = """You are a meticulous research analyst. Given a decision question, and both
internal document excerpts and live web search results, extract the most relevant,
decision-useful facts as a concise bulleted list. Clearly note which facts come from internal
documents vs. the web, and flag any conflicting evidence. Do not make a recommendation yet."""


def researcher_node(state: ResearchState) -> dict:
    query = state.query

    rag_results = retrieve(query, k=4)

    try:
        web_tool = TavilySearch(max_results=5, tavily_api_key=settings.tavily_api_key)
        web_results = [r.get("content", "") for r in web_tool.invoke(query).get("results", [])]
    except Exception as exc:  # noqa: BLE001
        web_results = [f"Web search unavailable ({exc})."]

    llm = get_llm("researcher", temperature=0.2)

    rag_block = "\n".join(f"- {r}" for r in rag_results)
    web_block = "\n".join(f"- {r}" for r in web_results)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Decision question: {query}\n\n"
            f"Internal document excerpts:\n{rag_block}\n\n"
            f"Web search results:\n{web_block}\n\n"
            "Produce a bulleted list of key facts relevant to this decision."
        ),
    ]
    response = llm.invoke(messages)
    notes = [line.strip("- ").strip() for line in response.content.split("\n") if line.strip()]

    return {"research_notes": notes}