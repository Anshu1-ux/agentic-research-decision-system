"""Drafter agent: synthesizes research notes (and prior critique, if any) into a draft."""
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_factory import get_llm
from src.state import ResearchState

SYSTEM_PROMPT = """You are a sharp analytical writer producing a draft decision brief. Write a
clear, well-structured brief (options considered, key evidence, trade-offs, risks) based on the
research notes provided. If prior critique is included, address every point raised in the
critique directly in this revision."""


def drafter_node(state: ResearchState) -> dict:
    query = state.query
    notes = state.research_notes
    critique = state.critique
    revision_count = state.revision_count

    llm = get_llm("drafter", temperature=0.4)

    notes_block = "\n".join(f"- {n}" for n in notes)
    prior_critique_block = f"\n\nPrevious critique to address:\n{critique}" if critique else ""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Decision question: {query}\n\nResearch notes:\n{notes_block}"
            f"{prior_critique_block}\n\nWrite the decision brief."
        ),
    ]
    response = llm.invoke(messages)

    return {"draft": response.content, "revision_count": revision_count + 1}