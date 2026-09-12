"""Decision maker agent: turns an approved draft into a final, structured recommendation."""
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_factory import get_llm
from src.state import ResearchState

SYSTEM_PROMPT = """You are a decisive advisor. Given a decision brief, produce a final
recommendation with this structure:

RECOMMENDATION: <one sentence>
CONFIDENCE: <High|Medium|Low>
KEY REASONS:
- <reason 1>
- <reason 2>
- <reason 3>
RISKS TO MONITOR:
- <risk 1>
- <risk 2>

Be concrete and decisive. Do not hedge unnecessarily. If the brief was never fully approved by
review, set CONFIDENCE no higher than Medium and say so explicitly in KEY REASONS."""


def decision_maker_node(state: ResearchState) -> dict:
    query = state.query
    draft = state.draft
    approved = state.approved

    llm = get_llm("decision_maker", temperature=0.2)

    approval_note = (
        "This brief passed critical review." if approved
        else "This brief did NOT pass critical review (max revisions reached) — "
             "flag this clearly and lower confidence accordingly."
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Decision question: {query}\n\nBrief:\n{draft}\n\n{approval_note}"
        ),
    ]
    response = llm.invoke(messages)

    return {"decision": response.content}