"""Builds the LangGraph StateGraph: researcher -> drafter <-> critic -> decision_maker -> escalation."""
from langgraph.graph import END, StateGraph

from src.agents.critic import critic_node
from src.agents.decision_maker import decision_maker_node
from src.agents.drafter import drafter_node
from src.agents.escalation import escalation_node
from src.agents.researcher import researcher_node
from src.state import ResearchState


def _route_after_critic(state: ResearchState) -> str:
    print(f"[DEBUG] revision_count={state.revision_count}, "
          f"max_iterations={state.max_iterations}, approved={state.approved}")
    if state.approved:
        return "decision_maker"
    if state.revision_count >= state.max_iterations:
        return "decision_maker"
    return "drafter"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("critic", critic_node)
    graph.add_node("decision_maker", decision_maker_node)
    graph.add_node("escalation", escalation_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "drafter")
    graph.add_edge("drafter", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"drafter": "drafter", "decision_maker": "decision_maker"},
    )
    graph.add_edge("decision_maker", "escalation")
    graph.add_edge("escalation", END)

    return graph.compile()