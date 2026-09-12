"""CLI entrypoint for the Agentic Research & Decision Intelligence System."""
import sys

from src.config import settings
from src.graph import build_graph
from src.state import ResearchState


def run(query: str) -> None:
    graph = build_graph()

    initial_state = ResearchState(query=query, max_iterations=settings.max_iterations)

    print(f"\n=== Running research & decision graph for ===\n{query}\n")

    final_state = graph.invoke(initial_state)

    print("\n=== Research notes ===")
    for note in final_state["research_notes"]:
        print(f"- {note}")

    print(f"\n=== Final draft (revision {final_state['revision_count']}) ===")
    print(final_state["draft"])

    print(f"\n=== Critic verdict (score={final_state['score']}, "
          f"approved={final_state['approved']}) ===")
    print(final_state["critique"])

    print("\n=== Final decision ===")
    print(final_state["decision"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py "your decision question here"')
        sys.exit(1)

    run(" ".join(sys.argv[1:]))