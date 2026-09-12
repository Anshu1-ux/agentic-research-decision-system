"""Escalation gate: decides whether a decision needs human review before being final.

Runs after the decision maker. Does not call an LLM — this is a deterministic
policy check, kept separate from the critic (which judges draft quality) so the
escalation threshold can be tuned independently of the critique loop's approval bar.
"""
from src.state import ResearchState

# Tunable policy: what counts as "needs a human to look at this before acting on it."
SCORE_ESCALATION_THRESHOLD = 0.75


def escalation_node(state: ResearchState) -> dict:
    reasons = []

    if not state.approved:
        reasons.append(
            f"Critic never approved the draft after {state.revision_count} revision(s) "
            f"(final score: {state.score:.2f})."
        )

    if state.score < SCORE_ESCALATION_THRESHOLD:
        reasons.append(
            f"Critic score ({state.score:.2f}) is below the "
            f"{SCORE_ESCALATION_THRESHOLD} escalation threshold."
        )

    needs_review = len(reasons) > 0

    return {
        "needs_human_review": needs_review,
        "escalation_reason": " ".join(reasons) if needs_review else "",
    }