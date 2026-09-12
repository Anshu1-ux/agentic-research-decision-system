"""Critic agent: scores the draft and decides whether it's ready to move forward.

Local models (via Ollama) are less reliable at strict JSON-only output than hosted
models, so parsing here is defensive: try clean JSON first, fall back to sanitized
JSON, then targeted regex extraction, and fail safe (never silently approve) if all fail.
"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_factory import get_llm
from src.state import ResearchState

SYSTEM_PROMPT = """You are a rigorous, skeptical editor reviewing a decision brief. Evaluate it
on: factual grounding in the provided research notes, logical coherence, whether trade-offs and
risks are addressed, and clarity.

Respond ONLY with a JSON object, nothing else before or after it, in this exact shape:
{"score": 0.0, "approved": false, "critique": "specific, actionable feedback"}

Approve (approved: true) only if score >= 0.8. Be specific in critique — name what's missing or
wrong, don't just say "needs improvement". Do not wrap the JSON in markdown code fences."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_critic_output(raw: str) -> dict:
    cleaned = raw.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(raw)
    if match:
        block = match.group(0)
        sanitized = re.sub(r'(?<!\\)\n', '\\n', block)
        sanitized = re.sub(r'(?<!\\)\t', '\\t', sanitized)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

    score_match = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
    approved_match = re.search(r'"approved"\s*:\s*(true|false)', raw, re.IGNORECASE)
    critique_match = re.search(r'"critique"\s*:\s*"(.*)', raw, re.DOTALL)
    if score_match and approved_match:
        critique_text = critique_match.group(1).rstrip() if critique_match else ""
        critique_text = re.sub(r'"\s*\}\s*$', '', critique_text)
        return {
            "score": float(score_match.group(1)),
            "approved": approved_match.group(1).lower() == "true",
            "critique": critique_text,
        }

    return {}


def critic_node(state: ResearchState) -> dict:
    query = state.query
    draft = state.draft
    notes = state.research_notes

    llm = get_llm("critic", temperature=0.0)

    notes_block = "\n".join(f"- {n}" for n in notes)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Decision question: {query}\n\nResearch notes:\n{notes_block}\n\n"
            f"Draft brief:\n{draft}\n\nEvaluate this draft."
        ),
    ]
    response = llm.invoke(messages)
    parsed = _parse_critic_output(response.content)

    if not parsed:
        return {
            "score": 0.0,
            "approved": False,
            "critique": f"Critic returned unparseable output, treating as not approved. "
                        f"Raw output: {response.content[:300]}",
        }

    try:
        score = float(parsed.get("score", 0.0))
        approved = bool(parsed.get("approved", False))
        critique = str(parsed.get("critique", ""))
    except (TypeError, ValueError):
        score, approved, critique = 0.0, False, "Critic output had malformed fields."

    return {"score": score, "approved": approved, "critique": critique}