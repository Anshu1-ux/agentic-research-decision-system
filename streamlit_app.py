"""Streamlit demo UI for the Agentic Research & Decision Intelligence System.

Calls the FastAPI backend (must be running separately: uvicorn src.api:app --port 8000).
"""
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Agentic Research & Decision System", layout="wide")

page = st.sidebar.radio("Navigate", ["Run Analysis", "Review Queue"])

# ---------------------------------------------------------------------------
# PAGE 1: Run Analysis
# ---------------------------------------------------------------------------
if page == "Run Analysis":
    st.title("🔍 Agentic Research & Decision Intelligence System")
    st.caption(
        "Multi-agent pipeline: Researcher (RAG + web search) → Drafter → Critic "
        "(critique loop) → Decision Maker → Escalation Gate"
    )

    query = st.text_area(
        "Decision question",
        placeholder="e.g. Should we migrate our data warehouse to a lakehouse architecture?",
        height=80,
    )

    if st.button("Run analysis", type="primary", disabled=not query.strip()):
        with st.spinner("Running researcher → drafter → critic loop → decision maker... "
                         "(can take several minutes on local models)"):
            try:
                response = requests.post(f"{API_BASE}/research", json={"query": query}, timeout=600)
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is it running? `uvicorn src.api:app --port 8000`")
                st.stop()
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        st.success("Done!")

        if result["needs_human_review"]:
            st.warning(
                f"⚠️ **Escalated for human review** (queue item #{result['review_id']}). "
                f"{result['escalation_reason']} Go to the **Review Queue** tab to act on it."
            )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📋 Research Notes")
            for note in result["research_notes"]:
                st.markdown(f"- {note}")

        with col2:
            verdict_icon = "✅" if result["approved"] else "⚠️"
            st.subheader(f"{verdict_icon} Critic Verdict")
            verdict_color = "green" if result["approved"] else "orange"
            st.markdown(
                f"**Score:** {result['score']:.2f} &nbsp;|&nbsp; "
                f":{verdict_color}[**{'Approved' if result['approved'] else 'Not approved'}**] "
                f"&nbsp;|&nbsp; **Revisions:** {result['revision_count']}"
            )
            st.markdown(result["critique"])

        st.divider()

        st.subheader("📝 Final Draft")
        with st.expander("Show full draft brief", expanded=False):
            st.markdown(result["draft"])

        st.divider()

        st.subheader("🎯 Final Decision")
        st.markdown(result["decision"])

# ---------------------------------------------------------------------------
# PAGE 2: Review Queue
# ---------------------------------------------------------------------------
else:
    st.title("🧑‍⚖️ Human Review Queue")
    st.caption("Decisions escalated because the critic never approved them, or the score fell below threshold.")

    status_filter = st.selectbox("Show", ["pending", "approved", "rejected"], index=0)

    try:
        resp = requests.get(f"{API_BASE}/reviews", params={"status": status_filter}, timeout=30)
        resp.raise_for_status()
        items = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Can't reach the API. Is it running? `uvicorn src.api:app --port 8000`")
        st.stop()

    if not items:
        st.info(f"No {status_filter} items.")
        st.stop()

    for item in items:
        with st.container(border=True):
            st.markdown(f"**#{item['id']} — {item['query']}**")
            st.markdown(f"Score: `{item['score']:.2f}` &nbsp;|&nbsp; Created: {item['created_at']}")
            st.markdown(f"*{item['escalation_reason']}*")

            if st.button("View details", key=f"view_{item['id']}"):
                detail_resp = requests.get(f"{API_BASE}/reviews/{item['id']}", timeout=30)
                detail = detail_resp.json()
                with st.expander("Draft", expanded=True):
                    st.markdown(detail["draft"])
                with st.expander("Critique"):
                    st.markdown(detail["critique"])
                with st.expander("Decision"):
                    st.markdown(detail["decision"])
                if detail.get("reviewer_notes"):
                    st.markdown(f"**Reviewer notes:** {detail['reviewer_notes']}")

            if status_filter == "pending":
                notes = st.text_input("Notes (optional)", key=f"notes_{item['id']}")
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    if st.button("✅ Approve", key=f"approve_{item['id']}", type="primary"):
                        requests.post(
                            f"{API_BASE}/reviews/{item['id']}/approve",
                            json={"notes": notes},
                            timeout=30,
                        )
                        st.rerun()
                with bcol2:
                    if st.button("❌ Reject", key=f"reject_{item['id']}"):
                        requests.post(
                            f"{API_BASE}/reviews/{item['id']}/reject",
                            json={"notes": notes},
                            timeout=30,
                        )
                        st.rerun()