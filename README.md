# Agentic Research & Decision Intelligence System

A multi-agent decision-support system built with **LangGraph**. A researcher agent gathers
evidence from both a vector store (RAG) and live web search, a drafter/critic pair iterates
in a bounded critique loop to improve quality, and low-confidence outputs are automatically
escalated to a human review queue rather than being silently trusted.

This project pairs with a document-intelligence/approval-workflow system as a second,
architecturally distinct pattern: **agent self-critique + automated escalation**, instead of
manual human-in-the-loop approval at every step.

## Architecture
    ┌─────────────┐
    │  Researcher │  RAG (Milvus) + live web search (Tavily)
    └──────┬──────┘
           │
    ┌──────▼──────┐

┌───►│ Drafter │ synthesizes findings into a decision brief
│ └──────┬──────┘
│ │
│ ┌──────▼──────┐
│ │ Critic │ scores + critiques the draft (structured JSON)
│ └──────┬──────┘
│ │
│ approved? ──No──┐
│ │
│ (revise, bounded by max_iterations)
└─────────────────────┘
│ Yes / max iterations reached
┌──────▼──────┐
│ Decision │ final structured recommendation
│ Maker │
└──────┬──────┘
│
┌──────▼──────┐
│ Escalation │ deterministic policy gate (no LLM call)
│ Gate │ flags low-confidence / unapproved decisions
└──────┬──────┘
│
needs review? ──Yes──► SQLite review queue ──► human approves/rejects
│
END


## Stack

- **Orchestration:** LangGraph, LangChain
- **LLMs:** Provider-agnostic per agent (OpenAI / Anthropic / Google / **Ollama**, currently
  running fully local on `llama3.2:3b` for zero API cost)
- **RAG:** Milvus (vector store), Ollama embeddings (`nomic-embed-text`)
- **Web search:** Tavily
- **Backend:** FastAPI
- **UI:** Streamlit (analysis view + human review queue)
- **Persistence:** SQLAlchemy + SQLite (review queue)
- **Evaluation:** Ragas (faithfulness, context precision/recall), LangSmith (tracing)
- **Deployment:** Docker Compose (Milvus + API + UI containerized; Ollama runs natively on
  the host and is reached via `host.docker.internal`, since Ollama needs GPU/Metal access
  that Docker on macOS can't pass through)

## Project layout

├── src/
│ ├── agents/ # researcher, drafter, critic, decision_maker, escalation
│ ├── rag/ # Milvus vectorstore, ingestion, retrieval
│ ├── eval/ # Ragas evaluation harness
│ ├── tools/ # search wrapper
│ ├── config.py # env settings + per-agent provider assignment
│ ├── llm_factory.py # returns the right chat model per agent
│ ├── state.py # Pydantic graph state
│ ├── graph.py # LangGraph wiring + critique-loop routing
│ ├── db.py / models.py # SQLAlchemy review queue
│ └── api.py # FastAPI: /research, /reviews/*
├── streamlit_app.py # UI: run analysis + review queue
├── main.py # CLI entrypoint
├── docker-compose.yml # Milvus + API + Streamlit
├── Dockerfile.api / Dockerfile.streamlit
└── sample_docs/ # documents ingested into Milvus for RAG


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in your keys
```

Required in `.env`: at minimum `TAVILY_API_KEY` (web search) and either a cloud LLM key or
a running local [Ollama](https://ollama.com) instance (`ollama pull llama3.2:3b && ollama pull nomic-embed-text`).

Start Milvus:
```bash
docker compose up -d etcd minio standalone
```

Run the CLI:
```bash
python3 main.py "Should we migrate our data warehouse to a lakehouse architecture?"
```

Or run the full stack (API + UI + Milvus, Dockerized):
```bash
docker compose up --build
```
Then open `http://localhost:8501`.

## Evaluation results

Ragas evaluation over 6 questions spanning 4 ingested documents:

| Metric | Score |
|---|---|
| Faithfulness | 0.975 |
| Context Recall | 0.976 |
| Context Precision | N/A* |

\* *`context_precision` consistently fails to parse with the local `llama3.2:3b` judge model
— it reliably struggles with this specific structured-JSON verdict prompt, even when its
underlying reasoning in prose is correct. Documented as a known limitation of small local
judge models rather than worked around, since swapping the judge to a hosted model would
resolve it directly.*

**Notable finding:** an early eval run surfaced `faithfulness: 0.0` — the researcher agent
was mislabeling web-search results as "internal document excerpts" because the Milvus
collection was effectively empty (only a placeholder sentence). This is exactly the kind of
attribution failure that's easy to miss on a casual read of the output but shows up
immediately under evaluation. Fixing the underlying RAG data (ingesting real documents)
resolved it, taking faithfulness from 0.0 → 0.975.

## Design notes

- **Bounded critique loop:** `max_iterations` guarantees the drafter/critic loop always
  terminates, even if the critic never approves.
- **Deterministic escalation, not another LLM judgment:** the escalation gate is a plain
  Python policy check (score threshold + approval status), not an LLM call — kept separate
  from the critic so the escalation bar can be tuned independently of critique quality, and
  so escalation is fast, free, and fully predictable.
- **Fail-safe JSON parsing:** the critic's structured output parsing tries strict JSON, then
  sanitized JSON (local models sometimes emit unescaped newlines inside string values), then
  targeted regex extraction — and never silently treats unparseable output as approval.
- **Swappable everything:** search tool, vector store, and LLM provider are all isolated
  behind thin interfaces, so any one can be swapped without touching agent logic.

## Known limitations

- Local 3B-parameter models are noticeably less reliable than hosted models (GPT-4o, Claude)
  at both structured JSON output and multi-turn self-correction — the critique loop
  sometimes repeats the same critique across revisions without the drafter fully resolving it.
- `context_precision` (Ragas) does not reliably evaluate with the local judge model (see above).
- Ollama cannot be containerized alongside the rest of the stack on macOS without losing GPU
  acceleration, so it runs natively rather than in Docker.