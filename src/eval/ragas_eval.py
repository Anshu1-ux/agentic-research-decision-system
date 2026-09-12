"""Evaluate RAG retrieval + generation quality using Ragas.

Run standalone: python -m src.eval.ragas_eval
This does NOT touch the live LangGraph agents — it's an offline eval harness
you run against a fixed set of test questions to track RAG quality over time.
"""
from datasets import Dataset
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.run_config import RunConfig

from src.rag.retrieval import retrieve

# A small fixed eval set: question + ground-truth answer.
# In a real project this would come from a labeled dataset, not be hand-written.
EVAL_QUESTIONS = [
    {
        "question": "What is a data lakehouse?",
        "ground_truth": "A data lakehouse combines the scalability and low cost of a "
                         "data lake with the structure and reliability of a data warehouse.",
    },
    {
        "question": "When should an organization avoid migrating to a lakehouse?",
        "ground_truth": "Organizations with a small, stable data model, a team without "
                         "Spark or distributed-systems experience, heavy investment in "
                         "warehouse-native features, or no serious ML workloads should "
                         "generally stay on a traditional warehouse.",
    },
    {
        "question": "How does lakehouse pricing compare to traditional data warehouse pricing?",
        "ground_truth": "Lakehouses decouple storage and compute, which can reduce total "
                         "cost of ownership by 30-50% for read-heavy, irregular workloads, "
                         "though traditional warehouses can be cheaper for small, "
                         "consistently-loaded workloads due to lower operational overhead.",
    },
    {
        "question": "What is the Medallion Architecture?",
        "ground_truth": "The Medallion Architecture organizes data into three layers: "
                         "Bronze (raw ingested data), Silver (cleaned and conformed data), "
                         "and Gold (business-level aggregates ready for BI tools).",
    },
    {
        "question": "What is the safest way to migrate from a warehouse to a lakehouse?",
        "ground_truth": "Run the lakehouse and legacy warehouse in parallel during "
                         "migration, validating that outputs match before cutting over, "
                         "rather than attempting a full cutover in one step.",
    },
    {
        "question": "What is the most common cause of failed lakehouse migrations?",
        "ground_truth": "Attempting a full cutover in one step, since undiscovered edge "
                         "cases in legacy ETL logic often only surface once real users "
                         "depend on the new system.",
    },
]


def build_eval_dataset() -> Dataset:
    rows = {"question": [], "contexts": [], "answer": [], "ground_truth": []}

    for item in EVAL_QUESTIONS:
        contexts = retrieve(item["question"], k=4)
        # For this eval we use the retrieved contexts themselves as the "answer"
        # being judged, since we're isolating retrieval quality here rather than
        # re-running the full drafter/critic pipeline for every eval item.
        answer = " ".join(contexts)

        rows["question"].append(item["question"])
        rows["contexts"].append(contexts)
        rows["answer"].append(answer)
        rows["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(rows)


def run_eval() -> None:
    dataset = build_eval_dataset()

    # Ragas needs an LLM to act as the judge for metrics like faithfulness.
    # Using the same local Ollama model keeps this eval free to run repeatedly.
    judge_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2:3b", temperature=0.0))

    # max_workers=1 forces sequential evaluation — a local CPU model can't
    # keep up with Ragas's default concurrent judge calls and times out.
    results = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=judge_llm,
        run_config=RunConfig(max_workers=1, timeout=120),
    )

    print("\n=== Ragas RAG Evaluation ===")
    print(results)

    df = results.to_pandas()
    print("\nColumns available:", list(df.columns))
    print("\nPer-question breakdown:")
    print(df[["user_input", "faithfulness", "context_precision", "context_recall"]].to_string())


if __name__ == "__main__":
    run_eval()