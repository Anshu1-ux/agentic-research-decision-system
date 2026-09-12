"""Centralized configuration: env vars + per-agent provider assignment."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    google_api_key: str
    anthropic_api_key: str
    tavily_api_key: str
    milvus_host: str
    milvus_port: str
    ollama_base_url: str
    max_iterations: int

    # Which provider backs each agent. Change these to mix and match.
    agent_providers: dict = field(default_factory=lambda: {
        "researcher": "ollama",
        "drafter": "ollama",
        "critic": "ollama",
        "decision_maker": "ollama",
    })

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            milvus_host=os.getenv("MILVUS_HOST", "localhost"),
            milvus_port=os.getenv("MILVUS_PORT", "19530"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        )


settings = Settings.load()