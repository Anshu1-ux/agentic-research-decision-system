"""Returns the right chat model for a given agent, based on config.Settings."""
from src.config import settings


def get_llm(agent_name: str, temperature: float = 0.3):
    provider = settings.agent_providers.get(agent_name, "openai")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature,
                           api_key=settings.openai_api_key)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=temperature,
                              api_key=settings.anthropic_api_key)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=temperature,
                                       google_api_key=settings.google_api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3.2:3b", temperature=temperature,
                           base_url=settings.ollama_base_url)

    raise ValueError(f"Unknown provider '{provider}' for agent '{agent_name}'")