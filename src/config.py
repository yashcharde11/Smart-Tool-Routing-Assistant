"""Configuration and the Azure OpenAI LLM factory.

Credentials are read from Streamlit secrets when running inside the app, and
fall back to environment variables so the evaluation scripts and unit tests can
run outside Streamlit. Nothing here is hard-coded — see
`.streamlit/secrets.toml.example` for the expected keys.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Load a local .env into the environment if present. This is a no-op in
# production (Streamlit Cloud / HF Spaces) where secrets come from the platform,
# and never overrides values already set in the real environment.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # python-dotenv optional at runtime
    pass

# --- RAG knobs (tune these without touching the graph) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, runs locally
CHUNK_SIZE = 800                        # characters per chunk
CHUNK_OVERLAP = 120                     # character overlap between chunks
RETRIEVE_TOP_K = 4                      # chunks pulled per DOC_SEARCH query

# Where Chroma persists its index. Kept small so it fits free hosting tiers.
CHROMA_DIR = os.environ.get("CHROMA_DIR", ".chroma")
CHROMA_COLLECTION = "uploaded_docs"

# Valid routing labels. Single source of truth used by router, graph and eval.
TOOLS = ("MATH", "DOC_SEARCH", "GENERAL")
DEFAULT_TOOL = "GENERAL"  # safe fallback on router parse failure


def _get_secret(*names: str, default: str | None = None) -> str | None:
    """Read a setting from Streamlit secrets first, then the environment.

    Accepts multiple candidate key names (aliases) and returns the first that
    resolves, so both ``AZURE_OPENAI_DEPLOYMENT`` and the SDK-style
    ``AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`` are accepted.
    """
    # Streamlit secrets (used in cloud deploys).
    try:
        import streamlit as st

        for name in names:
            if name in st.secrets:
                return str(st.secrets[name])
    except Exception:
        pass
    # Environment / .env.
    for name in names:
        val = os.environ.get(name)
        if val:
            return val
    return default


def azure_settings() -> dict[str, str]:
    """Collect the four Azure OpenAI settings, raising a clear error if absent."""
    settings = {
        "api_key": _get_secret("AZURE_OPENAI_API_KEY"),
        "endpoint": _get_secret("AZURE_OPENAI_ENDPOINT"),
        "api_version": _get_secret("AZURE_OPENAI_API_VERSION", default="2024-10-21"),
        "deployment": _get_secret(
            "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
            "AZURE_OPENAI_DEPLOYMENT",
            default="gpt-4.1",
        ),
    }
    missing = [k for k, v in settings.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing Azure OpenAI settings: "
            + ", ".join(missing)
            + ". Set them in .streamlit/secrets.toml or as environment variables "
            "(see .streamlit/secrets.toml.example)."
        )
    return settings  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0):
    """Return a cached AzureChatOpenAI client (GPT-4.1).

    Temperature defaults to 0 for deterministic routing. The same client is
    reused for every node; LangChain handles connection pooling.
    """
    from langchain_openai import AzureChatOpenAI

    s = azure_settings()
    return AzureChatOpenAI(
        azure_endpoint=s["endpoint"],
        api_key=s["api_key"],
        api_version=s["api_version"],
        azure_deployment=s["deployment"],
        temperature=temperature,
    )
