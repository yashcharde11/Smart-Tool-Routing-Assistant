"""General tool — a direct GPT-4.1 answer for questions that need no tool."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_llm

_SYSTEM = (
    "You are a helpful, concise assistant for an internal helpdesk. "
    "Answer the user's general question directly and accurately. "
    "If the question seems to require a specific uploaded document or a precise "
    "calculation you were not given, say so briefly."
)


def answer_general(question: str) -> str:
    """Send the question straight to the LLM and return its reply."""
    llm = get_llm()
    resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=question)])
    return resp.content.strip()
