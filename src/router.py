"""Router — asks GPT-4.1 which tool to use and parses the decision.

Returns a (tool, reason) pair. On any parse failure it falls back to the
default tool (GENERAL) so the graph never crashes on a malformed response.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from .config import DEFAULT_TOOL, TOOLS, get_llm
from .prompts.router_prompt import ROUTER_SYSTEM_PROMPT, build_router_user_prompt


def _parse_decision(raw: str) -> tuple[str, str]:
    """Extract {tool, reason} from the model output, tolerating extra prose."""
    # Find the first JSON object in the response.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            tool = str(data.get("tool", "")).strip().upper()
            reason = str(data.get("reason", "")).strip()
            if tool in TOOLS:
                return tool, (reason or "Router selected this tool.")
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: look for a bare label in the text.
    upper = raw.upper()
    for tool in TOOLS:
        if tool in upper:
            return tool, "Parsed tool label from router output (no valid JSON)."

    return DEFAULT_TOOL, "Router output could not be parsed; defaulted to GENERAL."


def route(question: str, has_documents: bool = False) -> tuple[str, str]:
    """Classify a question into one of MATH / DOC_SEARCH / GENERAL.

    Any exception (network, auth, parse) degrades gracefully to GENERAL.
    """
    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=build_router_user_prompt(question, has_documents)),
        ]
        resp = llm.invoke(messages)
        return _parse_decision(resp.content)
    except Exception as exc:  # noqa: BLE001 - we intentionally never crash routing
        return DEFAULT_TOOL, f"Router error ({type(exc).__name__}); defaulted to GENERAL."
