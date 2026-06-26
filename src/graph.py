"""LangGraph state machine.

The agent is a StateGraph with one router node and three tool nodes. A
conditional edge routes the shared state from the router to exactly one tool
node based on ``state['tool']``; that node writes the answer and the graph ends.

    router ──(conditional)──> calculator ─┐
                            ├─> rag       ├─> END
                            └─> general  ─┘

Adding a tool later = one new node + one new branch in the conditional edge.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .router import route
from .tools.calculator import CalculatorError, calculate
from .tools.general import answer_general
from .tools.rag import answer_with_rag


class RouterState(TypedDict, total=False):
    """Shared state carried through the graph; each node fills its part."""

    question: str
    has_documents: bool
    tool: str
    reason: str
    context: list  # retrieved RAG chunks (empty for non-RAG paths)
    answer: str


# --- Nodes ---------------------------------------------------------------

def router_node(state: RouterState) -> RouterState:
    tool, reason = route(state["question"], state.get("has_documents", False))
    return {"tool": tool, "reason": reason}


def calculator_node(state: RouterState) -> RouterState:
    try:
        answer = calculate(state["question"])
    except CalculatorError as exc:
        answer = (
            f"I routed this to the calculator but couldn't compute it: {exc} "
            "Try rephrasing as a clear arithmetic expression."
        )
    return {"answer": answer, "context": []}


def rag_node(state: RouterState) -> RouterState:
    answer, hits = answer_with_rag(state["question"])
    return {"answer": answer, "context": hits}


def general_node(state: RouterState) -> RouterState:
    return {"answer": answer_general(state["question"]), "context": []}


# --- Conditional edge ----------------------------------------------------

def _select_tool(state: RouterState) -> str:
    """Map the router's decision to a node name (default: general)."""
    return {
        "MATH": "calculator",
        "DOC_SEARCH": "rag",
        "GENERAL": "general",
    }.get(state.get("tool", "GENERAL"), "general")


@lru_cache(maxsize=1)
def build_graph():
    """Construct and compile the StateGraph once."""
    g = StateGraph(RouterState)
    g.add_node("router", router_node)
    g.add_node("calculator", calculator_node)
    g.add_node("rag", rag_node)
    g.add_node("general", general_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        _select_tool,
        {"calculator": "calculator", "rag": "rag", "general": "general"},
    )
    for node in ("calculator", "rag", "general"):
        g.add_edge(node, END)

    return g.compile()


def run_agent(question: str, has_documents: bool = False) -> RouterState:
    """Invoke the compiled graph for a single question and return final state."""
    graph = build_graph()
    initial: RouterState = {"question": question, "has_documents": has_documents}
    return graph.invoke(initial)
