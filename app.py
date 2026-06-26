"""Smart Tool-Routing Assistant — Streamlit UI.

Chat box + document sidebar. Every answer shows a badge with the tool the
router chose and a one-line reason, so the routing decision is visible during a
demo.
"""

from __future__ import annotations

import streamlit as st

from src.graph import run_agent
from src.tools.rag import has_documents, ingest, reset_documents

st.set_page_config(page_title="Smart Tool-Routing Assistant", page_icon="🧭")

# Badge styling per tool.
_TOOL_BADGE = {
    "MATH": ("🧮 Calculator", "#1f6feb"),
    "DOC_SEARCH": ("📄 Document Search (RAG)", "#8957e5"),
    "GENERAL": ("💬 Direct LLM", "#2da44e"),
}


def render_badge(tool: str, reason: str) -> None:
    label, color = _TOOL_BADGE.get(tool, (tool, "#6e7681"))
    st.markdown(
        f"""<div style="display:inline-block;padding:2px 10px;border-radius:12px;
        background:{color};color:white;font-size:0.8rem;font-weight:600;">
        Tool used: {label}</div>
        <div style="color:#6e7681;font-size:0.8rem;margin-top:4px;">↳ {reason}</div>""",
        unsafe_allow_html=True,
    )


# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, tool, reason, context

# --- Sidebar: document upload ---
with st.sidebar:
    st.header("📚 Documents")
    st.caption("Upload files for the DOC_SEARCH (RAG) tool. PDF, TXT, MD.")
    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    # Auto-index newly uploaded files. We track already-indexed files by
    # (name, size) in session state so re-runs don't re-embed the same file,
    # and so the user never has to remember to click a separate button.
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = {}

    if uploads:
        for f in uploads:
            data = f.read()
            key = f"{f.name}:{len(data)}"
            if key in st.session_state.indexed_files:
                continue
            with st.spinner(f"Embedding and indexing {f.name}…"):
                n_chunks = ingest(f.name, data)
            st.session_state.indexed_files[key] = n_chunks
            if n_chunks == 0:
                st.error(
                    f"⚠️ '{f.name}' produced 0 searchable chunks. If it's a "
                    "scanned/image PDF, the text can't be extracted — upload a "
                    "text-based PDF, or a .txt / .md version."
                )
            else:
                st.success(f"Indexed {n_chunks} chunk(s) from {f.name}.")

    if has_documents():
        indexed = [
            f"{k.rsplit(':', 1)[0]} ({n})"
            for k, n in st.session_state.indexed_files.items()
            if n > 0
        ]
        st.info("Indexed & searchable:\n\n- " + "\n- ".join(indexed))
        if st.button("Clear all documents", use_container_width=True):
            reset_documents()
            st.session_state.indexed_files = {}
            st.rerun()
    else:
        st.warning("No documents indexed yet. Upload a PDF, TXT, or MD file above.")

    st.divider()
    st.caption(
        "Router (GPT-4.1) classifies each question as MATH / DOC_SEARCH / "
        "GENERAL, then a LangGraph node runs the matching tool."
    )

# --- Main: chat ---
st.title("🧭 Smart Tool-Routing Assistant")
st.caption(
    "Ask a calculation, a question about your uploaded docs, or anything else — "
    "the agent picks the right tool and shows you which one."
)

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("tool"):
            render_badge(msg["tool"], msg.get("reason", ""))
        st.markdown(msg["content"])
        if msg.get("context"):
            with st.expander("Retrieved context"):
                for h in msg["context"]:
                    st.markdown(f"**{h['source']}** (distance {h['distance']:.3f})")
                    st.text(h["text"][:600])

# Input.
if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Routing…"):
                state = run_agent(prompt, has_documents=has_documents())
            render_badge(state.get("tool", "GENERAL"), state.get("reason", ""))
            st.markdown(state.get("answer", "(no answer)"))
            context = state.get("context") or []
            if context:
                with st.expander("Retrieved context"):
                    for h in context:
                        st.markdown(f"**{h['source']}** (distance {h['distance']:.3f})")
                        st.text(h["text"][:600])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": state.get("answer", "(no answer)"),
                    "tool": state.get("tool", "GENERAL"),
                    "reason": state.get("reason", ""),
                    "context": context,
                }
            )
        except Exception as exc:  # noqa: BLE001
            err = f"Something went wrong: {type(exc).__name__}: {exc}"
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": err, "tool": None, "reason": ""}
            )
