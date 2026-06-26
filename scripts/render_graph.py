"""Render the LangGraph state machine to a PNG for the README.

Uses LangGraph's built-in Mermaid renderer:

    python -m scripts.render_graph

Writes ``docs/graph.png``. The PNG render uses the Mermaid.INK web service; if
that is unreachable (offline), we fall back to writing the Mermaid source to
``docs/graph.mmd`` so you can render it elsewhere. No Azure credentials are
needed — building the graph does not call the LLM.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph import build_graph  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    graph = build_graph().get_graph()

    # Always save the Mermaid source (cheap, offline, diff-friendly).
    mmd_path = os.path.join(OUT_DIR, "graph.mmd")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(graph.draw_mermaid())
    print(f"Saved Mermaid source -> {mmd_path}")

    # Try the PNG render (needs network access to mermaid.ink).
    png_path = os.path.join(OUT_DIR, "graph.png")
    try:
        png = graph.draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png)
        print(f"Saved PNG -> {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"PNG render unavailable ({type(exc).__name__}: {exc}).")
        print(f"Use the Mermaid source at {mmd_path} instead.")


if __name__ == "__main__":
    main()
