"""Package init.

Ensure a modern sqlite3 for ChromaDB. Streamlit Community Cloud ships a Python
whose bundled sqlite3 is older than the 3.35 that Chroma requires, which makes
``import chromadb`` crash. Swapping in the ``pysqlite3-binary`` build fixes it.

This runs before any submodule imports chromadb. It is a guarded no-op on
platforms where pysqlite3 isn't installed (e.g. local Windows dev), so the
stdlib sqlite3 is used there instead.
"""

try:
    import sys

    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
