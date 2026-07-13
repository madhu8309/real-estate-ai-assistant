"""
Real Estate AI Assistant — Streamlit entry point.

Run with:
    streamlit run app.py
"""
import streamlit as st

from config.settings import settings
from auth.authentication import is_authenticated, render_login_form, logout
from core.document_loader import discover_documents, load_and_chunk
from core.vector_store import get_or_build_vector_store, index_exists, add_documents
from core.rag_chain import build_memory, build_rag_chain, ask_question
from utils.helpers import new_message, format_sources

st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def init_session_state():
    st.session_state.setdefault("chat_history", [])  # list of dicts (see utils.helpers.new_message)
    st.session_state.setdefault("vector_store", None)
    st.session_state.setdefault("rag_chain", None)
    st.session_state.setdefault("memory", None)


def ensure_chain_ready():
    """Lazily build (or load) the vector store + RAG chain once per session."""
    if st.session_state["rag_chain"] is not None:
        return True

    if not index_exists():
        return False

    with st.spinner("Loading knowledge base..."):
        st.session_state["vector_store"] = get_or_build_vector_store()
        st.session_state["memory"] = build_memory()
        st.session_state["rag_chain"] = build_rag_chain(
            st.session_state["vector_store"], st.session_state["memory"]
        )
    return True


def rebuild_index(force: bool = True):
    with st.spinner("Reading documents and rebuilding the FAISS index... this may take a minute."):
        chunks = load_and_chunk()
        vector_store = get_or_build_vector_store(chunks=chunks, force_rebuild=force)
        st.session_state["vector_store"] = vector_store
        st.session_state["memory"] = build_memory()
        st.session_state["rag_chain"] = build_rag_chain(vector_store, st.session_state["memory"])
    return len(chunks)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 🏠 {settings.APP_TITLE}")
        st.caption(f"Signed in as **{st.session_state.get('username', 'user')}**")

        st.divider()
        st.markdown("#### 📚 Knowledge base")
        docs = discover_documents()
        if docs:
            st.caption(f"{len(docs)} document(s) found in `data/documents/`")
            with st.expander("View files"):
                for d in docs:
                    st.write(f"- {d.name}")
        else:
            st.warning(
                "No PDF or DOCX files found in `data/documents/`. "
                "Add your dataset there, then click 'Rebuild index'."
            )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild index", use_container_width=True, disabled=not docs):
                try:
                    n = rebuild_index(force=True)
                    st.success(f"Index rebuilt from {n} chunks.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Failed to build index: {exc}")
        with col2:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state["chat_history"] = []
                if st.session_state["memory"] is not None:
                    st.session_state["memory"].clear()
                st.rerun()

        st.divider()
        with st.expander("⚙️ Model settings"):
            st.write(f"**Chat model:** `{settings.GEMINI_CHAT_MODEL}`")
            st.write(f"**Embedding model:** `{settings.GEMINI_EMBEDDING_MODEL}`")
            st.write(f"**Chunk size / overlap:** {settings.CHUNK_SIZE} / {settings.CHUNK_OVERLAP}")
            st.write(f"**Top-K retrieved chunks:** {settings.RETRIEVER_TOP_K}")

        st.divider()
        if st.button("🚪 Log out", use_container_width=True):
            logout()
            st.rerun()


# --------------------------------------------------------------------------
# Main chat UI
# --------------------------------------------------------------------------
def render_chat():
    st.markdown(f"## 🏠 {settings.APP_TITLE}")
    st.caption("Ask questions about your real estate documents — listings, contracts, market reports, and more.")

    config_problems = settings.validate()
    if config_problems:
        for p in config_problems:
            st.error(p)
        st.stop()

    chain_ready = ensure_chain_ready()

    # Render existing history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                st.caption(format_sources(msg["sources"]))

    if not chain_ready:
        st.info(
            "No index found yet. Add PDF/DOCX files to `data/documents/` and click "
            "**Rebuild index** in the sidebar to get started."
        )
        return

    question = st.chat_input("Ask about a property, contract clause, market trend...")
    if not question:
        return

    st.session_state["chat_history"].append(new_message("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_question(st.session_state["rag_chain"], question)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong while generating the answer: {exc}")
                return
        st.markdown(result.answer)
        if result.sources:
            st.caption(format_sources(result.sources))

    st.session_state["chat_history"].append(
        new_message("assistant", result.answer, result.sources)
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    init_session_state()

    if not is_authenticated():
        render_login_form()
        return

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
