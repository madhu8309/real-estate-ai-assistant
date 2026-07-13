"""
Small shared helpers: chat message formatting and lightweight session-scoped
chat history persistence.
"""
from datetime import datetime


def now_str() -> str:
    return datetime.now().strftime("%H:%M")


def new_message(role: str, content: str, sources: list[str] | None = None) -> dict:
    """Create a chat message dict for storage in st.session_state.chat_history."""
    return {
        "role": role,  # "user" or "assistant"
        "content": content,
        "sources": sources or [],
        "timestamp": now_str(),
    }


def format_sources(sources: list[str]) -> str:
    if not sources:
        return ""
    lines = "\n".join(f"- {s}" for s in sources)
    return f"**Sources:**\n{lines}"
