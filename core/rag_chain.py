"""
Builds the conversational retrieval (RAG) chain: Gemini chat model + FAISS
retriever + conversation memory, and formats answers with source citations.

NOTE ON CHAT MODEL NAMES (as of mid-2026):
Google periodically restricts older/preview chat models (e.g. "gemini-1.5-pro",
"gemini-2.5-flash") from new AI Studio accounts, even before they're formally
shut down for existing users ("This model is no longer available to new
users."). "gemini-flash-latest" is a Google-maintained alias that always
points at the current GA Flash model (gemini-3.5-flash as of June 2026), so
it's the safest default for new accounts. Pin an explicit version (e.g.
"gemini-3.5-flash") instead if you need reproducible behavior over time.
"""
from dataclasses import dataclass, field

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

SYSTEM_PROMPT = """You are a knowledgeable Real Estate AI Assistant. Answer the user's \
question using ONLY the context provided below, which comes from the real estate \
knowledge base (property listings, contracts, market reports, policies, etc.).

Rules:
- If the answer is not contained in the context, say you don't have that information \
in the knowledge base, and offer to help with something else. Do not make up facts.
- Be concise, professional, and specific (mention figures, addresses, terms, dates when present).
- If the question is a general greeting or small talk, respond naturally without forcing \
context into the answer.

Context:
{context}

Conversation so far:
{chat_history}

Question: {question}

Answer:"""

_DEFAULT_CHAT_MODEL = "gemini-flash-latest"


@dataclass
class RagResult:
    answer: str
    sources: list[str] = field(default_factory=list)


def _normalize_chat_model_name(raw_name: str | None) -> str:
    """Strip stray whitespace/quotes from the configured chat model name."""
    name = (raw_name or "").strip().strip('"').strip("'").strip()
    return name or _DEFAULT_CHAT_MODEL


def _build_llm() -> ChatGoogleGenerativeAI:
    model_name = _normalize_chat_model_name(settings.GEMINI_CHAT_MODEL)
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.TEMPERATURE,
            convert_system_message_to_human=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Error initializing chat model '{model_name}': {exc}\n"
            "If this says the model is no longer available to new users, set "
            "GEMINI_CHAT_MODEL in your .env to 'gemini-flash-latest' (or a specific "
            "current GA model such as 'gemini-3.5-flash') and restart the app."
        ) from exc


def build_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )


def build_rag_chain(vector_store: FAISS, memory: ConversationBufferMemory) -> ConversationalRetrievalChain:
    """Build a ConversationalRetrievalChain wired up with Gemini, FAISS, and memory."""
    llm = _build_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVER_TOP_K},
    )

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "chat_history", "question"],
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return chain


def ask_question(chain: ConversationalRetrievalChain, question: str) -> RagResult:
    """Run a question through the chain and return the answer plus source citations."""
    result = chain.invoke({"question": question})

    answer = result.get("answer", "").strip()
    source_docs = result.get("source_documents", []) or []

    seen = set()
    sources: list[str] = []
    for doc in source_docs:
        citation = doc.metadata.get("citation") or doc.metadata.get("source_file", "unknown source")
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)

    return RagResult(answer=answer, sources=sources)
