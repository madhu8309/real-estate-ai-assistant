"""
Builds, saves, loads, and updates a lightweight local vector store using Google
Generative AI embeddings.

NOTE ON MODEL NAMES (as of mid-2026):
Google has retired several older embedding models:
  - "embedding-001"        -> retired Oct 30, 2025
  - "text-embedding-004"   -> retired Jan 14, 2026
  - "gemini-embedding-001" -> scheduled to retire Jul 14, 2026
The current recommended model is "gemini-embedding-2" (GA). The Gemini API
expects the fully-qualified resource form "models/<model-name>" — a bare
name like "text-embedding-004" or a name with stray whitespace/quotes from
a .env file causes the "unexpected model name format" error. We normalize
whatever is configured so this can't happen regardless of how it's set.

NOTE ON RATE LIMITS:
The free tier caps embed_content requests at 100/minute. Embedding a whole
document set in one call can trip this (langchain_google_genai's internal
batching still fires several requests back-to-back with no backoff, and it
aborts the entire build on the first 429). To avoid that, we embed chunks
ourselves in small batches with a short pause between them, and retry with
backoff (honoring the server's suggested retry_delay when present) if a
429 slips through anyway.
"""
import re
import pickle
import math
import time
from pathlib import Path
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config.settings import settings

_INDEX_NAME = "local_index.pkl"

# Safe fallback if GEMINI_EMBEDDING_MODEL is unset in .env.
_DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-2"

# Keep batches small and paced so we stay comfortably under the free-tier
# 100-requests/minute cap even on large document sets.
_EMBED_BATCH_SIZE = 10
_INTER_BATCH_PAUSE_SECONDS = 2.0
_MAX_RETRIES_PER_BATCH = 6
_BASE_BACKOFF_SECONDS = 5.0

_RETRY_DELAY_RE = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalVectorStore:
    """Tiny persisted vector store that avoids native FAISS dependencies."""

    def __init__(
        self,
        embeddings: GoogleGenerativeAIEmbeddings,
        documents: list[Document],
        vectors: list[list[float]],
    ) -> None:
        self.embeddings = embeddings
        self.documents = documents
        self.vectors = vectors

    def as_retriever(self, search_type: str = "similarity", search_kwargs: dict[str, Any] | None = None):
        if search_type != "similarity":
            raise ValueError("LocalVectorStore only supports similarity search.")
        k = (search_kwargs or {}).get("k", settings.RETRIEVER_TOP_K)
        return LocalVectorStoreRetriever(vector_store=self, k=k)

    def similarity_search(self, query: str, k: int) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        scored = [
            (_cosine_similarity(query_vector, vector), document)
            for vector, document in zip(self.vectors, self.documents)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:k]]

    def save_local(self, vectorstore_dir: Path) -> None:
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        with (vectorstore_dir / _INDEX_NAME).open("wb") as file:
            pickle.dump(
                {"documents": self.documents, "vectors": self.vectors},
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load_local(
        cls,
        vectorstore_dir: Path,
        embeddings: GoogleGenerativeAIEmbeddings,
    ) -> "LocalVectorStore":
        with (vectorstore_dir / _INDEX_NAME).open("rb") as file:
            payload = pickle.load(file)
        return cls(
            embeddings=embeddings,
            documents=payload["documents"],
            vectors=payload["vectors"],
        )


class LocalVectorStoreRetriever(BaseRetriever):
    vector_store: Any
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return self.vector_store.similarity_search(query, self.k)


def _normalize_embedding_model_name(raw_name: str | None) -> str:
    """
    Make sure the embedding model name is in the exact form the Gemini API
    expects: "models/<name>", with no surrounding whitespace/quotes and no
    accidental double "models/models/" prefix.
    """
    name = (raw_name or "").strip().strip('"').strip("'").strip()

    if not name:
        return _DEFAULT_EMBEDDING_MODEL

    # Collapse any accidental "models/models/..." into a single prefix.
    while name.startswith("models/models/"):
        name = name[len("models/"):]

    if not name.startswith("models/"):
        name = f"models/{name}"

    return name


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    model_name = _normalize_embedding_model_name(settings.GEMINI_EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=settings.GOOGLE_API_KEY,
    )


def _is_rate_limit_error(message: str) -> bool:
    lowered = message.lower()
    return "429" in message or "quota" in lowered or "resourceexhausted" in lowered.replace(" ", "")


def _extract_retry_delay_seconds(message: str) -> float | None:
    match = _RETRY_DELAY_RE.search(message)
    if match:
        return float(match.group(1)) + 1.0  # small safety buffer
    return None


def _embed_batch_with_retry(embeddings: GoogleGenerativeAIEmbeddings, texts: list[str]) -> list[list[float]]:
    """Embed one small batch of texts, retrying with backoff if rate-limited."""
    last_error: Exception | None = None

    for attempt in range(1, _MAX_RETRIES_PER_BATCH + 1):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001 - inspect message to decide retry vs. raise
            message = str(exc)
            if not _is_rate_limit_error(message):
                raise

            last_error = exc
            if attempt == _MAX_RETRIES_PER_BATCH:
                break

            delay = _extract_retry_delay_seconds(message) or (_BASE_BACKOFF_SECONDS * attempt)
            print(
                f"[embeddings] Rate limited (attempt {attempt}/{_MAX_RETRIES_PER_BATCH}). "
                f"Waiting {delay:.0f}s before retrying this batch..."
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Embedding batch failed after {_MAX_RETRIES_PER_BATCH} retries due to persistent "
        f"rate limiting: {last_error}"
    ) from last_error


def _embed_chunks_in_batches(
    embeddings: GoogleGenerativeAIEmbeddings,
    chunks: list[Document],
    batch_size: int = _EMBED_BATCH_SIZE,
    pause_seconds: float = _INTER_BATCH_PAUSE_SECONDS,
) -> list[tuple[str, list[float]]]:
    """
    Embed every chunk's text in small, paced batches (to respect free-tier
    rate limits), retrying on 429s. Returns (text, vector) pairs in the same
    order as `chunks`, ready for FAISS.from_embeddings / add_embeddings.
    """
    texts = [c.page_content for c in chunks]
    total = len(texts)
    pairs: list[tuple[str, list[float]]] = []

    for start in range(0, total, batch_size):
        batch_texts = texts[start:start + batch_size]
        vectors = _embed_batch_with_retry(embeddings, batch_texts)
        pairs.extend(zip(batch_texts, vectors))

        done = min(start + batch_size, total)
        print(f"[embeddings] Embedded {done}/{total} chunks...")

        if done < total:
            time.sleep(pause_seconds)

    return pairs


def index_exists(vectorstore_dir: Path | None = None) -> bool:
    vectorstore_dir = vectorstore_dir or settings.VECTORSTORE_DIR
    return (vectorstore_dir / _INDEX_NAME).exists()


def build_vector_store(chunks: list[Document], vectorstore_dir: Path | None = None) -> LocalVectorStore:
    """Build a fresh local index from document chunks and persist it to disk."""
    if not chunks:
        raise ValueError(
            "No document chunks to index. Add PDF/DOCX files to the data directory first."
        )

    vectorstore_dir = vectorstore_dir or settings.VECTORSTORE_DIR
    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()
    metadatas = [c.metadata for c in chunks]

    try:
        text_embedding_pairs = _embed_chunks_in_batches(embeddings, chunks)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Error embedding content with model '{embeddings.model}': {exc}\n"
            "If this mentions quota/429, wait a minute for the free-tier quota to reset "
            "and click 'Rebuild index' again — the batching/retry logic will avoid "
            "re-embedding faster than the API allows.\n"
            "If this mentions an unknown/retired model, set GEMINI_EMBEDDING_MODEL "
            "in your .env to 'models/gemini-embedding-2'."
        ) from exc

    vectors = [vector for _, vector in text_embedding_pairs]
    documents = [
        Document(page_content=chunk.page_content, metadata=metadata)
        for chunk, metadata in zip(chunks, metadatas)
    ]
    vector_store = LocalVectorStore(embeddings=embeddings, documents=documents, vectors=vectors)
    vector_store.save_local(vectorstore_dir)
    return vector_store


def load_vector_store(vectorstore_dir: Path | None = None) -> LocalVectorStore:
    """Load a previously persisted local index from disk."""
    vectorstore_dir = vectorstore_dir or settings.VECTORSTORE_DIR
    embeddings = get_embeddings()
    return LocalVectorStore.load_local(vectorstore_dir, embeddings)


def get_or_build_vector_store(
    chunks: list[Document] | None = None,
    vectorstore_dir: Path | None = None,
    force_rebuild: bool = False,
) -> LocalVectorStore:
    """Load the index if it exists, otherwise build it from the given chunks."""
    vectorstore_dir = vectorstore_dir or settings.VECTORSTORE_DIR

    if not force_rebuild and index_exists(vectorstore_dir):
        return load_vector_store(vectorstore_dir)

    if chunks is None:
        raise ValueError("No existing index found and no chunks provided to build one.")

    return build_vector_store(chunks, vectorstore_dir)


def add_documents(
    vector_store: LocalVectorStore,
    chunks: list[Document],
    vectorstore_dir: Path | None = None,
) -> LocalVectorStore:
    """Add new chunks to an existing local index and persist the update."""
    vectorstore_dir = vectorstore_dir or settings.VECTORSTORE_DIR

    embeddings = get_embeddings()
    metadatas = [c.metadata for c in chunks]

    try:
        text_embedding_pairs = _embed_chunks_in_batches(embeddings, chunks)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Error embedding content with model '{embeddings.model}': {exc}"
        ) from exc

    vector_store.documents.extend(
        Document(page_content=chunk.page_content, metadata=metadata)
        for chunk, metadata in zip(chunks, metadatas)
    )
    vector_store.vectors.extend(vector for _, vector in text_embedding_pairs)
    vector_store.save_local(vectorstore_dir)
    return vector_store
