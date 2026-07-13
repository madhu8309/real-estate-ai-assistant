"""
Loads PDF, DOCX, HTML, and Markdown documents from the configured data
directory and splits them into chunks ready for embedding.
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    TextLoader,
)

from config.settings import settings

# The real estate knowledge base ships in four formats: PDF (brochures,
# payment plans, RERA summaries), DOCX (legal/registration/possession docs),
# HTML (builder sites, FAQs, privacy policy, listing portals), and Markdown
# (amenities/location/floor-plan/home-loan guides).
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".md"}


def discover_documents(data_dir: Path | None = None) -> list[Path]:
    """Return a sorted list of supported document paths under data_dir (recursive)."""
    data_dir = data_dir or settings.DATA_DIR
    if not data_dir.exists():
        return []
    files = [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def load_single_document(path: Path) -> list[Document]:
    """Load one PDF/DOCX/HTML/Markdown file into LangChain Document objects (one per page/section)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in (".html", ".htm"):
        loader = BSHTMLLoader(str(path), open_encoding="utf-8")
    elif suffix == ".md":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    for doc in docs:
        # Normalize metadata so downstream citation formatting is consistent
        doc.metadata["source_file"] = path.name
        doc.metadata.setdefault("page", doc.metadata.get("page", 0))
    return docs


def load_all_documents(data_dir: Path | None = None) -> list[Document]:
    """Load every supported document under data_dir into a flat list of Documents."""
    data_dir = data_dir or settings.DATA_DIR
    all_docs: list[Document] = []
    errors: list[str] = []

    for path in discover_documents(data_dir):
        try:
            all_docs.extend(load_single_document(path))
        except Exception as exc:  # noqa: BLE001 - surface all loader failures to the caller
            errors.append(f"{path.name}: {exc}")

    if errors:
        # Non-fatal: log to stdout so it shows in server logs; caller can also
        # surface `errors` in the UI if desired.
        print("Some documents failed to load:\n" + "\n".join(errors))

    return all_docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents using RecursiveCharacterTextSplitter with configured size/overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Give each chunk a stable, human-readable citation label
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["citation"] = f"{source} (page {page + 1})" if page is not None else source
        chunk.metadata["chunk_id"] = i

    return chunks


def load_and_chunk(data_dir: Path | None = None) -> list[Document]:
    """Convenience wrapper: load all documents then split into chunks."""
    docs = load_all_documents(data_dir)
    return split_documents(docs)
