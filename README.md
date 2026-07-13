# 🏠 Real Estate AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) chat assistant for real estate
documents (listings, contracts, market reports, policies), built with **Streamlit**,
**Google Gemini**, **LangChain**, and **FAISS**.

## Dataset

Ships pre-loaded with a **92-document real estate knowledge base** (`data/documents/`)
covering 3 fictional builders and 6 projects, consistent across every file so
cross-document questions have a real, traceable answer:

| Format | Count | Contents |
|---|---|---|
| PDF | 21 | Brochures, payment plans, RERA summaries, builder profiles |
| DOCX | 21 | Sale agreement terms, registration process, possession guidelines, cancellation/refund policy |
| HTML | 23 | Builder websites, FAQs, privacy policy, terms & conditions, listing portal pages |
| Markdown | 27 | Amenities/location/floor-plan guides, home loan info, customer support docs |

All prices, RERA numbers, and dates in the dataset are synthetic (demo data only).

## Features

- 🔐 Login page with bcrypt-hashed password authentication + session timeout
- 📄 PDF, DOCX, HTML, and Markdown ingestion (recursive folder scan of `data/documents/`)
- ✂️ Configurable chunking (`RecursiveCharacterTextSplitter`)
- 🧠 Google Gemini embeddings + FAISS vector store (persisted to disk)
- 💬 Multi-turn chat with conversational memory (follow-up questions work)
- 📚 Source citations under every answer (file name + page number)
- 🗂️ Chat history retained for the session, with a **Clear conversation** button
- 🔁 One-click **Rebuild index** from the sidebar when you add new documents
- 🧱 Modular folder structure, `.env`-based configuration, ready to deploy

## Project structure

```
real-estate-ai-assistant/
├── app.py                        # Streamlit entry point
├── auth/
│   ├── authentication.py         # Login form, session/auth state
│   └── generate_password_hash.py # CLI to create APP_PASSWORD_HASH
├── config/
│   └── settings.py                # Loads & validates all env-based config
├── core/
│   ├── document_loader.py         # PDF/DOCX loading + chunking
│   ├── vector_store.py            # FAISS build/load/persist
│   └── rag_chain.py               # Gemini + retriever + memory + citations
├── utils/
│   └── helpers.py                  # Chat message formatting helpers
├── data/documents/                 # <-- put your dataset (PDF/DOCX) here
├── vectorstore/                    # FAISS index is persisted here (gitignored)
├── .streamlit/config.toml          # Theme + server config
├── .env.example                    # Copy to .env and fill in
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
git clone <your-repo-url>
cd real-estate-ai-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

- `GOOGLE_API_KEY` — get one from [Google AI Studio](https://aistudio.google.com/app/apikey)
- `GEMINI_CHAT_MODEL` / `GEMINI_EMBEDDING_MODEL` — defaults work out of the box
- `APP_USERNAME` — the login username
- `APP_PASSWORD_HASH` — generate with:

  ```bash
  python -m auth.generate_password_hash
  ```

  Paste the printed hash into `.env` as `APP_PASSWORD_HASH`.

- `AUTH_SECRET_KEY` — any long random string

## 3. Dataset

The knowledge base is already in `data/documents/` (pdf/, docx/, html/, markdown/
subfolders — see the **Dataset** section above). To use your own instead,
just replace the contents with your own PDF, DOCX, HTML, or Markdown files
(subfolders are fine, the loader scans recursively) and click **Rebuild index**.

## 4. Run the app

```bash
streamlit run app.py
```

Log in with the username/password you configured, then click **Rebuild index**
in the sidebar the first time (it reads every file in `data/documents/`,
chunks it, embeds it with Gemini, and saves a FAISS index to `vectorstore/`).
On subsequent runs the existing index loads automatically — click **Rebuild
index** again any time you add or change source documents.

## How it works (architecture)

1. **Loading** — `core/document_loader.py` uses `PyPDFLoader` for PDFs,
   `Docx2txtLoader` for DOCX, `BSHTMLLoader` for HTML, and `TextLoader` for
   Markdown, tagging every page/section with its source filename and page number.
2. **Chunking** — `RecursiveCharacterTextSplitter` splits documents into
   overlapping chunks (`CHUNK_SIZE` / `CHUNK_OVERLAP` in `.env`), and each
   chunk gets a `citation` label like `market_report.pdf (page 4)`.
3. **Embedding + indexing** — `core/vector_store.py` embeds chunks with
   `GoogleGenerativeAIEmbeddings` and stores vectors in a local **FAISS**
   index, persisted under `vectorstore/`.
4. **Retrieval + generation** — `core/rag_chain.py` wires a
   `ConversationalRetrievalChain` (LangChain) around `ChatGoogleGenerativeAI`
   (Gemini), the FAISS retriever (top-K similarity search), and a
   `ConversationBufferMemory` so follow-up questions retain context.
   The chain returns both the answer and the source chunks used, which the
   UI renders as citations under each response.
5. **UI** — `app.py` (Streamlit) handles login gating, sidebar controls
   (rebuild index, clear chat, model info), and the chat interface itself,
   persisting messages in `st.session_state.chat_history` for the session.

## Deployment

### Streamlit Community Cloud
1. Push this repo to GitHub (`.env` is gitignored — do **not** commit it).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `app.py`.
3. In the app's **Settings → Secrets**, paste the contents of your `.env`
   file (Streamlit Cloud reads secrets as env vars automatically).
4. Deploy. On first login, click **Rebuild index** (or pre-build the index
   locally and commit `vectorstore/` if your dataset is static and small).

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t real-estate-ai-assistant .
docker run -p 8501:8501 --env-file .env real-estate-ai-assistant
```

### Other platforms (Render, Railway, AWS/GCP/Azure App Service, etc.)
Set the same environment variables from `.env.example` in the platform's
secret/env manager, and use `streamlit run app.py --server.port=$PORT
--server.address=0.0.0.0` as the start command.

## Notes & production hardening ideas

- **Auth**: current auth is single-user (one username/hash in `.env`). For
  multiple users, swap `auth/authentication.py` for a real user store or an
  SSO provider (Auth0, Okta, Google OAuth via `streamlit-authenticator`).
- **Vector store**: FAISS is file-based and fine for small/medium datasets
  on a single instance. For larger scale or multi-instance deployments,
  consider a managed vector DB (Pinecone, Weaviate, Qdrant) — only
  `core/vector_store.py` needs to change.
- **Rate limits / cost**: Gemini API calls are billed per request; consider
  caching frequent Q&A pairs or adding request throttling for public deployments.
- **Secrets**: never commit `.env`; use your platform's secret manager in production.
