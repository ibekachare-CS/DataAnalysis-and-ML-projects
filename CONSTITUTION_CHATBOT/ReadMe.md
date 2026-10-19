
# Tanzania Constitution AI — RAG Assistant

An English-Swahili chatbot that gives responses based on the **Tanzanian constitution**, using **Retrieval-Augmented Generation (RAG)**. The relevant text from the vector store is fetched and passed to Google Gemini, thus ensuring the response is grounded in the constitution and not the memory of the model.

> **Disclaimer:** This is a legal *information and education* tool. It is not a lawyer and does not give legal advice. For official text, consult the [Office of the Attorney General](https://oagmis.oag.go.tz) or [TanzLII](https://tanzlii.org/en/).

---

## Features

- **Bilingual Q&A** - ask in English or Swahili (or combination thereof); assistant answers in the language asked.
- **Grounded answers** - answers are constructed based on the extracted constitutional texts, along with the Article (*Ibara*) and Chapter (*Sura*) references. The prompt does not allow for made-up Article numbers or legal facts.
- **Token-by-token generation** - answers are streamed token by token through the web UI.
- **Persistent conversation history** - conversations are persisted in SQLite and accessible from the sidebar (create, open, rename, delete).
- **Recent context usage** - recent messages are provided to the model, while the last question is used to enhance retrieval for short follow-ups such as *"Na Waziri Mkuu?"*.
- **Complete local retrieval** - embeddings are computed locally using multilingual Sentence-Transformers model; generation only calls the Gemini API.
- **Flexible knowledge base** - add more PDF/DOCX/TXT/MD documents to the corpus (acts, regulations, judgments) and re-run ingestion script.

---

## How It Works

```
User question
      │
      ▼
Query preparation  (adds the previous user question + EN/SW legal keywords)
      │
      ▼
Multilingual embedding  (paraphrase-multilingual-MiniLM-L12-v2)
      │
      ▼
ChromaDB semantic search  (15 candidates → de-duplicated → top 8 chunks)
      │
      ▼
System prompt  (rules + retrieved context + recent chat history)
      │
      ▼
Google Gemini  (gemini-2.5-flash, temperature 0.1)
      │
      ▼
Cleaned answer  (streamed to the browser, saved to SQLite)
```

**Ingestion (`fill_db.py`)** reads the documents, preps the text, divides it into overlapping chunks (default 1500 characters with 300 character overlap), adds metadata (file, doc type, language, page number, article number, chapter) to each chunk and stores the embeddings in ChromaDB.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend / API | FastAPI, Uvicorn |
| Frontend | Jinja2 template with a single-page HTML/JS chat UI |
| Vector database | ChromaDB (persistent, local) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Document loading & chunking | LangChain (`PyPDFLoader`, `Docx2txtLoader`, `RecursiveCharacterTextSplitter`) |
| LLM | Google Gemini via its OpenAI-compatible API |
| Chat history | SQLite (Python standard library) |

---

## Project Structure

```
CONSTITUTION_CHATBOT/
├── main.py            # FastAPI app: routes, sessions, streaming
├── rag_engine.py      # Retrieval + response generation pipeline
├── gemini_client.py   # Gemini API client (normal + streaming)
├── prompts.py         # System prompt and context builder
├── history_store.py   # SQLite chat history (sessions + messages)
├── config.py          # Settings loaded from environment variables
├── fill_db.py         # Ingestion: documents → chunks → ChromaDB
├── templates/
│   └── index.html     # Chat web interface
├── chroma_db/         # Persisted vector database
└── data/              # Source documents (create this; see below)
```

---

## Getting Started

### 1. Prerequisites

- Python **3.10+**
- A **Google Gemini API key** from [Google AI Studio](https://aistudio.google.com/)

### 2. Clone and enter the project

```bash
git clone https://github.com/ibekachare-CS/DataAnalysis-and-ML-projects.git
cd DataAnalysis-and-ML-projects/CONSTITUTION_CHATBOT
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install fastapi "uvicorn[standard]" jinja2 pydantic python-dotenv requests \
            chromadb sentence-transformers \
            langchain-community langchain-text-splitters pypdf docx2txt
```

### 4. Configure environment variables

Create a `.env` file in this folder:

```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Build the knowledge base (only if needed)

A pre-built `chroma_db/` may already be included. To (re)build it, or to add your own documents:

1. Create a `data/` folder next to `main.py`.
2. Put the Constitution PDF (and any other `.pdf`, `.docx`, `.txt`, `.md` files) inside it.
3. Run:

```bash
python fill_db.py
```

This **clears the existing collection and re-ingests everything** in `data/`. The first run downloads the embedding model from Hugging Face.

> File names affect the document-type label: names containing `katiba`/`constitution` are tagged *Constitution*, `sheria`/`act` as *Law*, `kanuni`/`regulation` as *Regulation*, and `judgment`/`hukumu` as *Judgment*.

### 6. Run the app

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## Configuration

All settings are optional environment variables (set them in `.env`) except `GEMINI_API_KEY`.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model used for answers |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | OpenAI-compatible endpoint |
| `GEMINI_TIMEOUT_SECONDS` | `120` | Timeout for non-streaming requests |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Local embedding model |
| `DATA_PATH` | `data` | Folder with source documents |
| `CHROMA_PATH` | `chroma_db` | ChromaDB storage folder |
| `COLLECTION_NAME` | `tanzania_constitution` | ChromaDB collection name |
| `HISTORY_DB_PATH` | `chat_history.db` | SQLite chat history file |
| `CHUNK_SIZE` | `1500` | Characters per chunk |
| `CHUNK_OVERLAP` | `300` | Overlap between chunks |
| `N_RESULTS` | `8` | Chunks passed to the LLM |
| `RETRIEVAL_CANDIDATES` | `15` | Candidates retrieved before de-duplication |

> If you change `EMBEDDING_MODEL` or the chunk settings, re-run `python fill_db.py`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat web interface |
| `POST` | `/ask` | Ask a question, returns the full answer as JSON |
| `POST` | `/ask/stream` | Ask a question, streams the answer as plain text (session id in the `X-Session-Id` header) |
| `GET` | `/sessions` | List all chat sessions |
| `POST` | `/sessions` | Create a new session |
| `GET` | `/sessions/{id}` | Get a session with its messages |
| `PATCH` | `/sessions/{id}` | Rename a session |
| `DELETE` | `/sessions/{id}` | Delete a session |
| `GET` | `/health` | Health check |

Interactive API docs are available at **http://127.0.0.1:8000/docs**.

**Example request:**

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Rais ana mamlaka gani?"}'
```

```json
{
  "response": "...",
  "session_id": "..."
}
```

Pass the returned `session_id` in later requests to continue the same conversation.

---

## Example Questions

- *What is the Constitution?*
- *Katiba ni nini?*
- *Waziri Mkuu ana kazi gani?*
- *What are the fundamental rights in the Constitution?*
- *Eleza Ibara ya 13.*
- *Compare the roles of the President and the Prime Minister.*

---

## Limitations

- Answer quality depends on the documents ingested and on how well retrieval matches the question.
- Article and chapter metadata are detected automatically with pattern matching and can occasionally be missing or imprecise.
- Requires an internet connection and a valid API key for answer generation.
- Not a substitute for professional legal advice or the official text of the law.

---

## Possible Improvements

- Add a `requirements.txt` with pinned versions
- Re-ranking of retrieved chunks
- Source citations shown directly in the UI
- Evaluation set to measure retrieval and answer quality
- Docker support for easier deployment

---

## Author

**ibekachare-CS** — [GitHub](https://github.com/ibekachare-CS)

Part of the [DataAnalysis-and-ML-projects](https://github.com/ibekachare-CS/DataAnalysis-and-ML-projects) collection.
