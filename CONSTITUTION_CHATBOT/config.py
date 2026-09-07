import os

from dotenv import load_dotenv


load_dotenv()


# =========================================================
# DOCUMENT / DATABASE
# =========================================================

DATA_PATH = os.getenv(
    "DATA_PATH",
    "data"
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "chroma_db"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "tanzania_constitution"
)


# =========================================================
# CHAT HISTORY
# =========================================================

HISTORY_DB_PATH = os.getenv(
    "HISTORY_DB_PATH",
    "chat_history.db"
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

# Multilingual model for English + Swahili.
#
# This model runs locally.
# No embedding API key is required.

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================================================
# CHUNKING
# =========================================================

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1500"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "300"
    )
)


# =========================================================
# RETRIEVAL
# =========================================================

N_RESULTS = int(
    os.getenv(
        "N_RESULTS",
        "8"
    )
)


# Number of extra candidates retrieved
# before final selection.

RETRIEVAL_CANDIDATES = int(
    os.getenv(
        "RETRIEVAL_CANDIDATES",
        "15"
    )
)


# =========================================================
# LLM
# =========================================================

LLM_PROVIDER = "gemini"


# =========================================================
# GEMINI
# =========================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
)

GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GEMINI_TIMEOUT_SECONDS = int(
    os.getenv(
        "GEMINI_TIMEOUT_SECONDS",
        "120"
    )
)