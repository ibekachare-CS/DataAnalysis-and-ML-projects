"""
Tanzania Legal / Constitution RAG
ChromaDB Ingestion Pipeline

Pipeline:

Documents
    ↓
Load all supported files
    ↓
Clean text
    ↓
Legal-aware chunking
    ↓
Extract metadata
    ↓
Multilingual embeddings
    ↓
ChromaDB
"""


from pathlib import Path
import re
import hashlib

import chromadb

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from config import (
    DATA_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / DATA_PATH

CHROMA_DIR = BASE_DIR / CHROMA_PATH


# =========================================================
# SUPPORTED FILE TYPES
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".docx",
    ".md",
}


# =========================================================
# STARTUP
# =========================================================

print()
print("=" * 80)
print("TANZANIA LEGAL RAG - DOCUMENT INGESTION")
print("=" * 80)

print()
print(f"Data directory       : {DATA_DIR}")
print(f"ChromaDB directory   : {CHROMA_DIR}")
print(f"Collection           : {COLLECTION_NAME}")
print(f"Embedding model      : {EMBEDDING_MODEL}")
print(f"Chunk size           : {CHUNK_SIZE}")
print(f"Chunk overlap        : {CHUNK_OVERLAP}")
print()


# =========================================================
# CHECK DATA DIRECTORY
# =========================================================

if not DATA_DIR.exists():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raise FileNotFoundError(
        f"""
The data folder did not exist.

Created:

{DATA_DIR}

Put your legal documents inside it
and run fill_db.py again.
"""
    )


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text: str) -> str:

    if not text:
        return ""

    # Normalize line endings.
    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Remove null characters.
    text = text.replace(
        "\x00",
        " ",
    )

    # Normalize spaces while preserving
    # paragraph structure.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Remove isolated page numbers.
    text = re.sub(
        r"^\s*\d+\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove repeated blank lines.
    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# ARTICLE EXTRACTION
# =========================================================

def extract_article(text: str):

    if not text:
        return None

    patterns = [

        r"\bIbara\s+ya\s+(\d+)",

        r"\bIBARA\s+YA\s+(\d+)",

        r"\bIbara\s+(\d+)",

        r"\bIBARA\s+(\d+)",

        r"\bArticle\s+(\d+)",

        r"\bARTICLE\s+(\d+)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            return match.group(1)

    return None


# =========================================================
# CHAPTER EXTRACTION
# =========================================================

def extract_chapter(text: str):

    if not text:
        return None

    patterns = [

        r"\bSURA\s+YA\s+([A-Za-z0-9]+)",

        r"\bSURA\s+([A-Za-z0-9]+)",

        r"\bCHAPTER\s+([A-Za-z0-9]+)",

        r"\bChapter\s+([A-Za-z0-9]+)",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            return match.group(1)

    return None


# =========================================================
# DOCUMENT TYPE
# =========================================================

def determine_document_type(
    filename: str
) -> str:

    name = filename.lower()

    if (
        "katiba" in name
        or "constitution" in name
    ):
        return "Constitution"

    if (
        "sheria" in name
        or "act" in name
    ):
        return "Law"

    if (
        "kanuni" in name
        or "regulation" in name
    ):
        return "Regulation"

    if (
        "judgment" in name
        or "judgement" in name
        or "hukumu" in name
    ):
        return "Judgment"

    return "Legal Document"


# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language(text: str) -> str:

    if not text:
        return "Unknown"

    lower = text.lower()

    swahili_words = [
        "katiba",
        "ibara",
        "sheria",
        "rais",
        "bunge",
        "mahakama",
        "serikali",
        "haki",
        "waziri",
        "wananchi",
        "mamlaka",
        "uchaguzi",
        "jamhuri",
    ]

    english_words = [
        "constitution",
        "article",
        "law",
        "president",
        "parliament",
        "court",
        "government",
        "rights",
        "minister",
        "citizen",
        "authority",
        "election",
        "republic",
    ]

    swahili_score = sum(
        1
        for word in swahili_words
        if word in lower
    )

    english_score = sum(
        1
        for word in english_words
        if word in lower
    )

    if (
        swahili_score > english_score
    ):
        return "Swahili"

    if (
        english_score > swahili_score
    ):
        return "English"

    if (
        swahili_score > 0
        and english_score > 0
    ):
        return "Swahili/English"

    return "Unknown"


# =========================================================
# LOAD SINGLE FILE
# =========================================================

def load_file(
    file_path: Path
):

    extension = file_path.suffix.lower()

    print()
    print("-" * 70)
    print(
        f"Loading: "
        f"{file_path.name}"
    )

    try:

        if extension == ".pdf":

            loader = PyPDFLoader(
                str(file_path)
            )

        elif extension == ".txt":

            loader = TextLoader(
                str(file_path),
                encoding="utf-8",
            )

        elif extension == ".docx":

            loader = Docx2txtLoader(
                str(file_path)
            )

        elif extension == ".md":

            loader = TextLoader(
                str(file_path),
                encoding="utf-8",
            )

        else:

            return []

        documents = loader.load()

        print(
            f"Loaded "
            f"{len(documents)} "
            f"pages/sections."
        )

        return documents

    except Exception as e:

        print(
            f"ERROR loading "
            f"{file_path.name}: {e}"
        )

        return []


# =========================================================
# FIND ALL DOCUMENTS
# =========================================================

files = []

for file_path in DATA_DIR.rglob("*"):

    if not file_path.is_file():
        continue

    if (
        file_path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):
        continue

    files.append(file_path)


files.sort()


if not files:

    raise FileNotFoundError(
        f"""
No supported documents were found in:

{DATA_DIR}

Supported:

PDF
TXT
DOCX
MD
"""
    )


print(
    f"Found {len(files)} document(s)."
)

print()

for file_path in files:

    print(
        "  ✓",
        file_path.relative_to(
            DATA_DIR
        )
    )


# =========================================================
# LOAD DOCUMENTS
# =========================================================

all_documents = []

print()
print("=" * 80)
print("LOADING DOCUMENTS")
print("=" * 80)


for file_path in files:

    loaded = load_file(
        file_path
    )

    relative_path = str(
        file_path.relative_to(
            DATA_DIR
        )
    )

    document_type = (
        determine_document_type(
            file_path.name
        )
    )

    for document in loaded:

        document.metadata[
            "file"
        ] = file_path.name

        document.metadata[
            "file_path"
        ] = relative_path

        document.metadata[
            "extension"
        ] = file_path.suffix.lower()

        document.metadata[
            "document_type"
        ] = document_type

        all_documents.append(
            document
        )


if not all_documents:

    raise RuntimeError(
        "Documents were found but no text could be extracted."
    )


print()
print(
    f"Total pages/sections loaded: "
    f"{len(all_documents)}"
)


# =========================================================
# CLEAN DOCUMENTS
# =========================================================

print()
print("=" * 80)
print("CLEANING DOCUMENTS")
print("=" * 80)


clean_documents = []

for document in all_documents:

    text = clean_text(
        document.page_content
    )

    if not text:
        continue

    document.page_content = text

    clean_documents.append(
        document
    )


print(
    f"Usable pages/sections: "
    f"{len(clean_documents)}"
)


# =========================================================
# CHUNKING
# =========================================================

print()
print("=" * 80)
print("CREATING LEGAL TEXT CHUNKS")
print("=" * 80)


text_splitter = (
    RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        length_function=len,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "。 ",
            "; ",
            ": ",
            " ",
            "",
        ],
    )
)


chunks = (
    text_splitter.split_documents(
        clean_documents
    )
)


print(
    f"Created {len(chunks)} chunks."
)


# =========================================================
# PREPARE CHROMADB RECORDS
# =========================================================

documents = []

metadatas = []

ids = []


# Keep article tracking
# separately for each file.

current_articles = {}

current_chapters = {}


for index, chunk in enumerate(chunks):

    text = clean_text(
        chunk.page_content
    )

    if not text:
        continue

    original_file = (
        chunk.metadata.get(
            "file",
            "unknown",
        )
    )

    file_path = (
        chunk.metadata.get(
            "file_path",
            original_file,
        )
    )

    extension = (
        chunk.metadata.get(
            "extension",
            "",
        )
    )

    document_type = (
        chunk.metadata.get(
            "document_type",
            "Legal Document",
        )
    )

    page_number = (
        chunk.metadata.get(
            "page",
            0,
        )
    )

    try:

        page_number = (
            int(page_number) + 1
        )

    except Exception:

        page_number = 1


    # =====================================================
    # ARTICLE
    # =====================================================

    detected_article = (
        extract_article(text)
    )

    if detected_article:

        current_articles[
            file_path
        ] = detected_article

    article = (
        current_articles.get(
            file_path
        )
    )


    # =====================================================
    # CHAPTER
    # =====================================================

    detected_chapter = (
        extract_chapter(text)
    )

    if detected_chapter:

        current_chapters[
            file_path
        ] = detected_chapter

    chapter = (
        current_chapters.get(
            file_path
        )
    )


    # =====================================================
    # LANGUAGE
    # =====================================================

    language = detect_language(
        text
    )


    # =====================================================
    # METADATA
    # =====================================================

    metadata = {

        "source": (
            "Tanzania Legal Document"
        ),

        "document_type": (
            document_type
        ),

        "title": (
            Path(
                original_file
            ).stem
        ),

        "language": (
            language
        ),

        "page": (
            page_number
        ),

        "file": (
            original_file
        ),

        "file_path": (
            file_path
        ),

        "extension": (
            extension
        ),

        "article": (
            article
            if article
            else "unknown"
        ),

        "chapter": (
            chapter
            if chapter
            else "unknown"
        ),

        "chunk": (
            index
        ),

    }


    # =====================================================
    # STABLE CHUNK ID
    # =====================================================

    unique_text = (
        f"{file_path}|"
        f"{page_number}|"
        f"{index}|"
        f"{text}"
    )

    text_hash = hashlib.sha256(
        unique_text.encode(
            "utf-8"
        )
    ).hexdigest()[:24]


    chunk_id = (
        f"doc_{text_hash}"
    )


    documents.append(
        text
    )

    metadatas.append(
        metadata
    )

    ids.append(
        chunk_id
    )


print()
print(
    f"Prepared {len(documents)} "
    f"documents for embedding."
)


# =========================================================
# CREATE EMBEDDING FUNCTION
# =========================================================

print()
print("=" * 80)
print("LOADING EMBEDDING MODEL")
print("=" * 80)

print()
print(
    "Embedding model:"
)

print(
    EMBEDDING_MODEL
)

print()
print(
    "The first run may download the "
    "model from Hugging Face."
)

print()


embedding_function = (
    SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
)


print()
print(
    "✓ Embedding model loaded."
)


# =========================================================
# CONNECT TO CHROMADB
# =========================================================

print()
print("=" * 80)
print("CONNECTING TO CHROMADB")
print("=" * 80)


chroma_client = (
    chromadb.PersistentClient(
        path=str(
            CHROMA_DIR
        )
    )
)


# =========================================================
# CREATE / GET COLLECTION
# =========================================================

collection = (
    chroma_client.get_or_create_collection(

        name=COLLECTION_NAME,

        embedding_function=(
            embedding_function
        ),

        metadata={
            "description": (
                "Tanzania Constitution "
                "and Legal Documents"
            ),

            "embedding_model": (
                EMBEDDING_MODEL
            ),

            "language": (
                "Swahili and English"
            ),
        },
    )
)


# =========================================================
# REMOVE OLD DATA
# =========================================================

print()
print("=" * 80)
print("CLEARING OLD VECTOR DATA")
print("=" * 80)


existing = collection.get(
    include=[]
)

existing_ids = (
    existing.get("ids")
    or []
)


if existing_ids:

    print(
        f"Deleting "
        f"{len(existing_ids)} "
        f"old chunks..."
    )

    collection.delete(
        ids=existing_ids
    )

else:

    print(
        "No previous chunks found."
    )


# =========================================================
# INSERT + EMBED
# =========================================================

print()
print("=" * 80)
print("EMBEDDING AND INSERTING DOCUMENTS")
print("=" * 80)

print()

BATCH_SIZE = 100


for start in range(
    0,
    len(documents),
    BATCH_SIZE,
):

    end = min(
        start + BATCH_SIZE,
        len(documents),
    )


    batch_documents = (
        documents[start:end]
    )

    batch_metadatas = (
        metadatas[start:end]
    )

    batch_ids = (
        ids[start:end]
    )


    # Chroma automatically sends
    # these documents through the
    # explicit embedding function.

    collection.upsert(

        documents=batch_documents,

        metadatas=batch_metadatas,

        ids=batch_ids,
    )


    print(
        f"Embedded + inserted "
        f"{start + 1}-{end}"
    )


# =========================================================
# VERIFY
# =========================================================

database_count = (
    collection.count()
)


print()
print("=" * 80)
print("RAG DATABASE INGESTION COMPLETE")
print("=" * 80)

print()

print(
    f"Documents found       : "
    f"{len(files)}"
)

print(
    f"Pages/sections        : "
    f"{len(clean_documents)}"
)

print(
    f"Chunks created        : "
    f"{len(documents)}"
)

print(
    f"Database count        : "
    f"{database_count}"
)

print(
    f"Embedding model       : "
    f"{EMBEDDING_MODEL}"
)

print(
    f"Chunk size            : "
    f"{CHUNK_SIZE}"
)

print(
    f"Chunk overlap         : "
    f"{CHUNK_OVERLAP}"
)

print(
    f"Collection            : "
    f"{COLLECTION_NAME}"
)

print(
    f"ChromaDB              : "
    f"{CHROMA_DIR}"
)

print()


print(
    "Documents ingested:"
)

for file_path in files:

    print(
        f"  ✓ "
        f"{file_path.relative_to(DATA_DIR)}"
    )


print()

print(
    "You can add more documents to "
    "the data folder and run fill_db.py again."
)

print("=" * 80)