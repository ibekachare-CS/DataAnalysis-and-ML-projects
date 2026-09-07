"""
Tanzania Constitution / Legal RAG Engine.

Pipeline:

User Question
      ↓
Query preparation
      ↓
Multilingual embedding
      ↓
ChromaDB semantic retrieval
      ↓
Relevant legal chunks
      ↓
Context construction
      ↓
Gemini
      ↓
Answer
"""


import re
import requests
import chromadb

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    N_RESULTS,
    RETRIEVAL_CANDIDATES,
    EMBEDDING_MODEL,
)

from prompts import (
    build_system_prompt,
)

from gemini_client import (
    call_gemini_llm,
    stream_gemini_llm,
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

print(
    f"Loading embedding model: "
    f"{EMBEDDING_MODEL}"
)


embedding_function = (
    SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
)


print(
    "✓ Embedding model ready."
)


# =========================================================
# CHROMADB
# =========================================================

chroma_client = (
    chromadb.PersistentClient(
        path=CHROMA_PATH
    )
)


collection = (
    chroma_client.get_or_create_collection(

        name=COLLECTION_NAME,

        embedding_function=(
            embedding_function
        ),
    )
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(
    text: str
) -> str:

    if not text:
        return ""

    text = re.sub(
        r"\*\*",
        "",
        text,
    )

    text = re.sub(
        r"\*",
        "",
        text,
    )

    text = re.sub(
        r"__",
        "",
        text,
    )

    text = re.sub(
        r"_",
        "",
        text,
    )

    text = re.sub(
        r"^#+\s*",
        "",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(
        r"```[\s\S]*?```",
        "",
        text,
    )

    text = re.sub(
        r"`",
        "",
        text,
    )

    text = re.sub(
        r"~~",
        "",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# QUERY PREPARATION
# =========================================================

def build_search_query(
    user_query: str,
    history: list[dict] | None = None,
) -> str:

    query = user_query.strip()

    if not query:
        return query

    # For short follow-up questions ("what about article 12?",
    # "and duties?"), pull in the user's previous question so
    # retrieval still finds the right topic instead of just
    # searching for the follow-up fragment alone.

    context_prefix = ""

    if history:

        previous_user_messages = [
            item.get("content", "").strip()
            for item in history
            if item.get("role") == "user"
            and item.get("content", "").strip()
        ]

        if previous_user_messages:

            context_prefix = (
                previous_user_messages[-1]
                + "\n\n"
            )

    return (
        f"{context_prefix}{query}\n\n"
        "Tanzania Constitution "
        "Katiba ya Jamhuri ya Muungano wa Tanzania "
        "Article Ibara "
        "constitutional provision "
        "legal provision "
        "haki mamlaka wajibu sheria"
    )


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_rag_context(
    user_query: str,
    n_results: int = N_RESULTS,
    history: list[dict] | None = None,
) -> list[dict]:

    try:

        if collection.count() == 0:

            print(
                "WARNING: ChromaDB collection is empty."
            )

            return []


        search_query = (
            build_search_query(
                user_query,
                history=history,
            )
        )


        # Retrieve more candidates first.
        #
        # This gives us a larger pool from
        # which to select the strongest chunks.

        candidate_count = max(
            n_results,
            RETRIEVAL_CANDIDATES,
        )


        results = collection.query(

            query_texts=[
                search_query
            ],

            n_results=candidate_count,

            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )


        documents = (
            results.get(
                "documents"
            )
            or [[]]
        )[0]


        metadatas = (
            results.get(
                "metadatas"
            )
            or [[]]
        )[0]


        distances = (
            results.get(
                "distances"
            )
            or [[]]
        )[0]


        retrieved = []


        for index, document in enumerate(
            documents
        ):

            if not document:
                continue


            metadata = {}

            if index < len(
                metadatas
            ):

                metadata = (
                    metadatas[index]
                    or {}
                )


            distance = None

            if index < len(
                distances
            ):

                distance = (
                    distances[index]
                )


            metadata = dict(
                metadata
            )


            if distance is not None:

                metadata[
                    "distance"
                ] = float(
                    distance
                )


            retrieved.append(

                {
                    "text": document,

                    "metadata": metadata,
                }

            )


        # -------------------------------------------------
        # REMOVE DUPLICATE TEXT
        # -------------------------------------------------

        unique = []

        seen = set()


        for item in retrieved:

            text = item[
                "text"
            ].strip()

            fingerprint = (
                text.lower()
            )

            if fingerprint in seen:
                continue

            seen.add(
                fingerprint
            )

            unique.append(
                item
            )


        # -------------------------------------------------
        # FINAL NUMBER OF RESULTS
        # -------------------------------------------------

        return unique[
            :n_results
        ]


    except Exception as e:

        print(
            f"ChromaDB retrieval error: "
            f"{e}"
        )

        return []


# =========================================================
# DEBUG RETRIEVAL
# =========================================================

def print_retrieved_context(
    retrieved_docs: list[dict]
):

    print()
    print("=" * 80)
    print("RETRIEVED LEGAL CONTEXT")
    print("=" * 80)


    if not retrieved_docs:

        print(
            "NO DOCUMENTS RETRIEVED"
        )

        return


    for i, item in enumerate(
        retrieved_docs,
        start=1,
    ):

        metadata = item.get(
            "metadata",
            {},
        )

        text = item.get(
            "text",
            "",
        )


        print()
        print(
            f"--- RESULT {i} ---"
        )


        print(
            "File:",
            metadata.get(
                "file",
                "unknown",
            ),
        )


        print(
            "Document type:",
            metadata.get(
                "document_type",
                "unknown",
            ),
        )


        print(
            "Language:",
            metadata.get(
                "language",
                "unknown",
            ),
        )


        print(
            "Article:",
            metadata.get(
                "article",
                "unknown",
            ),
        )


        print(
            "Chapter:",
            metadata.get(
                "chapter",
                "unknown",
            ),
        )


        print(
            "Page:",
            metadata.get(
                "page",
                "unknown",
            ),
        )


        print(
            "Distance:",
            metadata.get(
                "distance",
                "unknown",
            ),
        )


        print()
        print(
            text[:1500]
        )


    print()
    print("=" * 80)


# =========================================================
# NORMAL RESPONSE
# =========================================================

def get_rag_response(
    user_query: str,
    history: list[dict] | None = None,
) -> str:

    retrieved_docs = (
        retrieve_rag_context(
            user_query,
            history=history,
        )
    )


    print(
        f"Retrieved "
        f"{len(retrieved_docs)} "
        f"legal chunks."
    )


    print_retrieved_context(
        retrieved_docs
    )


    system_prompt = (
        build_system_prompt(
            retrieved_docs,
            has_history=bool(history),
        )
    )


    try:

        response = (
            call_gemini_llm(

                system_prompt,

                user_query,

                history=history,
            )
        )


        return clean_text(
            response
        )


    except requests.exceptions.ConnectionError:

        return (
            "Error: Could not connect to "
            "Google Gemini. Check your internet "
            "connection and API configuration."
        )


    except requests.exceptions.Timeout:

        return (
            "Error: The Gemini request timed out. "
            "Please try again."
        )


    except Exception as e:

        print(
            f"RAG generation error: {e}"
        )

        return (
            f"Error getting response from Gemini: "
            f"{e}"
        )


# =========================================================
# STREAMING RESPONSE
# =========================================================

def get_rag_response_stream(
    user_query: str,
    history: list[dict] | None = None,
):

    retrieved_docs = (
        retrieve_rag_context(
            user_query,
            history=history,
        )
    )


    print(
        f"Retrieved "
        f"{len(retrieved_docs)} "
        f"legal chunks."
    )


    print_retrieved_context(
        retrieved_docs
    )


    system_prompt = (
        build_system_prompt(
            retrieved_docs,
            has_history=bool(history),
        )
    )


    sent_content = ""


    try:

        full_response = ""


        for chunk in (
            stream_gemini_llm(

                system_prompt,

                user_query,

                history=history,
            )
        ):

            if not chunk:
                continue


            full_response += chunk


            cleaned = clean_text(
                full_response
            )


            if len(cleaned) > len(
                sent_content
            ):

                new_content = (
                    cleaned[
                        len(sent_content):
                    ]
                )


                if new_content.strip():

                    yield new_content

                    sent_content = cleaned


    except requests.exceptions.ConnectionError:

        yield (
            "Error: Could not connect to "
            "Google Gemini."
        )


    except requests.exceptions.Timeout:

        yield (
            "Error: Gemini request timed out."
        )


    except Exception as e:

        print(
            f"RAG streaming error: "
            f"{e}"
        )

        yield (
            f"Error getting response from Gemini: "
            f"{e}"
        )