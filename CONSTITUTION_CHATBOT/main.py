import logging

from fastapi import FastAPI, HTTPException, Request

from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)

from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

import history_store

from rag_engine import (
    get_rag_response,
    get_rag_response_stream,
)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    __name__
)


# =========================================================
# CONVERSATION MEMORY
# =========================================================

# How many prior messages (user + assistant turns combined)
# are sent to the LLM as conversation context. Keeps the
# prompt bounded even in very long chats.

MAX_HISTORY_MESSAGES = 12


def _get_conversation_history(
    session_id: str,
) -> list[dict]:
    """
    Load the prior messages of a session, formatted for
    the LLM ({"role": ..., "content": ...}), most recent
    MAX_HISTORY_MESSAGES only.
    """

    session = history_store.get_session(
        session_id
    )

    if not session:
        return []

    messages = session["messages"][
        -MAX_HISTORY_MESSAGES:
    ]

    return [

        {
            "role": message["role"],
            "content": message["content"],
        }

        for message in messages

        if message["role"] in (
            "user",
            "assistant",
        )
    ]


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(

    title=(
        "Tanzania Constitution "
        "RAG Assistant"
    ),

    description=(
        "RAG assistant using the "
        "Tanzania Constitution."
    ),

    version="2.0.0",
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory="templates"
)


# =========================================================
# CHAT HISTORY STORAGE
# =========================================================

history_store.init_db()


# =========================================================
# REQUEST MODEL
# =========================================================

class QuestionRequest(
    BaseModel
):

    question: str

    # Optional: continue an existing chat.
    # If omitted, a new chat session is created
    # automatically and its id is returned to
    # the client so it can keep using it.

    session_id: str | None = None


class SessionCreateRequest(
    BaseModel
):

    title: str | None = None


class SessionRenameRequest(
    BaseModel
):

    title: str


# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home(
    request: Request
):

    return templates.TemplateResponse(

        request,

        "index.html",

        {},
    )


# =========================================================
# CHAT SESSIONS (HISTORY)
# =========================================================

@app.get("/sessions")
async def get_sessions():
    """
    List all chat sessions, most recently
    active first. Used to populate the
    sidebar chat history.
    """

    return history_store.list_sessions()


@app.post("/sessions")
async def create_session(
    payload: SessionCreateRequest
):
    """
    Start a new, empty chat session.
    """

    session = history_store.create_session(
        title=payload.title
    )

    return session


@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str
):
    """
    Fetch a session and its full message
    history, used when the user clicks a
    conversation in the sidebar.
    """

    session = history_store.get_session(
        session_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Chat session not found.",
        )

    return session


@app.patch("/sessions/{session_id}")
async def rename_session(
    session_id: str,
    payload: SessionRenameRequest,
):
    """
    Rename a chat session (e.g. the user
    edits the auto-generated title).
    """

    updated = history_store.rename_session(
        session_id,
        payload.title,
    )

    if not updated:

        raise HTTPException(
            status_code=404,
            detail="Chat session not found.",
        )

    return {"status": "ok"}


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str
):
    """
    Delete a chat session and all of its
    messages.
    """

    deleted = history_store.delete_session(
        session_id
    )

    if not deleted:

        raise HTTPException(
            status_code=404,
            detail="Chat session not found.",
        )

    return {"status": "ok"}


# =========================================================
# ASK
# =========================================================

@app.post("/ask")
async def ask(
    payload: QuestionRequest
):

    question = (
        payload.question.strip()
    )

    if not question:

        return JSONResponse(

            {
                "response":
                "Please enter a question."
            },

            status_code=400,
        )

    # Reuse the given session, or create a new
    # one automatically so every conversation is
    # saved to history by default.

    session_id = payload.session_id

    if not session_id or not history_store.session_exists(
        session_id
    ):

        session = history_store.create_session()

        session_id = session["id"]

        conversation_history = []

    else:

        # Grab the prior turns *before* the new
        # question is saved, so the model sees
        # everything that came before it, not the
        # question itself twice.

        conversation_history = (
            _get_conversation_history(
                session_id
            )
        )

    history_store.add_message(
        session_id,
        "user",
        question,
    )

    try:

        response = get_rag_response(
            question,
            history=conversation_history,
        )

        history_store.add_message(
            session_id,
            "assistant",
            response,
        )

        return JSONResponse(

            {
                "response": response,
                "session_id": session_id,
            }
        )

    except Exception as e:

        logger.exception(
            "RAG request failed"
        )

        return JSONResponse(

            {
                "response":
                "An error occurred while "
                "processing your question.",

                "session_id": session_id,
            },

            status_code=500,
        )


# =========================================================
# STREAM
# =========================================================

@app.post("/ask/stream")
async def ask_stream(
    payload: QuestionRequest
):

    question = (
        payload.question.strip()
    )

    if not question:

        return StreamingResponse(

            iter(
                [
                    "Please enter a question."
                ]
            ),

            media_type=(
                "text/plain; charset=utf-8"
            ),

            status_code=400,
        )

    session_id = payload.session_id

    if not session_id or not history_store.session_exists(
        session_id
    ):

        session = history_store.create_session()

        session_id = session["id"]

        conversation_history = []

    else:

        conversation_history = (
            _get_conversation_history(
                session_id
            )
        )

    history_store.add_message(
        session_id,
        "user",
        question,
    )

    async def stream_and_persist():

        full_response = ""

        try:

            for chunk in get_rag_response_stream(
                question,
                history=conversation_history,
            ):

                full_response += chunk

                yield chunk

        finally:

            # Save whatever was generated, even
            # if the client disconnected part
            # way through, so history stays
            # consistent with what the user saw.

            if full_response.strip():

                history_store.add_message(
                    session_id,
                    "assistant",
                    full_response,
                )

    return StreamingResponse(

        stream_and_persist(),

        media_type=(
            "text/plain; charset=utf-8"
        ),

        headers={

            "Cache-Control":
            "no-cache",

            "X-Accel-Buffering":
            "no",

            "X-Session-Id":
            session_id,
        },
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():

    return {

        "status": "ok",

        "service":
        "Tanzania Constitution RAG Assistant",

        "source":
        "Tanzania Constitution PDF",

        "vector_database":
        "ChromaDB",

        "llm":
        "Google Gemini",

        "voice":
        False,
    }


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="127.0.0.1",

        port=8000,

        reload=True,
    )