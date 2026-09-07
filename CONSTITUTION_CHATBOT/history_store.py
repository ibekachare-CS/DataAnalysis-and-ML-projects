"""
Chat History Store.

Persists chat sessions and messages to a local
SQLite database so that conversations survive
page reloads and server restarts, similar to how
ChatGPT keeps a sidebar of past conversations.

This module is intentionally dependency-free
(uses the Python standard library `sqlite3`)
so it does not add any new requirements to the
project.
"""

import sqlite3
import uuid
import datetime
import threading

from config import (
    HISTORY_DB_PATH,
)


# =========================================================
# CONNECTION HELPERS
# =========================================================

# SQLite connections are cheap to open and this app is
# a small single-instance service, so we open a fresh
# connection per operation rather than pooling one.
# A lock keeps writes serialized, which is all SQLite
# needs to stay safe across FastAPI's threadpool.

_write_lock = threading.Lock()


def _connect() -> sqlite3.Connection:

    conn = sqlite3.connect(
        HISTORY_DB_PATH,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    return conn


def _now() -> str:

    return (
        datetime.datetime.utcnow().isoformat()
        + "Z"
    )


# =========================================================
# SCHEMA
# =========================================================

def init_db() -> None:

    with _write_lock:

        conn = _connect()

        try:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id          TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES sessions (id)
                        ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_messages_session_id
                ON messages (session_id)
                """
            )

            conn.commit()

        finally:

            conn.close()


# =========================================================
# TITLE GENERATION
# =========================================================

def _make_title(first_message: str) -> str:

    text = " ".join(
        first_message.strip().split()
    )

    if not text:
        return "New chat"

    max_len = 48

    if len(text) <= max_len:
        return text

    return text[:max_len].rstrip() + "…"


# =========================================================
# SESSIONS
# =========================================================

def create_session(
    title: str | None = None,
) -> dict:

    session_id = str(
        uuid.uuid4()
    )

    now = _now()

    resolved_title = (
        title.strip()
        if title and title.strip()
        else "New chat"
    )

    with _write_lock:

        conn = _connect()

        try:

            conn.execute(
                """
                INSERT INTO sessions
                    (id, title, created_at, updated_at)
                VALUES
                    (?, ?, ?, ?)
                """,
                (
                    session_id,
                    resolved_title,
                    now,
                    now,
                ),
            )

            conn.commit()

        finally:

            conn.close()

    return {
        "id": session_id,
        "title": resolved_title,
        "created_at": now,
        "updated_at": now,
    }


def list_sessions() -> list[dict]:

    conn = _connect()

    try:

        rows = conn.execute(
            """
            SELECT
                s.id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count
            FROM sessions s
            LEFT JOIN messages m
                ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        conn.close()


def session_exists(session_id: str) -> bool:

    conn = _connect()

    try:

        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

        return row is not None

    finally:

        conn.close()


def get_session(session_id: str) -> dict | None:

    conn = _connect()

    try:

        session_row = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()

        if session_row is None:
            return None

        message_rows = conn.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        ).fetchall()

        session = dict(session_row)

        session["messages"] = [
            dict(row)
            for row in message_rows
        ]

        return session

    finally:

        conn.close()


def rename_session(
    session_id: str,
    title: str,
) -> bool:

    clean_title = title.strip()

    if not clean_title:
        return False

    with _write_lock:

        conn = _connect()

        try:

            cursor = conn.execute(
                """
                UPDATE sessions
                SET title = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    clean_title,
                    _now(),
                    session_id,
                ),
            )

            conn.commit()

            return cursor.rowcount > 0

        finally:

            conn.close()


def delete_session(session_id: str) -> bool:

    with _write_lock:

        conn = _connect()

        try:

            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,),
            )

            conn.commit()

            return cursor.rowcount > 0

        finally:

            conn.close()


def touch_session(session_id: str) -> None:

    with _write_lock:

        conn = _connect()

        try:

            conn.execute(
                """
                UPDATE sessions
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    _now(),
                    session_id,
                ),
            )

            conn.commit()

        finally:

            conn.close()


# =========================================================
# MESSAGES
# =========================================================

def add_message(
    session_id: str,
    role: str,
    content: str,
) -> dict:

    message_id = str(
        uuid.uuid4()
    )

    now = _now()

    with _write_lock:

        conn = _connect()

        try:

            conn.execute(
                """
                INSERT INTO messages
                    (id, session_id, role, content, created_at)
                VALUES
                    (?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    now,
                ),
            )

            # Auto-title the session from the first
            # user message, and always bump
            # updated_at so the sidebar sorts by
            # most-recently-active chat first.

            row = conn.execute(
                """
                SELECT title, (
                    SELECT COUNT(*) FROM messages
                    WHERE session_id = ?
                ) AS total
                FROM sessions
                WHERE id = ?
                """,
                (
                    session_id,
                    session_id,
                ),
            ).fetchone()

            new_title = None

            if (
                row is not None
                and row["title"] == "New chat"
                and role == "user"
                and row["total"] <= 1
            ):

                new_title = _make_title(
                    content
                )

                conn.execute(
                    """
                    UPDATE sessions
                    SET title = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        new_title,
                        now,
                        session_id,
                    ),
                )

            else:

                conn.execute(
                    """
                    UPDATE sessions
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        session_id,
                    ),
                )

            conn.commit()

        finally:

            conn.close()

    return {
        "id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": now,
    }
