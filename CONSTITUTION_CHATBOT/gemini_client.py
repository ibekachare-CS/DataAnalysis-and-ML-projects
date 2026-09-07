"""
Google Gemini API client.

Uses Google's OpenAI-compatible API.
"""

import json
import requests

from config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
)


# =========================================================
# ENDPOINT
# =========================================================

CHAT_ENDPOINT = (
    f"{GEMINI_BASE_URL}/chat/completions"
)


# =========================================================
# HEADERS
# =========================================================

def _headers() -> dict:

    if not GEMINI_API_KEY:

        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add GEMINI_API_KEY to your .env file."
        )

    return {
        "Authorization": (
            f"Bearer {GEMINI_API_KEY}"
        ),
        "Content-Type": "application/json",
    }


# =========================================================
# NORMAL REQUEST
# =========================================================

def call_gemini_llm(
    system_prompt: str,
    user_query: str,
    history: list[dict] | None = None,
) -> str:

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    if history:

        messages.extend(
            history
        )

    messages.append(

        {
            "role": "user",
            "content": user_query,
        }
    )

    payload = {

        "model": GEMINI_MODEL,

        "messages": messages,

        "temperature": 0.1,

        "stream": False,
    }

    response = requests.post(

        CHAT_ENDPOINT,

        headers=_headers(),

        json=payload,

        timeout=GEMINI_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):

        raise RuntimeError(
            str(data["error"])
        )

    choices = (
        data.get("choices")
        or []
    )

    if not choices:

        raise RuntimeError(
            f"Unexpected Gemini response: {data}"
        )

    message = choices[0].get(
        "message",
        {}
    )

    content = message.get(
        "content"
    )

    if not content:

        raise RuntimeError(
            f"Gemini returned empty content: {data}"
        )

    return content


# =========================================================
# STREAMING REQUEST
# =========================================================

def stream_gemini_llm(
    system_prompt: str,
    user_query: str,
    history: list[dict] | None = None,
):

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
    ]

    if history:

        messages.extend(
            history
        )

    messages.append(

        {
            "role": "user",
            "content": user_query,
        }
    )

    payload = {

        "model": GEMINI_MODEL,

        "messages": messages,

        "temperature": 0.1,

        "stream": True,
    }

    with requests.post(

        CHAT_ENDPOINT,

        headers=_headers(),

        json=payload,

        timeout=(10, None),

        stream=True,

    ) as response:

        response.raise_for_status()

        for line in response.iter_lines(
            decode_unicode=True
        ):

            if not line:
                continue

            if not line.startswith(
                "data:"
            ):
                continue

            data_str = (
                line[len("data:"):].strip()
            )

            if data_str == "[DONE]":
                break

            try:

                data = json.loads(
                    data_str
                )

            except json.JSONDecodeError:

                continue

            if data.get("error"):

                raise RuntimeError(
                    str(data["error"])
                )

            choices = (
                data.get("choices")
                or []
            )

            if not choices:
                continue

            delta = choices[0].get(
                "delta",
                {}
            )

            content = delta.get(
                "content"
            )

            if content:

                yield content