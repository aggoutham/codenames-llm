from __future__ import annotations

import json
import os
import re

from openai import OpenAI


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat endpoint.

    Points at local Ollama by default. Swapping to Azure later is just
    changing OLLAMA_BASE_URL / MODEL_NAME env vars to the vLLM endpoint -
    no code changes needed.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None, trace_limit: int = 300):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = model or os.environ.get("LLM_MODEL", "qwen2.5:3b")
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
        # Every raw request/response passes through here - this is the debug trace.
        self.trace: list[dict] = []
        self.trace_limit = trace_limit

    def chat(self, messages: list[dict], tag: str | None = None) -> str:
        """One raw call. `messages` is the FULL conversation so far - system, then
        alternating user/assistant turns. The server has no memory; this list IS the memory."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""
        self._record(tag, messages, resp, text)
        return text

    def _record(self, tag: str | None, messages: list[dict], resp, text: str) -> None:
        self.trace.append(
            {
                "tag": tag or "?",
                "model": self.model,
                "messages": list(messages),
                "raw_response": resp.model_dump(),
                "text": text,
            }
        )
        if len(self.trace) > self.trace_limit:
            del self.trace[: len(self.trace) - self.trace_limit]

    def chat_json(self, messages: list[dict], retries: int = 2, tag: str | None = None) -> tuple[dict, str]:
        """Call the model and parse a JSON object from its reply.

        Returns (parsed_dict, raw_text) - the caller appends raw_text to its own
        history as the assistant turn. On a parse failure we retry using a scratch
        copy of the messages so a failed attempt doesn't pollute the caller's real
        conversation history.
        """
        last_error = None
        working = list(messages)
        for attempt in range(retries + 1):
            retry_tag = tag if attempt == 0 else f"{tag}-json-retry{attempt}"
            text = self.chat(working, tag=retry_tag)
            try:
                return _extract_json(text), text
            except ValueError as e:
                last_error = e
                working = working + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "That wasn't valid JSON. Reply with ONLY a single JSON object."},
                ]
        raise last_error


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"could not extract JSON from model output: {text!r}")
