from __future__ import annotations

import json
import os
import re

from openai import OpenAI


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat endpoint.

    Points at local Ollama by default. Pointing it at a remote GPU host is just a
    different base_url / model (constructor args, or the LLM_BASE_URL / LLM_MODEL env
    vars) - no code changes needed.

    Context length is a SERVER setting (OLLAMA_CONTEXT_LENGTH on the Ollama host):
    Ollama's OpenAI-compatible endpoint ignores a per-request num_ctx.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        trace_limit: int = 300,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = model or os.environ.get("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
        # Explicit limits. The SDK default is a 10-minute timeout plus 2 silent retries, which
        # against a remote GPU means one hung request stalls a run for 10 minutes and retried
        # requests quietly inflate the timings we're trying to measure.
        if timeout is None:
            timeout = float(os.environ.get("LLM_TIMEOUT_SEC", "180"))
        if max_retries is None:
            max_retries = int(os.environ.get("LLM_MAX_RETRIES", "1"))
        self.client = OpenAI(
            base_url=self.base_url, api_key="not-needed", timeout=timeout, max_retries=max_retries
        )
        # Every raw request/response passes through here - this is the debug trace.
        self.trace: list[dict] = []
        self.trace_limit = trace_limit

    def chat(self, messages: list[dict], tag: str | None = None) -> tuple[str, dict]:
        """One raw call. `messages` is the FULL conversation so far - system, then
        alternating user/assistant turns. The server has no memory; this list IS the memory.

        Returns (text, usage) where usage is {prompt_tokens, completion_tokens, total_tokens}
        (empty dict if the backend didn't report it).
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""
        self._record(tag, messages, resp, text)
        usage = resp.model_dump().get("usage") or {}
        return text, usage

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

    def chat_json(
        self, messages: list[dict], retries: int = 2, tag: str | None = None
    ) -> tuple[dict, str, dict]:
        """Call the model and parse a JSON object from its reply.

        Returns (parsed_dict, raw_text, usage) - the caller appends raw_text to its own
        history as the assistant turn. On a parse failure we retry using a scratch
        copy of the messages so a failed attempt doesn't pollute the caller's real
        conversation history.
        """
        last_error = None
        working = list(messages)
        for attempt in range(retries + 1):
            retry_tag = tag if attempt == 0 else f"{tag}-json-retry{attempt}"
            text, usage = self.chat(working, tag=retry_tag)
            try:
                return _extract_json(text), text, usage
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
