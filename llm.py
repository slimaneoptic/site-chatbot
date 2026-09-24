"""Minimal OpenRouter client: Gemini 3.8 Flash first, automatic fallback, JSON output and streaming.

Set OPENROUTER_API_KEY. Optional OPENROUTER_MODELS="model-a,model-b" overrides the order,
and OPENROUTER_REASONING=low|medium|high overrides how long thinking models may think.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterator

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
# Tried in order: when one fails, is busy, or the account has no credits left (paid models),
# the next one takes over automatically. Chosen with eval_assistant.py on 2026-09-24.
DEFAULT_MODELS = [
    "google/gemini-3.8-flash",  # 18/18 correct in both runs, ~2.8 s median with low thinking, ~$0.003 per request
    "google/gemini-3.1-flash-lite",  # paid backup: 17–18/18, ~2 s
    "nex-agi/nex-n2.5-pro:free",  # free backups (used when credits run out): accurate but slow (~1 min)
    "nvidia/nemotron-3-super-120b-a12b:free",
]
# Thinking effort per model, only where it was measured; other models keep their own default.
REASONING = {"google/gemini-3.8-flash": "low"}  # low: same accuracy, about twice as fast as the default
RETRYABLE = {402, 404, 408, 429, 500, 502, 503, 504}  # 402 = paid model but no credits left: move on to the next model


class LLMError(Exception):
    pass


def api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("openrouter_key")


def available() -> bool:
    return bool(api_key())


def reasoning(model: str) -> dict:
    effort = os.environ.get("OPENROUTER_REASONING") or REASONING.get(model, "")
    return {"reasoning": {"effort": effort}} if effort else {}


def models() -> list[str]:
    custom = os.environ.get("OPENROUTER_MODELS", "")
    return [m.strip() for m in custom.split(",") if m.strip()] or DEFAULT_MODELS


def _headers() -> dict:
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json", "X-Title": "Slimane portfolio demos"}


def _clean_json(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    return fenced.group(1) if fenced else text


def _failure(errors: list[str]) -> LLMError:
    out_of_credits = errors and all("credits" in e for e in errors if ":free" not in e)
    reason = "The AI account has no credits left and the free backups are busy. " if out_of_credits else "All AI models are busy right now. "
    return LLMError(reason + "; ".join(errors))


def complete_json(messages: list[dict], schema: dict, name: str = "result", timeout: int = 60) -> tuple[dict, str]:
    """Return (parsed JSON, model used). Tries each model until one returns valid JSON."""
    if not available():
        raise LLMError("No OpenRouter API key configured.")
    errors = []
    for model in models():
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": 4000,  # thinking models spend tokens before the answer
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
            "provider": {"require_parameters": True},
            **reasoning(model),
        }
        try:
            r = requests.post(URL, headers=_headers(), json=body, timeout=timeout)
        except requests.RequestException as err:
            errors.append(f"{model}: {type(err).__name__}")
            continue
        if r.status_code == 401:
            raise LLMError("The OpenRouter API key was rejected.")
        if r.status_code in RETRYABLE:
            errors.append(f"{model}: {'no OpenRouter credits' if r.status_code == 402 else r.status_code}")
            continue
        try:
            data = json.loads(r.text, strict=False)
            content = data["choices"][0]["message"]["content"] or ""
            return json.loads(_clean_json(content), strict=False), model
        except (ValueError, KeyError, IndexError, TypeError):
            errors.append(f"{model}: unusable answer")
    raise _failure(errors)


def stream_text(messages: list[dict], timeout: int = 90) -> Iterator[str]:
    """Yield answer text as it streams. Falls back to the next model if one fails before any text."""
    if not available():
        raise LLMError("No OpenRouter API key configured.")
    errors = []
    for model in models():
        body = {"model": model, "messages": messages, "stream": True, "max_tokens": 4000, **reasoning(model)}
        try:
            r = requests.post(URL, headers=_headers(), json=body, stream=True, timeout=timeout)
        except requests.RequestException as err:
            errors.append(f"{model}: {type(err).__name__}")
            continue
        if r.status_code == 401:
            raise LLMError("The OpenRouter API key was rejected.")
        if r.status_code != 200:
            errors.append(f"{model}: {'no OpenRouter credits' if r.status_code == 402 else r.status_code}")
            r.close()
            continue
        started = False
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue  # blank lines and ": keep-alive" comments
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload, strict=False)
            except ValueError:
                continue
            if "error" in chunk:
                if started:
                    yield "\n\n(The answer was interrupted. Please ask again.)"
                    return
                errors.append(f"{model}: {chunk['error'].get('code')}")
                break
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                started = True
                yield delta
        r.close()
        if started:
            return
    raise _failure(errors)
