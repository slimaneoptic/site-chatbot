"""Minimal OpenRouter client: Gemini 3.8 Flash first, automatic fallback, JSON output and streaming.

Set OPENROUTER_API_KEY. Optional OPENROUTER_MODELS="model-a,model-b" overrides the order,
and OPENROUTER_REASONING=low|medium|high overrides how long thinking models may think.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterator

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
RETRYABLE = {402, 403, 404, 408, 429, 500, 502, 503, 504}  # 402 = paid model but no credits left, 403 = moderation/region: next model
# USD per million tokens (input, output), live OpenRouter prices read 2026-10-13. Used only when
# the API does not return usage.cost. Unknown models count as free. Prices double on 2027-01-01.
PRICES = {
    "google/gemini-3.8-flash": (0.75, 3.75),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
}
TRANSLATE_MODEL = "google/gemini-3.1-flash-lite"
FIRST_TOKEN_S = 7.0  # a model that has written nothing after this long hands over to the next one (thinking spikes)
DROPPED_THINK_TOKENS = 1000  # billed thinking a model dropped at FIRST_TOKEN_S is assumed to have done (estimate)
INTERRUPTED = "\n\n(The answer was interrupted. Please ask again.)"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # OpenRouter usage.cost when present, else a PRICES estimate
    model: str = ""
    calls: int = 0  # model calls made (0 for no-match / cache / extractive)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def _chars_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _usage(model: str, raw: dict | None, messages: list[dict], output: str) -> Usage:
    """Real usage from the API when present, else a chars/4 estimate."""
    raw = raw or {}
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    if not isinstance(prompt, int):
        prompt = _chars_tokens("".join(str(m.get("content", "")) for m in messages))
    if not isinstance(completion, int):
        completion = _chars_tokens(output)
    cost = raw.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        cost = estimate_cost(model, prompt, completion)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, cost_usd=float(cost), model=model, calls=1)


class LLMError(Exception):
    pass


def api_key() -> str | None:
    """The OpenRouter key without surrounding spaces, newlines or quotes (a pasted key often has them)."""
    raw = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("openrouter_key") or ""
    key = raw.strip().strip("'\"").strip()
    return key or None


KEY_URL = "https://openrouter.ai/api/v1/key"
_KEY_STATE = {"state": "unchecked", "key": None}  # "ok" | "rejected" | "unchecked", for the key it was measured on


def key_state() -> str:
    """"ok" | "missing" | "rejected" | "unchecked" for the current key."""
    key = api_key()
    if not key:
        return "missing"
    return _KEY_STATE["state"] if _KEY_STATE["key"] == key else "unchecked"


def mark_key(state: str) -> None:
    _KEY_STATE["state"], _KEY_STATE["key"] = state, api_key()


def check_key(get=None, timeout: float = 5) -> str:
    """Ask OpenRouter whether the key is valid (GET /api/v1/key). Sets and returns the key state; never raises."""
    if not api_key():
        return "missing"
    try:
        r = (get or requests.get)(KEY_URL, headers=_headers(), timeout=timeout)
        if r.status_code == 200:
            mark_key("ok")
        elif r.status_code in (401, 403):
            mark_key("rejected")
    except Exception:  # noqa: BLE001 - unknown stays "unchecked"
        pass
    return key_state()


def check_key_async(get=None) -> None:
    import threading
    threading.Thread(target=check_key, args=(get,), daemon=True, name="key-check").start()


def available() -> bool:
    return bool(api_key()) and key_state() != "rejected"


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
            mark_key("rejected")
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


def stream_text(messages: list[dict], timeout: int = 90, *, max_tokens: int = 4000, deadline_s: float | None = None,
                on_usage: Callable[[Usage], None] | None = None, session_id: str | None = None) -> Iterator[str]:
    """Yield answer text as it streams. Falls back to the next model if one fails before any text.

    `deadline_s` stops trying further models after that many seconds (raises LLMError).
    `on_usage` is called once with the Usage of the model that answered, when the stream ends. A model dropped for
    writing nothing after FIRST_TOKEN_S was still billed for its prompt and thinking: that estimate is added to the
    usage (calls included), and when no model answers on_usage still gets it before LLMError is raised."""
    if not available():
        raise LLMError("No OpenRouter API key configured.")
    errors = []
    start = time.monotonic()
    dropped = Usage()  # models given up on after FIRST_TOKEN_S (billed anyway)
    for model in models():
        wait = timeout
        if deadline_s is not None:
            left = deadline_s - (time.monotonic() - start)
            if left <= 0:
                errors.append("time limit reached")
                break
            wait = max(1, min(timeout, int(left) + 1))
        body = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens,
                "usage": {"include": True}, **reasoning(model)}
        if session_id:
            body["user"] = session_id[:128]
        try:
            r = requests.post(URL, headers=_headers(), json=body, stream=True, timeout=wait)
        except requests.RequestException as err:
            errors.append(f"{model}: {type(err).__name__}")
            continue
        if r.status_code == 401:
            r.close()
            mark_key("rejected")
            raise LLMError("The OpenRouter API key was rejected.")
        if r.status_code != 200:
            errors.append(f"{model}: {'no OpenRouter credits' if r.status_code == 402 else r.status_code}")
            r.close()
            continue
        started = False
        out: list[str] = []
        raw_usage = None
        r.encoding = "utf-8"
        sent_at = time.monotonic()
        has_next = model != models()[-1]
        try:
            for line in r.iter_lines(decode_unicode=True):
                if not started and has_next and time.monotonic() - sent_at > FIRST_TOKEN_S:
                    errors.append(f"{model}: no text after {FIRST_TOKEN_S:g} s")
                    prompt = _chars_tokens("".join(str(m.get("content", "")) for m in messages))
                    dropped.prompt_tokens += prompt
                    dropped.completion_tokens += DROPPED_THINK_TOKENS
                    dropped.cost_usd += estimate_cost(model, prompt, DROPPED_THINK_TOKENS)
                    dropped.calls += 1
                    dropped.model = dropped.model or model
                    break
                if not line or not line.startswith("data:"):
                    continue  # blank lines and ": keep-alive" comments
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload, strict=False)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if isinstance(chunk.get("usage"), dict):
                    raw_usage = chunk["usage"]
                if "error" in chunk:
                    if started:
                        out.append(INTERRUPTED)
                        yield INTERRUPTED
                        return
                    err = chunk["error"]
                    errors.append(f"{model}: {err.get('code') if isinstance(err, dict) else 'error'}")
                    break
                delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    started = True
                    out.append(delta)
                    yield delta
        except (requests.RequestException, OSError) as err:  # dropped connection or read timeout mid-stream
            if started:
                out.append(INTERRUPTED)
                yield INTERRUPTED
                return
            errors.append(f"{model}: {type(err).__name__}")
        finally:
            r.close()
            if started and on_usage is not None:
                u = _usage(model, raw_usage, messages, "".join(out))
                if dropped.calls:
                    u = Usage(u.prompt_tokens + dropped.prompt_tokens, u.completion_tokens + dropped.completion_tokens,
                              u.cost_usd + dropped.cost_usd, u.model, u.calls + dropped.calls)
                on_usage(u)
        if started:
            return
    if dropped.calls and on_usage is not None:
        on_usage(dropped)
    raise _failure(errors)


def quick_json(messages: list[dict], schema: dict, *, model: str = TRANSLATE_MODEL, max_tokens: int = 250,
               timeout: float = 3, name: str = "result", post=None) -> tuple[dict, Usage]:
    """One small structured call to a fast model, no fallback chain (for extras like starter questions).
    Raises LLMError on any failure."""
    if not available():
        raise LLMError("No OpenRouter API key configured.")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}}
    try:
        r = (post or requests.post)(URL, headers=_headers(), json=body, timeout=timeout)
    except requests.RequestException as err:
        raise LLMError(type(err).__name__) from err
    if r.status_code == 401:
        mark_key("rejected")
    if r.status_code != 200:
        raise LLMError(f"{model}: {r.status_code}")
    try:
        data = json.loads(r.text, strict=False)
        content = data["choices"][0]["message"]["content"] or ""
        out = json.loads(_clean_json(content), strict=False)
    except (ValueError, KeyError, IndexError, TypeError) as err:
        raise LLMError(f"{model}: unusable answer") from err
    return out, _usage(model, data.get("usage"), messages, content)


_LANG_NAMES = {"ar": "Arabic", "latin": "English", "en": "English", "fr": "French", "es": "Spanish", "de": "German",
               "it": "Italian", "pt": "Portuguese"}


def translate_query(text: str, target_lang: str, *, on_usage: Callable[[Usage], None] | None = None, timeout: int = 10,
                    hint: str = "") -> str:
    """Translate a short search question into the site's language (Flash-Lite, max 80 tokens).
    `hint`: words the site uses (titles, frequent words), preferred where they fit, so the search finds them.
    Returns `text` unchanged on any failure; never raises."""
    if not text.strip() or not available():
        return text
    lang = _LANG_NAMES.get(target_lang, target_lang)
    messages = [
        {"role": "system", "content": f"Translate the user's search question into {lang}. Reply with the translation only, one line."
                                      + (f" Where they fit the meaning, use the words this website uses: {hint[:900]}." if hint else "")},
        {"role": "user", "content": text[:500]},
    ]
    try:
        r = requests.post(URL, headers=_headers(), json={"model": TRANSLATE_MODEL, "messages": messages, "max_tokens": 80,
                                                         "usage": {"include": True}}, timeout=timeout)
        if r.status_code != 200:
            return text
        data = json.loads(r.text, strict=False)
        content = (data["choices"][0]["message"]["content"] or "").strip()
        line = content.splitlines()[0].strip().strip('"\'«»“”') if content else ""
        if on_usage is not None:
            try:
                on_usage(_usage(TRANSLATE_MODEL, data.get("usage"), messages, content))
            except Exception:  # noqa: BLE001 - a bad callback must not break search
                pass
        if not line or len(line) > 4 * len(text) + 40:
            return text
        return line
    except Exception:  # noqa: BLE001 - any failure keeps the original question
        return text
