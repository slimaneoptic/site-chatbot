"""Minimal OpenRouter client: Gemini 3.8 Flash first, automatic fallback, JSON output and streaming.

Set OPENROUTER_API_KEY. Optional OPENROUTER_MODELS="model-a,model-b" overrides the order,
and OPENROUTER_REASONING=low|medium|high overrides how long thinking models may think.
Optional ANTHROPIC_API_KEY adds Claude Haiku 5.5 (Anthropic API) to the chain: after the paid OpenRouter models,
before the free ones (so it answers when OpenRouter credit runs out), and as the backup of the small extras
(translation, starter questions). ANTHROPIC_DAILY_CAP (USD, default 1.00) is its hard spend cap per day and
instance, counted from each response's usage. Without ANTHROPIC_API_KEY nothing changes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import requests

import budget

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

# Claude through the Anthropic API, used only when ANTHROPIC_API_KEY is set (see the module docstring).
CLAUDE_MODEL = "claude-haiku-5-5"
CLAUDE_EFFORT = "low"  # fast first words; thinking stays on (adaptive) for answers and extraction, off for the extras
CLAUDE_DAILY_CAP = 1.0  # USD per day and instance, unless ANTHROPIC_DAILY_CAP says otherwise
# USD per million tokens: (input, output, 5-minute cache write, cache read) for prompts up to 100,000 tokens, then
# for longer prompts. Official Anthropic prices (platform.claude.com/docs/en/about-claude/pricing), read 2026-10-15.
CLAUDE_PRICES = ((0.10, 0.50, 0.125, 0.01), (0.50, 2.50, 0.625, 0.05))
CLAUDE_LONG_PROMPT = 100_000
PRICES[CLAUDE_MODEL] = CLAUDE_PRICES[0][:2]  # estimate_cost() for Claude too (used by the extraction estimate)


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


def openrouter_ok() -> bool:
    """An OpenRouter key is set and was not rejected."""
    return bool(api_key()) and key_state() != "rejected"


def available() -> bool:
    """True when some model can be asked: OpenRouter (a key that was not rejected) or Claude (ANTHROPIC_API_KEY)."""
    return openrouter_ok() or claude_on()


def reasoning(model: str) -> dict:
    effort = os.environ.get("OPENROUTER_REASONING") or REASONING.get(model, "")
    return {"reasoning": {"effort": effort}} if effort else {}


def models() -> list[str]:
    custom = os.environ.get("OPENROUTER_MODELS", "")
    return [m.strip() for m in custom.split(",") if m.strip()] or DEFAULT_MODELS


def chain() -> list[str]:
    """The models stream_text and complete_json try, in order: the OpenRouter list, with Claude (only when
    ANTHROPIC_API_KEY is set) after the paid models and before the first free one (":free", about a minute each)."""
    order = models()
    if not claude_on():
        return order
    i = next((k for k, m in enumerate(order) if m.endswith(":free")), len(order))
    return order[:i] + [CLAUDE_MODEL] + order[i:]


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


# ---------------------------------------------------------------- Claude (Anthropic API), only with ANTHROPIC_API_KEY

_CLAUDE_STATE: dict = {"rejected": None, "key": None, "client": None}  # a key the API refused; the reused client
CLAUDE_CAP = budget.DailyCap(CLAUDE_DAILY_CAP)  # one per process, like the Budget counters


def anthropic_key() -> str | None:
    """ANTHROPIC_API_KEY without surrounding spaces, newlines or quotes; None when unset (Claude is then never used)."""
    raw = os.environ.get("ANTHROPIC_API_KEY") or ""
    key = raw.strip().strip("'\"").strip()
    return key or None


def claude_on() -> bool:
    key = anthropic_key()
    return bool(key) and key != _CLAUDE_STATE["rejected"]


def claude_cap() -> budget.DailyCap:
    """The process-wide Claude cap, set from ANTHROPIC_DAILY_CAP (USD, default 1.00) on each use."""
    try:
        usd = float(os.environ.get("ANTHROPIC_DAILY_CAP", "").strip() or CLAUDE_DAILY_CAP)
    except ValueError:
        usd = CLAUDE_DAILY_CAP
    CLAUDE_CAP.daily_usd = max(0.0, usd) if math.isfinite(usd) else CLAUDE_DAILY_CAP
    return CLAUDE_CAP


def claude_snapshot() -> dict:
    """For /healthz: the Claude model, whether it is on, and today's spend against the cap (never the key)."""
    return {"model": CLAUDE_MODEL, "on": claude_on(), **claude_cap().snapshot()}


def _claude_client():
    """One SDK client per key, reused so the connection stays warm. No retries inside the SDK: the next model in the
    chain is the retry. The SDK is imported only here, when a Claude call is about to be made."""
    try:
        import anthropic
    except ImportError as err:
        raise LLMError(f"{CLAUDE_MODEL}: the anthropic package is not installed") from err
    key = anthropic_key()
    if _CLAUDE_STATE["client"] is None or _CLAUDE_STATE["key"] != key:
        _CLAUDE_STATE["client"], _CLAUDE_STATE["key"] = anthropic.Anthropic(api_key=key, max_retries=0), key
    return _CLAUDE_STATE["client"]


def _claude_errors() -> tuple:
    """What a failed Claude call raises: SDK errors, and network errors from inside a stream (call after
    _claude_client(), which made sure the SDK is there)."""
    import anthropic
    import httpx2
    return (anthropic.AnthropicError, httpx2.HTTPError, OSError, ValueError)


def _claude_reason(err: Exception) -> str:
    """A short reason for the errors list. A 401 marks the key as rejected, so it is not tried again."""
    status = getattr(err, "status_code", None)
    body = getattr(err, "body", None)
    kind = body["error"].get("type") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
    if status == 401:
        _CLAUDE_STATE["rejected"] = anthropic_key()
        return "Anthropic key rejected"
    if "credit balance" in str(getattr(err, "message", "") or err).lower():
        return "no Anthropic credits"
    return str(kind or status or type(err).__name__)


def _claude_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """The chat-format messages as (system text, turns) for the Messages API: the system turns become the system
    prompt word for word, empty turns are dropped, and the turns start with the user's."""
    system = "\n\n".join(str(m["content"]) for m in messages if m.get("role") == "system" and m.get("content"))
    turns = [{"role": m["role"], "content": str(m["content"])} for m in messages
             if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    return system, turns


def _claude_args(system: str, turns: list[dict], max_tokens: int, *, schema: dict | None = None, think: bool = True,
                 session_id: str | None = None) -> dict:
    """The request: same system prompt and turns as the OpenRouter call, low effort. Thinking is adaptive (Haiku 5.5's
    default) for answers and extraction, off for the small extras, where it could use up their few output tokens."""
    args: dict = {"model": CLAUDE_MODEL, "max_tokens": max_tokens, "messages": turns,
                  "output_config": {"effort": CLAUDE_EFFORT}}
    if system:
        args["system"] = system
    if not think:
        args["thinking"] = {"type": "disabled"}
    if schema is not None:
        args["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    if session_id:  # an opaque id only: a session may be "ip:<address>"
        args["metadata"] = {"user_id": hashlib.sha256(session_id.encode("utf-8", "replace")).hexdigest()[:32]}
    return args


def _claude_cost(inp: int, out: int, write: int = 0, read: int = 0) -> float:
    tier = CLAUDE_PRICES[1] if inp + write + read > CLAUDE_LONG_PROMPT else CLAUDE_PRICES[0]
    return (inp * tier[0] + out * tier[1] + write * tier[2] + read * tier[3]) / 1_000_000


def _claude_worst(system: str, turns: list[dict], max_tokens: int, extra: int = 0) -> float:
    """The most one call can cost, held against the cap before the call: each UTF-8 byte of the prompt counted as a
    token (a token is at least one byte), room for the request's own framing, and max_tokens of output."""
    prompt = len(system.encode()) + sum(len(t["content"].encode()) + 16 for t in turns) + extra + 256
    return _claude_cost(prompt, max_tokens)


def _claude_usage(msg, messages: list[dict], text: str, exact: bool) -> Usage:
    """Usage of one Claude call. exact: `msg` is the final message (its usage is the bill). Otherwise `msg` is the
    stream's snapshot or None: input tokens from it, output tokens estimated from the text (chars/4); no snapshot and
    no text means the call never started (nothing billed)."""
    u = getattr(msg, "usage", None)
    if u is None and not text:
        return Usage()
    inp = int(getattr(u, "input_tokens", 0) or 0)
    write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
    out = int(getattr(u, "output_tokens", 0) or 0)
    if not exact:
        if not (inp or write or read):
            inp = _chars_tokens("".join(str(m.get("content", "")) for m in messages))
        out = max(out, _chars_tokens(text))
    return Usage(prompt_tokens=inp + write + read, completion_tokens=out, cost_usd=_claude_cost(inp, out, write, read),
                 model=CLAUDE_MODEL, calls=1)


def _claude_call(messages: list[dict], *, max_tokens: int, timeout: float, schema: dict | None = None,
                 think: bool = True) -> tuple[str, Usage]:
    """One non-streaming Claude call (JSON output when `schema` is given). Returns (text, usage). Raises LLMError when
    today's cap is reached or the call fails. What it costs is counted toward the cap either way."""
    system, turns = _claude_messages(messages)
    worst = _claude_worst(system, turns, max_tokens, len(json.dumps(schema)) if schema is not None else 0)
    cap = claude_cap()
    ticket = cap.hold(worst)
    if ticket is None:
        raise LLMError(f"{CLAUDE_MODEL}: daily cap reached")
    cost = 0.0
    try:
        client = _claude_client()
        import anthropic
        cost = worst  # sent: until the answer says otherwise, it may cost up to this
        try:
            msg = client.messages.create(**_claude_args(system, turns, max_tokens, schema=schema, think=think),
                                         timeout=timeout)
        except _claude_errors() as err:
            if isinstance(err, anthropic.APIStatusError):
                cost = 0.0  # the API answered with an error: nothing billed
            raise LLMError(f"{CLAUDE_MODEL}: {_claude_reason(err)}") from err
        usage = _claude_usage(msg, messages, "", exact=True)
        cost = usage.cost_usd
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if msg.stop_reason == "refusal" or not text.strip():
            raise LLMError(f"{CLAUDE_MODEL}: {'refusal' if msg.stop_reason == 'refusal' else 'no text'}")
        return text, usage
    finally:
        cap.settle(ticket, cost)


def _claude_stream(messages: list[dict], timeout: float, max_tokens: int, session_id: str | None, errors: list[str],
                   dropped: Usage, on_usage: Callable[[Usage], None] | None) -> Iterator[str]:
    """stream_text's Claude step. Yields the answer text and returns True once text was streamed (the chain ends
    there; a failure after that ends the text with INTERRUPTED). Returns False when it failed before any text: the
    reason goes to `errors` and a billed attempt to `dropped`. No hand-over after FIRST_TOKEN_S here: the models after
    Claude are the free backups, which take about a minute."""
    system, turns = _claude_messages(messages)
    worst = _claude_worst(system, turns, max_tokens)
    cap = claude_cap()
    ticket = cap.hold(worst)
    if ticket is None:
        errors.append(f"{CLAUDE_MODEL}: daily cap reached")
        return False
    out: list[str] = []
    stream = final = None
    try:
        try:
            client = _claude_client()
        except LLMError as err:
            errors.append(str(err))
            return False
        failed = ""
        try:
            with client.messages.stream(**_claude_args(system, turns, max_tokens, session_id=session_id),
                                        timeout=timeout) as stream:
                for event in stream:
                    if event.type == "text" and event.text:
                        out.append(event.text)
                        yield event.text
                final = stream.get_final_message()
        except _claude_errors() as err:  # before the stream opened, or a dropped connection / error event inside it
            failed = _claude_reason(err)
        if final is not None and final.stop_reason == "refusal":
            failed = "refusal"
        if failed and out:
            out.append(INTERRUPTED)
            yield INTERRUPTED
        elif failed:
            errors.append(f"{CLAUDE_MODEL}: {failed}")
        return bool(out)
    finally:
        snapshot = final
        if snapshot is None and stream is not None:
            try:
                snapshot = stream.current_message_snapshot
            except AssertionError:  # no message_start yet
                snapshot = None
        usage = _claude_usage(snapshot, messages, "".join(out), exact=final is not None)
        # an unfinished stream may have cost more than the text shows (thinking): hold the cap to the worst case
        cap.settle(ticket, usage.cost_usd if final is not None or not usage.calls else max(usage.cost_usd, worst))
        if out and on_usage is not None:
            on_usage(_with_dropped(usage, dropped))
        elif not out and usage.calls:
            dropped.prompt_tokens += usage.prompt_tokens
            dropped.completion_tokens += usage.completion_tokens
            dropped.cost_usd += usage.cost_usd
            dropped.calls += usage.calls
            dropped.model = dropped.model or CLAUDE_MODEL


def _with_dropped(u: Usage, dropped: Usage) -> Usage:
    """`u` plus what the models dropped earlier in the chain were billed (calls included)."""
    if not dropped.calls:
        return u
    return Usage(u.prompt_tokens + dropped.prompt_tokens, u.completion_tokens + dropped.completion_tokens,
                 u.cost_usd + dropped.cost_usd, u.model, u.calls + dropped.calls)


def complete_json(messages: list[dict], schema: dict, name: str = "result", timeout: int = 60) -> tuple[dict, str]:
    """Return (parsed JSON, model used). Tries each model of chain() until one returns valid JSON."""
    if not available():
        raise LLMError("No OpenRouter API key configured.")
    errors = []
    order = chain()
    for i, model in enumerate(order):
        if model == CLAUDE_MODEL:
            try:
                text, _ = _claude_call(messages, max_tokens=4000, timeout=timeout, schema=schema)
                return json.loads(_clean_json(text), strict=False), model
            except LLMError as err:
                errors.append(str(err))
            except ValueError:
                errors.append(f"{model}: unusable answer")
            continue
        if not openrouter_ok():  # only with Claude on: no OpenRouter key, or it was rejected
            errors.append(f"{model}: no usable OpenRouter key")
            continue
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
            if CLAUDE_MODEL not in order[i + 1:]:
                raise LLMError("The OpenRouter API key was rejected.")
            errors.append(f"{model}: OpenRouter key rejected")
            continue
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
    """Yield answer text as it streams. Falls back to the next model of chain() if one fails before any text.

    `deadline_s` stops trying further models after that many seconds (raises LLMError).
    `on_usage` is called once with the Usage of the model that answered, when the stream ends. A model dropped for
    writing nothing after FIRST_TOKEN_S was still billed for its prompt and thinking: that estimate is added to the
    usage (calls included), and when no model answers on_usage still gets it before LLMError is raised. A Claude
    attempt that failed before any text but was billed is added the same way, with its real usage."""
    if not available():
        raise LLMError("No OpenRouter API key configured.")
    errors = []
    start = time.monotonic()
    dropped = Usage()  # models given up on after FIRST_TOKEN_S (billed anyway)
    order = chain()
    for i, model in enumerate(order):
        wait = timeout
        if deadline_s is not None:
            left = deadline_s - (time.monotonic() - start)
            if left <= 0:
                errors.append("time limit reached")
                break
            wait = max(1, min(timeout, int(left) + 1))
        if model == CLAUDE_MODEL:
            if (yield from _claude_stream(messages, wait, max_tokens, session_id, errors, dropped, on_usage)):
                return
            continue
        if not openrouter_ok():  # only with Claude on: no OpenRouter key, or it was rejected
            errors.append(f"{model}: no usable OpenRouter key")
            continue
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
            if CLAUDE_MODEL not in order[i + 1:]:
                raise LLMError("The OpenRouter API key was rejected.")
            errors.append(f"{model}: OpenRouter key rejected")
            continue
        if r.status_code != 200:
            errors.append(f"{model}: {'no OpenRouter credits' if r.status_code == 402 else r.status_code}")
            r.close()
            continue
        started = False
        out: list[str] = []
        raw_usage = None
        r.encoding = "utf-8"
        sent_at = time.monotonic()
        has_next = model != order[-1]
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
                on_usage(_with_dropped(_usage(model, raw_usage, messages, "".join(out)), dropped))
        if started:
            return
    if dropped.calls and on_usage is not None:
        on_usage(dropped)
    raise _failure(errors)


def quick_json(messages: list[dict], schema: dict, *, model: str = TRANSLATE_MODEL, max_tokens: int = 250,
               timeout: float = 3, name: str = "result", post=None) -> tuple[dict, Usage]:
    """One small structured call to a fast model, no fallback chain (for extras like starter questions); only when
    it fails and ANTHROPIC_API_KEY is set, Claude gets one try (thinking off, same limits). Raises LLMError on any
    failure."""
    if not available():
        raise LLMError("No OpenRouter API key configured.")
    if openrouter_ok():
        try:
            return _quick_openrouter(messages, schema, model, max_tokens, timeout, name, post)
        except LLMError:
            if not claude_on():
                raise
    text, usage = _claude_call(messages, max_tokens=max_tokens, timeout=timeout, schema=schema, think=False)
    try:
        return json.loads(_clean_json(text), strict=False), usage
    except ValueError as err:
        raise LLMError(f"{CLAUDE_MODEL}: unusable answer") from err


def _quick_openrouter(messages: list[dict], schema: dict, model: str, max_tokens: int, timeout: float, name: str,
                      post) -> tuple[dict, Usage]:
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
    """Translate a short search question into the site's language (Flash-Lite, max 80 tokens; Claude with thinking
    off when that fails and ANTHROPIC_API_KEY is set).
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
    content, usage_of = None, None
    if openrouter_ok():
        try:
            r = requests.post(URL, headers=_headers(), json={"model": TRANSLATE_MODEL, "messages": messages, "max_tokens": 80,
                                                             "usage": {"include": True}}, timeout=timeout)
            if r.status_code == 200:
                data = json.loads(r.text, strict=False)
                content = (data["choices"][0]["message"]["content"] or "").strip()
                usage_of = lambda: _usage(TRANSLATE_MODEL, data.get("usage"), messages, content)  # noqa: E731
        except Exception:  # noqa: BLE001 - Claude (when set) or the original question
            content = None
    if content is None:
        if not claude_on():
            return text
        try:
            answer, claude_usage = _claude_call(messages, max_tokens=80, timeout=timeout, think=False)
        except Exception:  # noqa: BLE001 - any failure keeps the original question
            return text
        content, usage_of = answer.strip(), lambda: claude_usage
    try:
        line = content.splitlines()[0].strip().strip('"\'«»“”') if content else ""
        if on_usage is not None:
            try:
                on_usage(usage_of())
            except Exception:  # noqa: BLE001 - a bad callback must not break search
                pass
        if not line or len(line) > 4 * len(text) + 40:
            return text
        return line
    except Exception:  # noqa: BLE001 - any failure keeps the original question
        return text
