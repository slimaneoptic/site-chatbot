import os
import sys
from pathlib import Path

import pytest

# Unit tests start from an empty answer cache: the deployed api/starter_answers.json is loaded only when asked.
os.environ.setdefault("STARTER_ANSWERS", os.path.join(os.path.dirname(__file__), "no-starter-answers.json"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budget  # noqa: E402
import llm  # noqa: E402


def _no_live_claude():
    raise AssertionError("a test reached the real Anthropic API: give it a fake client (see tests/test_claude.py)")


@pytest.fixture(autouse=True)
def _claude_off(monkeypatch):
    """Claude stays off unless a test sets ANTHROPIC_API_KEY and a fake client itself: a key in the shell that runs
    the tests must never turn into live calls. Each test also starts with a fresh cap and no remembered key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_DAILY_CAP", raising=False)
    monkeypatch.setattr(llm, "_claude_client", _no_live_claude)
    monkeypatch.setattr(llm, "_CLAUDE_STATE", {"rejected": None, "key": None, "client": None})
    monkeypatch.setattr(llm, "CLAUDE_CAP", budget.DailyCap(llm.CLAUDE_DAILY_CAP))
