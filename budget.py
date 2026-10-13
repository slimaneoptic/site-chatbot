"""Daily guards for the public demo: AI answers per session and per IP, a global USD budget
and site reads per IP. One shared instance per process; counters reset at 00:00 UTC."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

MESSAGES = {
    "session_cap": "You have used the AI answers for this visit, so here are the most relevant passages from the site instead.",
    "ip_cap": "Today's AI answers for your network are used up, so here are the most relevant passages from the site instead. AI answers come back tomorrow.",
    "global_budget": "The demo has used today's AI budget, so here are the most relevant passages from the site instead. AI answers come back tomorrow.",
    "ip_crawl_cap": "You have read the maximum number of sites for today. You can still ask about the sites already loaded, or come back tomorrow.",
}


@dataclass
class BudgetDecision:
    allowed: bool
    reason: str  # "ok" | "session_cap" | "ip_cap" | "global_budget" | "ip_crawl_cap"
    message: str  # friendly visitor text, "" when ok


OK = BudgetDecision(True, "ok", "")


def _deny(reason: str) -> BudgetDecision:
    return BudgetDecision(False, reason, MESSAGES[reason])


class Budget:
    def __init__(self, session_cap: int = 12, ip_cap: int = 30, daily_usd: float = 1.0,
                 ip_crawl_cap: int = 5, clock=time.time):
        self.session_cap = session_cap
        self.ip_cap = ip_cap
        self.daily_usd = daily_usd
        self.ip_crawl_cap = ip_crawl_cap
        self._clock = clock
        self._lock = threading.Lock()
        self._day = ""
        self._reset_if_new_day()

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), timezone.utc).date().isoformat()

    def _reset_if_new_day(self) -> None:  # call with the lock held (or from __init__)
        day = self._today()
        if day != self._day:
            self._day = day
            self._spent = 0.0
            self._questions = 0
            self._crawls = 0
            self._by_session: dict[str, int] = {}
            self._by_ip: dict[str, int] = {}
            self._crawls_by_ip: dict[str, int] = {}

    def check(self, session: str, ip: str) -> BudgetDecision:
        with self._lock:
            self._reset_if_new_day()
            if self._spent >= self.daily_usd:
                return _deny("global_budget")
            if self._by_session.get(session, 0) >= self.session_cap:
                return _deny("session_cap")
            if self._by_ip.get(ip, 0) >= self.ip_cap:
                return _deny("ip_cap")
            return OK

    def record(self, session: str, ip: str, usage) -> None:
        """Count one question when usage.calls > 0; always add its cost."""
        with self._lock:
            self._reset_if_new_day()
            self._spent += max(0.0, float(getattr(usage, "cost_usd", 0.0) or 0.0))
            if getattr(usage, "calls", 0) > 0:
                self._questions += 1
                self._by_session[session] = self._by_session.get(session, 0) + 1
                self._by_ip[ip] = self._by_ip.get(ip, 0) + 1

    def check_crawl(self, ip: str) -> BudgetDecision:
        with self._lock:
            self._reset_if_new_day()
            if self._crawls_by_ip.get(ip, 0) >= self.ip_crawl_cap:
                return _deny("ip_crawl_cap")
            return OK

    def record_crawl(self, ip: str) -> None:
        with self._lock:
            self._reset_if_new_day()
            self._crawls += 1
            self._crawls_by_ip[ip] = self._crawls_by_ip.get(ip, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            self._reset_if_new_day()
            return {"day": self._day, "spent_usd": round(self._spent, 6), "questions": self._questions, "crawls": self._crawls}
