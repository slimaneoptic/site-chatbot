"""Daily guards for the public demo: AI answers per session and per IP, a global USD budget,
site reads per IP and PDF uploads per IP. One shared instance per process; counters reset at 00:00 UTC.
Also pick_client_ip(): the visitor address for the per-IP caps, shared by api/main.py and app.py, and DailyCap: the
hard daily USD cap of llm.py's Claude fallback."""
from __future__ import annotations

import ipaddress
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

MESSAGES = {
    "session_cap": "You have used the AI answers for this visit, so here are the most relevant passages from the site instead.",
    "ip_cap": "Today's AI answers for your network are used up, so here are the most relevant passages from the site instead. AI answers come back tomorrow.",
    "global_budget": "The demo has used today's AI budget, so here are the most relevant passages from the site instead. AI answers come back tomorrow.",
    "ip_crawl_cap": "You have read the maximum number of sites for today. You can still ask about the sites already loaded, or come back tomorrow.",
    "ip_upload_cap": "You have uploaded the maximum number of PDFs for today. You can still ask about the files already loaded, or come back tomorrow.",
}


def pick_client_ip(forwarded, peer: str | None, hops: int = 1, platform_ip: str | None = None) -> str:
    """Visitor address for the daily caps. `forwarded` is the X-Forwarded-For header (a string or the list of
    its values). Its leftmost entry is whatever the visitor sent, so trusting it would let anyone dodge the
    per-IP caps. Take instead the entry our own proxy wrote: the `hops`-th from the right (1 = the address the
    platform's proxy saw). `platform_ip` (a header the platform sets and overwrites) wins when present.
    hops=0 or no header: the TCP peer. An IPv6 address counts as its /64 network (see ip_bucket)."""
    if platform_ip and platform_ip.strip():
        return ip_bucket(platform_ip.strip()[:64])
    if isinstance(forwarded, str):
        forwarded = [forwarded]
    entries = [e.strip() for e in ",".join(v for v in (forwarded or []) if v).split(",") if e.strip()]
    if entries and hops > 0:
        return ip_bucket(entries[-min(hops, len(entries))][:64])
    return ip_bucket((peer or "unknown")[:64])


def ip_bucket(ip: str) -> str:
    """The key the per-IP caps count under: an IPv4 address as is; an IPv6 address as its /64 network, because one
    connection usually holds a whole /64 (per-address caps would not limit it); an IPv4 address written as IPv6
    (::ffff:1.2.3.4) as the IPv4 address. Anything else unchanged."""
    try:
        addr = ipaddress.ip_address(ip.strip().strip("[]"))
    except ValueError:
        return ip
    if addr.version == 6:
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return f"{ipaddress.IPv6Address(int(addr) >> 64 << 64)}/64"  # int() also takes a zone id (fe80::1%eth0)
    return ip


@dataclass
class BudgetDecision:
    allowed: bool
    reason: str  # "ok" | "session_cap" | "ip_cap" | "global_budget" | "ip_crawl_cap" | "ip_upload_cap"
    message: str  # friendly visitor text, "" when ok


OK = BudgetDecision(True, "ok", "")
RESERVE_USD = 0.002   # what one in-flight answer may still cost (an answer costs ~$0.0004-0.001)
RESERVE_TTL_S = 300   # a reservation nobody settled (a crash between check and record) frees itself


def _deny(reason: str) -> BudgetDecision:
    return BudgetDecision(False, reason, MESSAGES[reason])


class Budget:
    def __init__(self, session_cap: int = 12, ip_cap: int = 30, daily_usd: float = 1.0,
                 ip_crawl_cap: int = 5, clock=time.time, ip_upload_cap: int = 10):
        self.session_cap = session_cap
        self.ip_cap = ip_cap
        self.daily_usd = daily_usd
        self.ip_crawl_cap = ip_crawl_cap
        self.ip_upload_cap = ip_upload_cap
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
            self._uploads = 0
            self._uploads_by_ip: dict[str, int] = {}
            self._pending: list[tuple[float, str, str]] = []  # (time, session, ip) of answers in flight

    def check(self, session: str, ip: str, reserve: bool = False) -> BudgetDecision:
        """May this session/IP get an AI answer now? reserve=True (the caller will call the model) holds one unit
        until record(..., reserved=True) settles it, so a burst of parallel requests can't all pass the caps
        before the first one is counted."""
        with self._lock:
            self._reset_if_new_day()
            now = self._clock()
            self._pending = [r for r in self._pending if now - r[0] < RESERVE_TTL_S]
            if self._spent + RESERVE_USD * len(self._pending) >= self.daily_usd:
                return _deny("global_budget")
            if self._by_session.get(session, 0) + sum(r[1] == session for r in self._pending) >= self.session_cap:
                return _deny("session_cap")
            if self._by_ip.get(ip, 0) + sum(r[2] == ip for r in self._pending) >= self.ip_cap:
                return _deny("ip_cap")
            if reserve:
                self._pending.append((now, session, ip))
            return OK

    def record(self, session: str, ip: str, usage, reserved: bool = False) -> None:
        """Count one question when usage.calls > 0; always add its cost. reserved=True also frees the unit
        that check(..., reserve=True) held."""
        with self._lock:
            self._reset_if_new_day()
            if reserved:
                for i, r in enumerate(self._pending):
                    if r[1] == session and r[2] == ip:
                        del self._pending[i]
                        break
            self._spent += max(0.0, float(getattr(usage, "cost_usd", 0.0) or 0.0))
            if getattr(usage, "calls", 0) > 0:
                self._questions += 1
                self._by_session[session] = self._by_session.get(session, 0) + 1
                self._by_ip[ip] = self._by_ip.get(ip, 0) + 1

    def can_spend(self) -> bool:
        """True while today's global budget has room (extras such as starter questions use only this)."""
        with self._lock:
            self._reset_if_new_day()
            return self._spent + RESERVE_USD * len(self._pending) < self.daily_usd

    def add_spend(self, usd: float) -> None:
        """Add a cost to today's global spend without counting a question for anyone."""
        with self._lock:
            self._reset_if_new_day()
            self._spent += max(0.0, float(usd or 0.0))

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

    def check_upload(self, ip: str) -> BudgetDecision:
        with self._lock:
            self._reset_if_new_day()
            if self._uploads_by_ip.get(ip, 0) >= self.ip_upload_cap:
                return _deny("ip_upload_cap")
            return OK

    def record_upload(self, ip: str) -> None:
        with self._lock:
            self._reset_if_new_day()
            self._uploads += 1
            self._uploads_by_ip[ip] = self._uploads_by_ip.get(ip, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            self._reset_if_new_day()
            return {"day": self._day, "spent_usd": round(self._spent, 6), "questions": self._questions, "crawls": self._crawls,
                    "uploads": self._uploads, "in_flight": len(self._pending)}


class DailyCap:
    """A hard daily USD cap for one paid AI provider (the Claude fallback in llm.py), kept like Budget's counters: in
    this instance's memory, reset at 00:00 UTC. hold() books a call's worst-case cost before the call, so parallel
    calls can't pass the cap together; settle() swaps that hold for the real cost read from the response usage."""

    def __init__(self, daily_usd: float, clock=time.time):
        self.daily_usd = daily_usd
        self._clock = clock
        self._lock = threading.Lock()
        self._day = ""
        self._reset_if_new_day()

    def _reset_if_new_day(self) -> None:  # call with the lock held (or from __init__)
        day = datetime.fromtimestamp(self._clock(), timezone.utc).date().isoformat()
        if day != self._day:
            self._day, self._spent, self._held = day, 0.0, 0.0

    def hold(self, usd: float) -> tuple[str, float] | None:
        """Book `usd` when it still fits under today's cap: returns the ticket for settle(), or None (cap reached)."""
        usd = max(0.0, float(usd or 0.0))
        with self._lock:
            self._reset_if_new_day()
            if self._spent + self._held + usd > self.daily_usd:
                return None
            self._held += usd
            return (self._day, usd)

    def settle(self, ticket: tuple[str, float], cost_usd: float) -> None:
        """Free the ticket's hold (unless the day changed since) and add the call's real cost."""
        with self._lock:
            self._reset_if_new_day()
            day, usd = ticket
            if day == self._day:
                self._held = max(0.0, self._held - usd)
            self._spent += max(0.0, float(cost_usd or 0.0))

    def snapshot(self) -> dict:
        with self._lock:
            self._reset_if_new_day()
            return {"day": self._day, "spent_usd": round(self._spent, 6), "cap_usd": self.daily_usd,
                    "in_flight_usd": round(self._held, 6)}
