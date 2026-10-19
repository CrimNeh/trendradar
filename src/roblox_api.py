"""Thin client for public Roblox web APIs.

No auth, no keys. Shared token bucket + exponential backoff on 429/5xx.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 7 sort ids verified live. Add only after confirming they return data.
SORTS = {
    "up-and-coming": "Up-and-Coming",
    "top-trending": "Top Trending",
    "top-playing-now": "Top Playing Now",
    "top-earning": "Top Earning",
    "top-revisited": "Top Revisited",
    "top-rated": "Top Rated",
    "fun-with-friends": "Fun with Friends",
}

EXPLORE = "https://apis.roblox.com/explore-api/v1/get-sort-content"
GAMES = "https://games.roblox.com/v1/games"
VOTES = "https://games.roblox.com/v1/games/votes"
SERVERS = "https://games.roblox.com/v1/games/{uid}/servers/Public"
THUMBS = "https://thumbnails.roblox.com/v1/games/icons"


class RateLimiter:
    def __init__(self, rate: float = 4.0, per: float = 1.0):
        self.interval = per / rate
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


_RATE_LIMITED = object()      # sentinel: endpoint refused us, do not retry

_RATE_LIMITED = object()      # sentinel: endpoint refused us, do not retry

_limiter = RateLimiter(rate=4.0)
_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept": "application/json"})


def _get(url: str, params: dict | None = None, tries: int = 4,
         fail_fast_429: bool = False) -> Any:
    """GET with retries.

    fail_fast_429: for OPTIONAL enrichment endpoints. A 429 there means the
    endpoint is throttling us for the day, not that one request went wrong -
    retrying just wastes minutes. Measured: `/servers/Public` answers 429 in
    0.2 s, and the old retry loop turned that into ~2 minutes of sleeping per
    game (4 tries x up to 30 s), which looked like a hang.
    """
    last = None
    for attempt in range(tries):
        _limiter.wait()
        try:
            r = _session.get(url, params=params, timeout=20)
            if r.status_code == 429:
                if fail_fast_429:
                    return _RATE_LIMITED
                ra = float(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(ra, 30))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (400, 404):
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** attempt + random.random())
    if last:
        raise last
    return None


def get_sort(sort_id: str, device: str = "computer", country: str = "all") -> list[dict]:
    """One page of a discovery carousel (~90-100 games)."""
    d = _get(EXPLORE, {"sessionId": str(random.randint(1, 10**9)),
                       "sortId": sort_id, "device": device, "country": country})
    return (d or {}).get("games") or []


def get_details(universe_ids: list[int]) -> list[dict]:
    out: list[dict] = []
    for i in range(0, len(universe_ids), 50):
        d = _get(GAMES, {"universeIds": ",".join(str(x) for x in universe_ids[i:i + 50])})
        if d:
            out.extend(d.get("data") or [])
    return out


def get_votes(universe_ids: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for i in range(0, len(universe_ids), 50):
        d = _get(VOTES, {"universeIds": ",".join(str(x) for x in universe_ids[i:i + 50])})
        if d:
            for row in d.get("data") or []:
                out[row["id"]] = row
    return out


_servers_disabled = False


def servers_available() -> bool:
    """False once the endpoint has rate-limited us in this process."""
    return not _servers_disabled


def get_server_stats(universe_id: int, limit: int = 50) -> dict:
    """Server fullness. `limit` maxes at 50 (100 returns 400).

    Optional enrichment: the endpoint is heavily rate limited and often
    refuses entirely. On the first 429 the whole endpoint is disabled for the
    rest of the run, so 40 games cannot cost 40 x 2 minutes of waiting.
    """
    global _servers_disabled
    empty = {"servers": 0, "total_players": 0, "avg_fill": 0.0, "max_players": 0}
    if _servers_disabled:
        return empty

    d = _get(SERVERS.format(uid=universe_id), {"limit": min(limit, 50)},
             tries=2, fail_fast_429=True)
    if d is _RATE_LIMITED:
        _servers_disabled = True
        return empty
    rows = (d or {}).get("data") or []
    if not rows:
        return empty
    total = sum(r.get("playing", 0) for r in rows)
    cap = sum(r.get("maxPlayers", 0) for r in rows)
    return {"servers": len(rows), "total_players": total,
            "avg_fill": round(total / cap, 4) if cap else 0.0,
            "max_players": max(r.get("maxPlayers", 0) for r in rows)}


def get_icon_urls(universe_ids: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for i in range(0, len(universe_ids), 50):
        d = _get(THUMBS, {"universeIds": ",".join(str(x) for x in universe_ids[i:i + 50]),
                          "size": "150x150", "format": "Png", "isCircular": "false"})
        if d:
            for row in d.get("data") or []:
                if row.get("imageUrl"):
                    out[row["targetId"]] = row["imageUrl"]
    return out


if __name__ == "__main__":
    for sid, name in SORTS.items():
        print(f"{name:20s} {len(get_sort(sid)):3d} games")
