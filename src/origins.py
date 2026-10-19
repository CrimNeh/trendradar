"""Origin signals: what starts a Roblox trend before Roblox notices.

Trends almost never begin inside Roblox. They arrive from outside and
materialise there days or weeks later. This module tracks three of those
sources, all free and keyless:

1. STEAM (mechanic import)
   A mechanic works on Steam or mobile -> someone reimplements it in Roblox
   two weeks later. Roblox's "steal a brainrot", "99 nights in the forest"
   and "grow a garden" all trace back to PC/mobile originals. We track the
   most-played Steam charts and tag by genre, so a rising extraction-shooter
   or farming sim is a hint about what Roblox will clone next.

2. ANILIST (IP wave)
   New anime season -> within a week Roblox fills with clones. We track
   upcoming and currently-airing anime ranked by popularity. Lead time is
   1-4 weeks, which is enough to build something.

3. DEVFORUM (platform features)
   Roblox ships a new capability and experiences built on it get an
   algorithmic boost. DevForum announcements land weeks before the boost.
   Longest lead time of any source here.

Each writes rows into `origin_signal` with a normalised `score`, so the
dashboard can rank them next to Roblox-native signals.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(__file__))
from db import OriginHistory, OriginSignal, Session, init_db, utcnow  # noqa: E402

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
_session = requests.Session()
_session.headers.update({"User-Agent": UA})

STEAM_CHARTS = "https://api.steampowered.com/ISteamChartsService/GetMostPlayedGames/v1/"
STEAMSPY = "https://steamspy.com/api.php"
ANILIST = "https://graphql.anilist.co"
DEVFORUM = "https://devforum.roblox.com"

# Steam genres that historically get cloned into Roblox fast.
# The score bonus reflects how often that has actually happened.
CLONEABLE = {
    "survival": 1.0, "simulation": 0.95, "casual": 0.9, "adventure": 0.7,
    "action": 0.6, "rpg": 0.65, "strategy": 0.55, "indie": 0.8,
    "massively multiplayer": 0.5, "racing": 0.6, "sports": 0.5,
}


def _get(url: str, params: dict | None = None, tries: int = 3):
    for attempt in range(tries):
        try:
            r = _session.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            return r
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


# ------------------------------------------------------------------- steam

def collect_steam(top_n: int = 40) -> list[dict]:
    """Most-played Steam games, enriched with genre and concurrent players.

    Steam's charts endpoint returns appids only; SteamSpy fills in name,
    genre and CCU without a key. Enrichment is capped because SteamSpy
    throttles aggressively (~1 req/sec sustained).
    """
    r = _get(STEAM_CHARTS)
    if not r:
        return []
    try:
        ranks = r.json()["response"]["ranks"]
    except (KeyError, json.JSONDecodeError):
        return []

    out = []
    for row in ranks[:top_n]:
        aid = row.get("appid")
        if not aid:
            continue
        rs = _get(STEAMSPY, {"request": "appdetails", "appid": aid})
        time.sleep(1.1)
        if not rs:
            continue
        try:
            j = rs.json()
        except json.JSONDecodeError:
            continue
        name = (j.get("name") or "").strip()
        if not name:
            continue

        genre = (j.get("genre") or "").lower()
        tags = [g.strip() for g in genre.split(",") if g.strip()]
        clone = max((CLONEABLE.get(g, 0.3) for g in tags), default=0.3)

        peak = row.get("peak_in_game") or j.get("ccu") or 0
        # Rank position is the main interest signal; cloneability is a bonus
        # because a popular non-cloneable game is not actionable here.
        rank_score = max(0.0, 100.0 - (row.get("rank", 100) - 1) * 1.6)
        score = round(min(100.0, rank_score * (0.5 + 0.5 * clone)), 1)

        prev = row.get("last_week_rank")
        climbing = bool(prev and prev > row.get("rank", 999))

        out.append({
            "source": "steam",
            "key": str(aid),
            "title": name[:256],
            "score": score,
            "tag": ",".join(tags)[:64],
            "meta": json.dumps({
                "rank": row.get("rank"), "peak": peak, "price": j.get("price"),
                "cloneability": round(clone, 2), "climbing": climbing,
                "genre": genre,
            }),
        })
    return out


# ------------------------------------------------------------------- anime

ANIME_Q = """
query ($page: Int) {
  Page(page: $page, perPage: 30) {
    media(type: ANIME, status: NOT_YET_RELEASED, sort: POPULARITY_DESC, isAdult: false) {
      id
      title { romaji english }
      startDate { year month day }
      popularity
      genres
      format
    }
  }
}
"""


def collect_anime() -> list[dict]:
    """Upcoming anime by popularity. Lead time to a clone wave: 1-4 weeks."""
    payload = {"query": ANIME_Q, "variables": {"page": 1}}
    try:
        r = requests.post(ANILIST, json=payload, timeout=25, headers={"User-Agent": UA})
        if r.status_code != 200:
            return []
        media = r.json()["data"]["Page"]["media"]
    except (KeyError, json.JSONDecodeError, requests.RequestException):
        return []

    now = datetime.now(timezone.utc)
    out = []
    for m in media:
        title = (m["title"].get("english") or m["title"].get("romaji") or "").strip()
        if not title:
            continue
        sd = m.get("startDate") or {}
        start = None
        if sd.get("year"):
            try:
                start = datetime(sd["year"], sd.get("month") or 1,
                                 sd.get("day") or 1, tzinfo=timezone.utc)
            except ValueError:
                start = None

        days_to = (start - now).days if start else 999
        if days_to < -14:
            continue  # already aired, the wave has passed

        pop = m.get("popularity") or 0
        # Popularity normalised to 0-100, boosted when the premiere is near:
        # a clone built now can be live before episode one drops.
        base = min(100.0, (pop / 60000.0) * 100.0)
        if 0 <= days_to <= 30:
            base = min(100.0, base * 1.25)
        elif days_to <= 90:
            base = min(100.0, base * 1.1)

        out.append({
            "source": "anime",
            "key": str(m["id"]),
            "title": title[:256],
            "score": round(base, 1),
            "tag": ",".join(m.get("genres") or [])[:64],
            "meta": json.dumps({
                "days_to_premiere": days_to, "popularity": pop,
                "format": m.get("format"), "genres": m.get("genres"),
            }),
        })
    return out


# ---------------------------------------------------------------- devforum

# Category slugs verified to return RSS. `updates` is the platform-feature
# feed (longest lead time); `latest` catches early build announcements.
DEVFORUM_FEEDS = [
    ("updates", f"{DEVFORUM}/c/updates.rss"),
    ("latest", f"{DEVFORUM}/latest.rss"),
]

# Topics that usually precede a Roblox-wide shift.
FEATURE_HINT = re.compile(
    r"\b(release|beta|launch|announc|introduc|new feature|update|api|"
    r"open source|early access|testing)\b", re.I)


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def collect_devforum() -> list[dict]:
    """Recent DevForum topics. Long lead time: devs post early builds here
    months before a public release, and platform updates land weeks before
    the algorithm starts boosting experiences that use them."""
    out = []
    seen: set[str] = set()

    for channel, url in DEVFORUM_FEEDS:
        r = _get(url)
        if not r:
            continue
        text = r.text
        for item in re.findall(r"<item>(.*?)</item>", text, re.S):
            m_title = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item, re.S)
            m_link = re.search(r"<link>(.*?)</link>", item, re.S)
            m_date = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
            if not m_title:
                continue
            title = _strip_tags(m_title.group(1))
            link = (m_link.group(1) if m_link else "").strip()
            if not title or link in seen:
                continue
            seen.add(link)

            published = None
            if m_date:
                for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
                    try:
                        published = datetime.strptime(m_date.group(1).strip(), fmt)
                        break
                    except ValueError:
                        continue
            days_old = (
                (datetime.now(timezone.utc) - published).days
                if published else 30
            )
            if days_old > 30:
                continue

            # Platform-wide feature announcements matter most.
            score = 40.0
            if channel == "updates":
                score += 30
            if FEATURE_HINT.search(title):
                score += 15
            score += max(0, 15 - days_old)

            out.append({
                "source": "devforum",
                "key": link[:64] or title[:64],
                "title": title[:256],
                "score": round(min(100.0, score), 1),
                "tag": channel,
                "meta": json.dumps({"url": link, "days_old": days_old,
                                    "channel": channel}),
            })
    return out


# ------------------------------------------------------------------- write

def upsert(rows: list[dict]) -> int:
    """Write the latest state AND append today's point to the history.

    Latest state keeps the best score ever seen; history keeps one point per
    day (re-running within a day overwrites that day, it never duplicates).
    """
    if not rows:
        return 0
    init_db()
    now = utcnow()
    today = now.strftime("%Y-%m-%d")
    n = 0
    with Session() as s:
        for row in rows:
            meta = {}
            try:
                meta = json.loads(row.get("meta") or "{}")
            except json.JSONDecodeError:
                pass

            existing = s.query(OriginSignal).filter(
                OriginSignal.source == row["source"],
                OriginSignal.key == row["key"]).first()
            if existing is None:
                s.add(OriginSignal(observed=now, **row))
            else:
                existing.title = row["title"]
                existing.score = max(existing.score or 0, row["score"])
                existing.meta = row["meta"]
                existing.tag = row["tag"]
                existing.observed = now

            extra = meta.get("peak") if row["source"] == "steam" else meta.get("popularity")
            hist = s.query(OriginHistory).filter(
                OriginHistory.source == row["source"],
                OriginHistory.key == row["key"],
                OriginHistory.day == today).first()
            if hist is None:
                s.add(OriginHistory(
                    source=row["source"], key=row["key"], day=today,
                    score=row["score"], rank=meta.get("rank"),
                    extra=float(extra) if extra is not None else None))
            else:
                hist.score = row["score"]
                hist.rank = meta.get("rank")
                hist.extra = float(extra) if extra is not None else None
            n += 1
        s.commit()
    return n


def trends(days: int = 7) -> list[dict]:
    """Signals ordered by how much they climbed over the window.

    Needs at least two daily points per signal; with fewer, nothing is
    returned rather than a made-up slope.
    """
    init_db()
    from datetime import timedelta
    cutoff = (utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    out: list[dict] = []
    with Session() as s:
        rows = s.query(OriginHistory).filter(OriginHistory.day >= cutoff)\
            .order_by(OriginHistory.source, OriginHistory.key, OriginHistory.day).all()
        by: dict[tuple, list] = {}
        for r in rows:
            by.setdefault((r.source, r.key), []).append(r)
        titles = {(x.source, x.key): x.title for x in s.query(OriginSignal)}
    for (src, key), pts in by.items():
        if len(pts) < 2 or pts[0].score is None or pts[-1].score is None:
            continue
        delta = pts[-1].score - pts[0].score
        out.append({
            "source": src, "title": titles.get((src, key), key),
            "from": round(pts[0].score, 1), "to": round(pts[-1].score, 1),
            "delta": round(delta, 1), "days": len(pts),
        })
    out.sort(key=lambda x: -x["delta"])
    return out


def collect_all() -> dict:
    steam = collect_steam(top_n=40)
    anime = collect_anime()
    forum = collect_devforum()
    return {
        "steam": upsert(steam),
        "anime": upsert(anime),
        "devforum": upsert(forum),
    }


if __name__ == "__main__":
    print(collect_all())
