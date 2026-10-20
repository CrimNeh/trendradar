"""Find new games and new videos through the YouTube Data API.

    python src/yt_discovery.py                 # budgeted run
    python src/yt_discovery.py --budget 800    # raise the ceiling
    python src/yt_discovery.py --dry           # show the plan, spend nothing

WHY THIS EXISTS

RSS polling can only see channels we already track. If a new creator starts
covering a game we have never heard of, the feed collector will never find it -
there is no feed for a channel nobody added.

search.list sees the whole platform. That is what makes it worth its 100 units.

BUDGET

The free quota is 10,000 units/day and is shared: the channel importer uses
some, this uses some, and anything else on the same key competes. So this module
thinks in units, not queries:

    search.list          100 units
    videos.list            1 unit per 50 ids

A default run spends 400 units on four searches plus a couple for stats - about
4% of the day. That leaves the importer and manual runs plenty of room, and it
is enough to cover the four queries that actually produce new games.

WHAT QUERIES

Not "roblox" - that returns the same mega-channels RSS already covers. The
useful queries are the ones phrased the way a creator titles a video about a
game nobody has indexed yet:

    "new roblox game"        what it says
    "roblox <tracked game>"  catches titles our token pass is too strict for
    localized variants        non-English markets move first on some games

Everything found is written through the same matcher and dedupe path as RSS, so
a video found twice is stored once.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import match as matcher  # noqa: E402
import yt_api  # noqa: E402
from db import Mention, Session, YtChannel, YtVideo, utcnow  # noqa: E402
from youtube_collect import games_frame  # noqa: E402

# One search costs 100. Four searches = 400 units, a safe daily slice.
DEFAULT_BUDGET = 400
# Beyond this the results are old enough that RSS would have caught them.
LOOKBACK_DAYS = 3

# Generic discovery. Deliberately few: each one costs a full search.
BASE_QUERIES = [
    "new roblox game",
    "roblox update today",
]

# Localized. Some games trend in Brazil or Indonesia days before the English
# market notices, so one query per major non-English market is worth its cost.
LOCALE_QUERIES = {
    "pt": "jogo novo roblox",
    "es": "juego nuevo roblox",
    "id": "game roblox baru",
    "ru": "новая игра роблокс",
}


def _published_after() -> str:
    cut = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    return cut.strftime("%Y-%m-%dT%H:%M:%SZ")


def plan(queries: list[str], budget: int) -> list[str]:
    """Take queries in order until the budget cannot cover another search."""
    cost = yt_api.COST["search"]
    left = min(budget, yt_api.api.quota_left())
    out = []
    for q in queries:
        if left < cost + yt_api.COST["videos"] * 4:
            break
        out.append(q)
        left -= cost
    return out


def _game_query(s: Session, limit: int = 2) -> list[str]:
    """Queries built from games that moved in the last day.

    Searching for a game that is already rising finds the videos our token pass
    rejected for being too vague ("this game is insane" with the name only in
    the thumbnail). Those are exactly the videos worth having.
    """
    rows = s.execute(
        __import__("sqlalchemy").text("""
            select g.name from games g
            join snapshots sn on sn.universe_id = g.universe_id
            where sn.ts = (select max(ts) from snapshots)
              and sn.chart_count > 0
            order by sn.playing desc limit 40
        """)).fetchall()
    names = [r[0] for r in rows if r[0]]
    # Strip emoji and tags so the query is a real phrase a human would type.
    clean = []
    for n in names:
        t = "".join(c for c in n if c.isalnum() or c in " -!").strip()
        t = " ".join(t.split())[:40]
        if len(t) >= 4:
            clean.append(t)
        if len(clean) >= limit:
            break
    return [f"roblox {c}" for c in clean]


def run(budget: int = DEFAULT_BUDGET, dry: bool = False) -> dict:
    stats = {"queries": 0, "found": 0, "new_videos": 0, "mentions": 0,
             "units_before": 0, "units_after": 0, "skipped": ""}

    if not yt_api.api.available:
        stats["skipped"] = "YT_API_KEY не задан"
        return stats

    stats["units_before"] = yt_api.api.quota_used()

    with Session() as s:
        games = games_frame(s)
        if games.empty:
            stats["skipped"] = "нет игр в базе"
            return stats

        # Assemble the query list in priority order: localized first (they find
        # things English search cannot), then generic, then per-game.
        queries = list(LOCALE_QUERIES.values()) + BASE_QUERIES
        queries += _game_query(s, limit=2)

        chosen = plan(queries, budget)
        stats["planned"] = len(chosen)

        if dry:
            stats["skipped"] = "dry run"
            stats["plan"] = chosen
            return stats

        seen_ids: set[str] = set()
        matched_ids = {v for (v,) in s.query(Mention.video_id).distinct().all()}
        known_videos = {v for (v,) in s.query(YtVideo.video_id).all()}

    collected: list[dict] = []
    for q in chosen:
        try:
            found = yt_api.api.search(q, max_results=25, order="date",
                                      published_after=_published_after())
        except yt_api.QuotaExceeded as e:
            stats["skipped"] = f"квота: {e}"
            break
        except Exception as e:
            stats["skipped"] = f"{q}: {type(e).__name__}"
            continue

        stats["queries"] += 1
        stats["found"] += len(found)
        for v in found:
            vid = v.get("video_id")
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                collected.append(v)

    if not collected:
        stats["units_after"] = yt_api.api.quota_used()
        return stats

    # Exact stats for everything found, batched 50 per unit. RSS gives views
    # already, but not likes or comments, and search results carry neither.
    stats_by_id: dict[str, dict] = {}
    try:
        for chunk_start in range(0, len(collected), 50):
            ids = [v["video_id"] for v in collected[chunk_start:chunk_start + 50]]
            for info in yt_api.api.videos(ids):
                stats_by_id[info["video_id"]] = info
    except yt_api.QuotaExceeded:
        pass

    unknown_places: dict[int, tuple[dict, str]] = {}
    with Session() as s:
        games = games_frame(s)
        place_to_uid = {int(p): int(u) for u, p in
                        zip(games.universe_id, games.root_place_id) if pd.notna(p)}
        matched = {v for (v,) in s.query(Mention.video_id).distinct().all()}

        for v in collected:
            vid = v["video_id"]
            info = stats_by_id.get(vid, {})
            published = v.get("published") or ""
            try:
                pub = datetime.fromisoformat(
                    published.replace("Z", "+00:00")).astimezone(timezone.utc
                                                                 ).replace(tzinfo=None)
            except (ValueError, AttributeError):
                pub = utcnow()

            hours = max((utcnow() - pub).total_seconds() / 3600, 0.5)
            views = int(info.get("views") or 0)

            if vid not in known_videos:
                known_videos.add(vid)
                s.add(YtVideo(
                    video_id=vid, channel_id=v.get("channel_id"),
                    title=(v.get("title") or "")[:512], published=pub,
                    observed=utcnow(), views=views,
                    views_per_hour=round(views / hours, 1)))
                stats["new_videos"] += 1

            if vid in matched:
                continue

            desc = v.get("description") or ""
            hit = None

            # pass 1: an explicit place link in the description
            for pid in matcher.extract_place_ids(desc):
                uid = place_to_uid.get(pid)
                if uid:
                    hit = {"universe_id": uid, "method": "link",
                           "containment": 1.0, "matched": f"place:{pid}", "score": 2.0}
                    break
                unknown_places.setdefault(pid, (v, "en"))

            # pass 2: the game name in the title
            if hit is None and not matcher.suspicious_title(v.get("title")):
                hit = matcher.best_match(v.get("title") or "", games)

            if hit:
                s.add(Mention(
                    video_id=vid, universe_id=int(hit["universe_id"]),
                    observed=utcnow(), containment=hit.get("containment"),
                    matched=(hit.get("matched") or "")[:128],
                    channel_id=v.get("channel_id"), lang="en",
                    method=hit.get("method"), score=hit.get("score")))
                matched.add(vid)
                stats["mentions"] += 1

            # A discovered channel is worth adding: it proved it covers games.
            cid = v.get("channel_id")
            if cid and s.get(YtChannel, cid) is None:
                s.add(YtChannel(
                    channel_id=cid, handle=(v.get("channel_name") or "")[:64],
                    lang="en", active=1))

        # Games linked from descriptions that we do not track yet.
        if unknown_places:
            from youtube_collect import _ingest_new_games, _resolve_place
            fresh = []
            for pid in list(unknown_places)[:25]:
                uid = _resolve_place(pid)
                if uid and s.get(Game := __import__("db").Game, uid) is None:
                    fresh.append(uid)
            stats["new_games"] = _ingest_new_games(s, fresh)

        s.commit()

    stats["units_after"] = yt_api.api.quota_used()
    stats["units_spent"] = stats["units_after"] - stats["units_before"]
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help=f"max units to spend this run (default {DEFAULT_BUDGET})")
    p.add_argument("--dry", action="store_true", help="plan only, spend nothing")
    a = p.parse_args()

    if not yt_api.api.available:
        print("YT_API_KEY не задан — discovery пропущен, RSS работает как обычно")
        return

    print(f"квота: использовано {yt_api.api.quota_used()}, "
          f"осталось {yt_api.api.quota_left()}")
    r = run(budget=a.budget, dry=a.dry)

    if r.get("plan"):
        print("\nплан запросов:")
        for q in r["plan"]:
            print(f"   «{q}»  (100 units)")

    print()
    print(f"запросов        : {r['queries']}")
    print(f"видео найдено   : {r['found']}")
    print(f"новых в базе    : {r['new_videos']}")
    print(f"совпадений      : {r['mentions']}")
    print(f"новых игр       : {r.get('new_games', 0)}")
    print(f"квоты потрачено : {r.get('units_spent', 0)}")
    if r.get("skipped"):
        print(f"примечание      : {r['skipped']}")


if __name__ == "__main__":
    main()
