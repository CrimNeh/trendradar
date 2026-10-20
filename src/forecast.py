"""Forward-looking signals: what is about to happen outside Roblox.

    python src/forecast.py

digest_html.py shows the present - who is gaining right now. This module answers
the question that actually buys lead time: what drops in the next few weeks, and
which Roblox niches will feel it.

Three sources feed it, all already collected by origins.py:

  ANIME    AniList upcoming titles with premiere dates. The clone wave for a
           popular title starts 1-4 weeks before episode one, not after.
  DEVFORUM Roblox platform changes. A new API can make a genre cheap to build
           that was expensive last month.
  STEAM    Games climbing the charts. A hit elsewhere is a proven formula that
           has not reached Roblox yet.

The genre-to-niche mapping is imported from link.py so the dashboard and the
markdown report cannot disagree about what "survival" means.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from db import Game, OriginSignal, Session  # noqa: E402
from link import GENRE_MAP  # noqa: E402

# Beyond this the premiere is too far out to act on today.
HORIZON_DAYS = 120
# An anime below this popularity will not move a Roblox niche on its own.
MIN_ANIME_POP = 20_000


def _meta(raw) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _niches_for(genres: list[str]) -> list[str]:
    """Map AniList genres onto Roblox niche labels, most specific first."""
    out: list[str] = []
    for g in genres:
        for niche in GENRE_MAP.get(str(g).lower(), []):
            if niche not in out:
                out.append(niche)
    return out


def niche_counts() -> dict[str, int]:
    """How many Roblox games already sit in each niche."""
    with Session() as s:
        rows = s.query(Game.niche).all()
    counts: dict[str, int] = {}
    for (n,) in rows:
        if n:
            counts[n] = counts.get(n, 0) + 1
    return counts


def upcoming_anime(counts: dict[str, int]) -> list[dict]:
    """Anime premiering inside the horizon, ranked by actionable impact.

    Impact is not popularity alone. A hugely popular title whose genres map to
    a niche already crowded with 40 games is a worse bet than a mid-size title
    pointing at a niche with two. So the score blends reach with emptiness.
    """
    with Session() as s:
        rows = s.query(OriginSignal).filter(OriginSignal.source == "anime").all()

    out = []
    for r in rows:
        m = _meta(r.meta)
        days = m.get("days_to_premiere")
        pop = m.get("popularity") or 0
        if days is None or days > HORIZON_DAYS or pop < MIN_ANIME_POP:
            continue
        if days < 0:
            continue   # already aired, the wave has passed

        genres = m.get("genres") or []
        niches = _niches_for(genres)
        if not niches:
            continue

        # Emptiness of the target niches: fewer existing games = more room.
        live = [counts.get(n, 0) for n in niches]
        avg_live = sum(live) / len(live) if live else 0
        room = max(0.0, 1.0 - min(avg_live, 40) / 40.0)

        # Proximity: a premiere in 3 days is actionable, one in 100 is not yet.
        near = max(0.0, 1.0 - days / HORIZON_DAYS)

        reach = min(1.0, pop / 60_000)
        impact = round((reach * 0.5 + near * 0.3 + room * 0.2) * 100, 1)

        out.append({
            "title": r.title,
            "days": int(days),
            "pop": int(pop),
            "format": m.get("format", ""),
            "genres": genres,
            "niches": niches[:3],
            "live": avg_live,
            "impact": impact,
            "score": float(r.score or 0),
        })

    return sorted(out, key=lambda x: -x["impact"])


def platform_changes(limit: int = 10) -> list[dict]:
    """Recent DevForum posts that unlock or cheapen a way of building games."""
    with Session() as s:
        rows = (s.query(OriginSignal)
                 .filter(OriginSignal.source == "devforum")
                 .order_by(OriginSignal.score.desc())
                 .limit(60).all())

    out = []
    seen: set[str] = set()
    for r in rows:
        m = _meta(r.meta)
        days = m.get("days_old")
        if days is None or days > 45:
            continue
        # Release notes arrive every week and say little; skip them.
        low = (r.title or "").lower()
        if low.startswith("release notes"):
            continue
        if r.title in seen:
            continue
        seen.add(r.title)

        kind = "beta" if "beta" in low else ("feature" if "introduc" in low else "update")
        out.append({
            "title": r.title,
            "days": int(days),
            "kind": kind,
            "url": m.get("url", ""),
            "channel": m.get("channel", ""),
            "score": float(r.score or 0),
        })
        if len(out) >= limit:
            break
    return out


def steam_climbing() -> list[dict]:
    """Steam titles that are rising and cheap to rebuild on Roblox."""
    with Session() as s:
        rows = s.query(OriginSignal).filter(OriginSignal.source == "steam").all()

    out = []
    for r in rows:
        m = _meta(r.meta)
        clone = m.get("cloneability") or 0
        if not m.get("climbing") or clone < 0.6:
            continue
        out.append({
            "title": r.title,
            "peak": int(m.get("peak") or 0),
            "genre": (m.get("genre") or "")[:44],
            "clone": float(clone),
            "score": float(r.score or 0),
        })
    return sorted(out, key=lambda x: -x["score"])[:8]


def summary_line(anime: list[dict], changes: list[dict], steam: list[dict]) -> list[str]:
    """Three short sentences for the top of the page."""
    bits: list[str] = []
    soon = [a for a in anime if 0 <= a["days"] <= 14]
    if soon:
        a = soon[0]
        # "in 0 days" is not a sentence a person would say. Premiere day is its
        # own case: the wave is already starting.
        when = ("сегодня" if a["days"] == 0 else
                ("завтра" if a["days"] == 1 else f"через {a['days']} дн"))
        bits.append(f"Премьера «{a['title'][:42]}» {when} — "
                    f"спрос ждать в нишах {', '.join(a['niches'][:2])}")
    elif anime:
        a = anime[0]
        bits.append(f"Ближайшая крупная премьера «{a['title'][:42]}» через {a['days']} дн")

    if changes:
        c = changes[0]
        bits.append(f"Roblox выкатил {len(changes)} изменений за месяц, свежее — "
                    f"«{c['title'][:40]}» ({c['days']} дн назад)")

    if steam:
        s0 = steam[0]
        bits.append(f"В Steam растёт «{s0['title'][:30]}» (пик {s0['peak']:,}) — "
                    f"формула ещё не перенесена в Roblox")

    return bits


def build() -> dict:
    counts = niche_counts()
    return {
        "anime": upcoming_anime(counts),
        "changes": platform_changes(),
        "steam": steam_climbing(),
        "counts": counts,
    }


def main() -> None:
    d = build()
    print(f"anime: {len(d['anime'])}  changes: {len(d['changes'])}  steam: {len(d['steam'])}")
    for a in d["anime"][:6]:
        print(f"  {a['days']:>4} дн  impact {a['impact']:>5.1f}  "
              f"{a['title'][:44]:46s} -> {','.join(a['niches'][:2])} "
              f"(игр в нишах {a['live']:.0f})")
    print()
    for l in summary_line(d["anime"], d["changes"], d["steam"]):
        print("  •", l)


if __name__ == "__main__":
    main()
