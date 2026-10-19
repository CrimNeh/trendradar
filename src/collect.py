"""One collection pass: all carousels -> details -> snapshots.

~25-40 HTTP calls typical. Idempotent: safe to run every 15 min.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import roblox_api as api
from classify import classify
from db import ChannelHit, Game, Session, Snapshot, init_db, utcnow

# Server stats are optional enrichment and the endpoint rate limits hard
# (returns 429 for a whole day at a time). Off by default; enable with
# --servers when you want to try, the circuit breaker protects the run either
# way. Set SERVER_SAMPLE=0 to keep it off everywhere.
SERVER_SAMPLE = int(os.environ.get("SERVER_SAMPLE", "40"))


def _parse_dt(raw: str | None):
    if not raw or raw.startswith("0001"):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def collect(do_servers: bool = True) -> dict:
    init_db()
    ts = utcnow()

    membership: dict[int, list[tuple[str, int]]] = {}
    sponsored: set[int] = set()
    for sort_id in api.SORTS:
        for rank, g in enumerate(api.get_sort(sort_id), start=1):
            uid = g.get("universeId")
            if not uid:
                continue
            uid = int(uid)
            membership.setdefault(uid, []).append((sort_id, rank))
            if g.get("isSponsored"):
                sponsored.add(uid)

    if not membership:
        return {"error": "no games discovered", "ts": ts}

    def say(msg: str) -> None:
        """Progress to the console. Buffered output made a slow run look hung."""
        print(msg, flush=True)

    found = len(membership)
    say(f"[1/4] discovery: {found} games in {len(api.SORTS)} carousels")

    uids = list(membership)
    details = {d["id"]: d for d in api.get_details(uids)}
    say(f"[2/4] details: {len(details)} resolved")
    # Present in a carousel but missing from details is a transient API gap -
    # skip rather than storing a blank row.
    uids = [u for u in uids if u in details]
    if not uids:
        return {"error": "details endpoint returned nothing", "ts": ts}

    votes = api.get_votes(uids)
    icons = api.get_icon_urls(uids)

    by_ccu = sorted(((u, details[u].get("playing") or 0) for u in uids), key=lambda x: -x[1])
    srv: dict[int, dict] = {}
    if do_servers and SERVER_SAMPLE > 0:
        say(f"[3/4] server stats: up to {SERVER_SAMPLE} games "
            f"(optional, can be slow or rate limited)")
        for n, (uid, _) in enumerate(by_ccu[:SERVER_SAMPLE], 1):
            srv[uid] = api.get_server_stats(uid)
            if not api.servers_available():
                say(f"      endpoint rate limited after {n} game(s), skipping the rest")
                break
        say(f"      done: {len(srv)} game(s)")
    else:
        say("[3/4] server stats: skipped")

    written = 0
    with Session() as s:
        for uid in uids:
            d = details[uid]
            v = votes.get(uid) or {}
            name = d.get("name")
            niche, mechs = classify(name, d.get("description"))

            g = s.get(Game, uid)
            if g is None:
                g = Game(universe_id=uid)
                s.add(g)
            g.root_place_id = d.get("rootPlaceId")
            g.name = (name or "")[:256]
            g.description = (d.get("description") or "")[:4000]
            cr = d.get("creator") or {}
            g.creator_id = cr.get("id")
            g.creator_name = cr.get("name")
            g.creator_type = cr.get("type")
            g.max_players = d.get("maxPlayers")
            g.created = _parse_dt(d.get("created"))
            g.updated = _parse_dt(d.get("updated"))
            g.icon_url = icons.get(uid)
            g.niche = niche
            g.tags = mechs
            g.last_seen = ts

            hits = membership[uid]
            st = srv.get(uid, {})
            s.add(Snapshot(
                universe_id=uid, ts=ts,
                playing=d.get("playing", 0),
                visits=d.get("visits", 0),
                favorites=d.get("favoritedCount", 0),
                up_votes=v.get("upVotes"), down_votes=v.get("downVotes"),
                max_players=d.get("maxPlayers"),
                servers=st.get("servers"), avg_fill=st.get("avg_fill"),
                chart_count=len(hits),
                best_chart_rank=min((r for _, r in hits), default=None),
                is_sponsored=1 if uid in sponsored else 0,
            ))
            for sort_id, rank in hits:
                s.add(ChannelHit(universe_id=uid, ts=ts, sort_id=sort_id, rank=rank))
            written += 1
        s.commit()

    say(f"[4/4] saved {written} games")
    return {"ts": ts, "found": found, "games": written,
            "server_calls": len(srv), "skipped_no_details": found - len(uids),
            "sponsored": len(sponsored), "error": None}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--servers", action="store_true",
                   help="also fetch server-fullness stats (slow, often rate limited)")
    p.add_argument("--no-servers", action="store_true",
                   help="skip server stats (now the default; kept for compatibility)")
    a = p.parse_args()
    do_servers = a.servers and not a.no_servers
    print(collect(do_servers=do_servers))
