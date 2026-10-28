"""One collection pass: all carousels -> details -> snapshots.

~25-40 HTTP calls typical. Idempotent: safe to run every 15 min.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

import roblox_api as api
from classify import classify
from db import ChannelHit, Game, Session, Snapshot, init_db, utcnow

# Server stats are optional enrichment and the endpoint rate limits hard
# (returns 429 for a whole day at a time). Off by default; enable with
# --servers when you want to try, the circuit breaker protects the run either
# way. Set SERVER_SAMPLE=0 to keep it off everywhere.
SERVER_SAMPLE = int(os.environ.get("SERVER_SAMPLE", "40"))

# Orphan sweep thresholds. See collect_orphans() for why the floor matters:
# the digest and the forecast ledger both drop games under 500 CCU, so a
# snapshot of a game that can never qualify only costs storage. Deliberately
# the same number as digest_html.MIN_CCU and predictions.MIN_BASELINE_CCU -
# one floor, stated once per consumer, never two different ones.
ORPHAN_MIN_CCU = int(os.environ.get("ORPHAN_MIN_CCU", "500"))
ORPHAN_RECHECK_H = float(os.environ.get("ORPHAN_RECHECK_H", "24"))


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
            # Promotion. A game discovered via YouTube and later picked up by a
            # carousel is no longer an orphan: it belongs to the market sample
            # now, and its whole history (orphan-era snapshots included) becomes
            # eligible for the median and the base rate. Without this it would
            # stay 'orphan' forever and be excluded from the metrics even while
            # appearing in every chart.
            g.origin = "carousel"

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


def collect_orphans(ts, limit: int = 600) -> dict:
    """Snapshot the games discovery found but the carousels never show.

    Measured on this database 2026-10-28: 496 of 1200 games (41%) had no CCU
    history at all, every one of them discovered through a YouTube mention
    rather than a Roblox carousel. They were not junk - several carry 500M+
    visits - they were simply invisible, because collect() only walks the
    carousels and a game outside them is added to `games` and never measured
    again.

    Asked Roblox directly for all 499 of them: median CCU 116, but 161 (32%)
    sat above the 500 floor, and that group's median was 1812 - comparable to
    or larger than the carousel sample. So this is not a long tail of noise; a
    third of it is material and was being thrown away.

    Two deliberate limits:

    * ORPHAN_MIN_CCU - below it, no snapshot is written. 338 of the 499 were
      under 500 CCU, mostly single-digit traffic from one small video. They can
      never enter a forecast (the digest and the ledger both floor at 500), so
      storing them forever only costs rows.
    * ORPHAN_RECHECK_H - a game is re-queried at most once a day. Without it
      the same permanently-tiny games would be asked about on all 8 daily runs
      and dominate the sweep's cost. `orphan_checked` is stamped whether or
      not a snapshot was written, so the skip is durable.

    Cost: one call per 50 games (roblox_api.get_details batches), so a full
    sweep of 500 is 10 calls - cheaper than a single carousel pass.

    These snapshots are excluded from the market median and the base rate
    until the week-2 comparison is done; see digest_html.INCLUDE_ORPHANS.
    """
    init_db()
    cutoff = ts - timedelta(hours=ORPHAN_RECHECK_H)
    with Session() as s:
        rows = (s.query(Game.universe_id)
                 .filter(Game.origin == "orphan")
                 .filter((Game.orphan_checked.is_(None))
                         | (Game.orphan_checked < cutoff))
                 .order_by(Game.orphan_checked.asc().nullsfirst())
                 .limit(limit).all())
    uids = [r[0] for r in rows]
    if not uids:
        return {"asked": 0, "snapshotted": 0, "below_floor": 0, "unavailable": 0}

    details = {d["id"]: d for d in api.get_details(uids)}
    written = below = gone = 0
    with Session() as s:
        for uid in uids:
            g = s.get(Game, uid)
            if g is None:
                continue
            g.orphan_checked = ts
            d = details.get(uid)
            # Roblox returns nothing for a deleted or private universe. Leave
            # the row as 'orphan' so it is retried tomorrow: a game reinstated
            # later is picked up without intervention.
            if d is None:
                gone += 1
                continue
            ccu = d.get("playing") or 0
            g.name = (d.get("name") or g.name or "")[:256]
            g.last_seen = ts
            if ccu < ORPHAN_MIN_CCU:
                below += 1
                continue
            s.add(Snapshot(
                universe_id=uid, ts=ts, playing=ccu,
                visits=d.get("visits", 0),
                favorites=d.get("favoritedCount", 0),
                max_players=d.get("maxPlayers"),
                # No carousel membership by definition: zero rather than NULL,
                # so "not in any chart" stays distinguishable from "not measured".
                chart_count=0, best_chart_rank=None, is_sponsored=0,
            ))
            written += 1
        s.commit()

    print(f"[orphans] asked {len(uids)}, snapshotted {written}, "
          f"below {ORPHAN_MIN_CCU} CCU {below}, unavailable {gone}", flush=True)
    return {"asked": len(uids), "snapshotted": written,
            "below_floor": below, "unavailable": gone}


def _recent_snapshot_hours() -> float | None:
    """Hours since the newest snapshot, or None if the table is empty.

    Used to refuse a duplicate run. Two triggers fire at the same minute -
    GitHub's cron and the external pinger - and `concurrency` with
    `cancel-in-progress: false` queues the second instead of dropping it. Both
    pass, and the whole market is written twice. Measured over 41 hours: 429
    duplicate pairs less than 3 minutes apart, roughly doubling storage, the
    YouTube quota draw and the snapshot density `excess_samples` reads.
    """
    from sqlalchemy import func
    with Session() as s:
        newest = s.query(func.max(Snapshot.ts)).scalar()
    if newest is None:
        return None
    return (utcnow() - newest).total_seconds() / 3600


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--servers", action="store_true",
                   help="also fetch server-fullness stats (slow, often rate limited)")
    p.add_argument("--no-servers", action="store_true",
                   help="skip server stats (now the default; kept for compatibility)")
    p.add_argument("--force", action="store_true",
                   help="run even if a snapshot was taken less than --min-gap hours ago")
    p.add_argument("--min-gap", type=float, default=0.5,
                   help="skip if the newest snapshot is younger than this, "
                        "in hours (default 0.5 = 30 min)")
    a = p.parse_args()

    do_servers = a.servers and not a.no_servers

    # The guard lives here, not in the workflow, because it has to hold no
    # matter WHICH trigger fired. A cron line cannot know the pinger already
    # ran, and the pinger cannot know that cron did.
    if not a.force:
        age = _recent_snapshot_hours()
        if age is not None and age < a.min_gap:
            print(f"пропуск: последний снимок {age*60:.0f} мин назад "
                  f"(порог {a.min_gap*60:.0f} мин) — это повторный триггер, "
                  f"а не новый прогон. --force, если нужен явный запуск.")
            raise SystemExit(0)

    stats = collect(do_servers=do_servers)
    print(stats)
    # The orphan sweep runs after the carousel pass, and only when that pass
    # actually wrote something - a failed collection should not be masked by a
    # sweep reporting its own clean counts.
    if isinstance(stats, dict) and stats.get("ts") and not stats.get("error"):
        print(collect_orphans(stats["ts"]))
