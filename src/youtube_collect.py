"""Multilingual YouTube collection.

Roblox growth is no longer English-led: Brazil, the Philippines, Indonesia,
Russia and MENA are where player counts move. Channels are stored with a
`lang`, so YouTube pressure can be sliced by market.

Matching a video to a game runs three passes (see match.py):
  1. link   - Roblox game URL in the description (exact, any language)
  2. token  - game name appears in the title
  3. embed  - multilingual semantic match for what the first two missed

A link that points at a game we do NOT track yet is also how the system
finds games outside the charts: it is resolved and added to the catalogue.
That is the first real answer to the "we only see charted games" blind spot.

    python src/youtube_collect.py --seed           # seed channels
    python src/youtube_collect.py --discover-lang  # find more per language
    python src/youtube_collect.py                  # poll everything
    python src/youtube_collect.py --no-embed       # skip the semantic pass
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))

import match as matcher
import roblox_api as rbx
import youtube as yt
from classify import classify
from sqlalchemy import text
from db import Game, Mention, Session, YtChannel, YtVideo, init_db, utcnow

# Seed handles. Ones that fail to resolve are skipped, which is why the list
# is generous - names get taken and rotate.
SEED = {
    "en": ["@KreekCraft", "@Flamingo", "@Laughability", "@DenisDaily",
           "@ItsSpoox", "@NapkinNate", "@PinkRage", "@BaconSwag"],
    "es": ["@RobloxEnEspanol", "@Rubius", "@elrubio", "@ChuyMine",
           "@NickieJuega", "@rufiblox", "@ElTrollino"],
    "pt": ["@AuthenticGames", "@JuliaMineGirl", "@Lokis", "@Athosgamer",
           "@RobinHoodGamer1", "@TazerCraft"],
    "ru": ["@EdisonPts", "@Rai4err", "@SvarkaShow", "@delvirRBX",
           "@PozziGamer", "@vladusgame"],
    "id": ["@MiawAug", "@MipanZuzuzuRoblox", "@modinner", "@DuniaFarashROBLOX",
           "@BibiCantik01"],
    "ar": ["@AbuAbed", "@Bandar", "@RobloxArabia"],
    "tr": ["@EfeUyga", "@RobloxTurkiye"],
    "vi": ["@Misthy", "@RobloxVietnam"],
    "ph": ["@RobloxPH"],
}

DISCOVERY_QUERIES = {
    "en": "roblox new game",
    "es": "roblox en español",
    "pt": "roblox português",
    "ru": "roblox игры",
    "id": "roblox bahasa indonesia",
    "ar": "روبلوكس",
    "tr": "roblox türkçe",
    "vi": "roblox việt nam",
    "ph": "roblox pilipinas",
}

MAX_NEW_GAMES_PER_RUN = 25

# Ceiling on titles sent through the semantic pass in one run.
#
# Run #83 (2026-10-28 16:00) was cancelled at the job's 35-minute timeout with
# `Collect YouTube` sitting at 32.4 minutes, after which the digest, the
# forecast record and the resolve all skipped. The cause was volume, not a bug:
# 16814 videos arrived in that 12h window against 4815 the day before (3.5x),
# and this pass embedded every unmatched title with no cap. 19842 videos in the
# table have no mention, so a busy window is unbounded work.
#
# The pass is opportunistic - it re-runs whenever `youtube_full` is due, and
# anything left unembedded stays in `pending` for the next tick, because a
# title only drops out of the queue once it gets a mention. So cutting the
# batch loses nothing; it just spreads the same work over more runs.
#
# 4000 is chosen to keep the pass near ~4 minutes at the rate measured against
# this table (roughly 60ms/title including the index build), leaving the rest
# of the budget for the steps that actually produce the page and the ledger.
MAX_EMBED_PER_RUN = int(os.environ.get("YT_MAX_EMBED", "4000"))

# Nothing older than this can be evidence of what is trending NOW.
#
# Two separate reasons, both measured:
#
# 1. They are not evidence. The queue held 4266 videos older than 30 days (up
#    to 1915 days) carrying the highest raw view counts, purely because they
#    have had years to accumulate. Ranking by popularity without an age floor
#    promoted a 520-day-old Minecraft animation to the front of the pass. A
#    video from last year says what happened last year.
#
# 2. The queue has to drain, not grow. Throughput is MAX_EMBED_PER_RUN per run
#    and `youtube_full` fires twice a day, i.e. ~8000 videos/day. Candidate
#    counts measured 2026-10-28: <=3d 3768, <=7d 8473, <=30d 15576. A 30-day
#    window is larger than daily throughput, so the backlog would grow forever
#    and every video would age in the queue while waiting its turn - a video
#    admitted on day 25 would be matched on day 27 and the signal already
#    stale. A 7-day window sits just above throughput, so the queue drains.
#
# 7 days, not 3: the 1-7 day band is 8473 - 3768 = 4705 videos and contains
# the bulk of genuine trend evidence (a video needs a few days to accumulate
# the views that make it worth matching). Cutting to 3 would throw that away
# to save time the pass does not need.
#
# Videos outside the window stay in `yt_videos` as reference data - they are
# not deleted, just excluded from the semantic pass.
EMBED_MAX_AGE_DAYS = int(os.environ.get("YT_EMBED_MAX_AGE_DAYS", "7"))

# When a new game enters the catalogue, already-checked videos get one more
# chance to match it - but only this many, only this recent. See
# _ingest_new_games() for why the bound is not optional.
EMBED_RECHECK_DAYS = int(os.environ.get("YT_EMBED_RECHECK_DAYS", "3"))
EMBED_RECHECK_MAX = int(os.environ.get("YT_EMBED_RECHECK_MAX", "2000"))

# Concurrent RSS fetches. Measured: 150 feeds in 16 threads take 2.4 s, so the
# network is not the bottleneck - the reason 70 channels had never been polled
# is that the earlier run died before reaching them, not that fetching was slow.
WORKERS = int(os.environ.get("YT_WORKERS", "16"))

# Tiered polling. A channel that produced a matched video in the last week is
# far more likely to do it again than one that has been silent for months, so
# the active ones get checked every run and the quiet ones rotate.
#
#   hot   - produced a mention in the last 7 days  -> every run
#   warm  - produced anything in the last 30 days  -> every other run
#   cold  - never produced, or silent for 60+ days -> one eighth of the runs
#
# Rotation is by run number so it is deterministic and every channel is covered
# eventually, without keeping per-channel cursors.
TIER_HOT_DAYS = 7
TIER_WARM_DAYS = 30
COLD_EVERY = 8

# A channel with no uploads on its feed is dead weight. After this many
# consecutive empty polls it is deactivated instead of retried forever.
MAX_EMPTY_POLLS = int(os.environ.get("YT_MAX_EMPTY", "5"))


# --------------------------------------------------------------- channels

def discover_language_channels(lang: str, limit: int = 10) -> list[str]:
    """Channels ranking for a language-specific Roblox query ('@handle')."""
    query = DISCOVERY_QUERIES.get(lang)
    if not query:
        return []
    html = yt._get(
        f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}")
    if not html:
        return []
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", html)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    renderers: list[dict] = []
    yt._walk(data, "videoRenderer", renderers)
    handles, seen = [], set()
    for v in renderers[:limit * 2]:
        try:
            url = v["ownerText"]["runs"][0]["navigationEndpoint"]["browseEndpoint"].get(
                "canonicalBaseUrl", "")
        except (KeyError, IndexError, TypeError):
            continue
        if not url.startswith("/@"):
            continue
        h = url[1:]
        if h.lower() in seen:
            continue
        seen.add(h.lower())
        handles.append(h)
        if len(handles) >= limit:
            break
    return handles


def add_channel(s, handle: str, lang: str = "en") -> str | None:
    handle = handle if handle.startswith("@") else "@" + handle
    cid = yt.resolve_channel(handle)
    if not cid:
        return None
    row = s.get(YtChannel, cid)
    if row is None:
        s.add(YtChannel(channel_id=cid, handle=handle, lang=lang))
    else:
        row.active = 1
        if lang and row.lang != lang:
            row.lang = lang
    return cid


def seed_channels(langs: list[str] | None = None) -> dict:
    init_db()
    stats = {"added": 0, "failed": 0}
    with Session() as s:
        for lang, handles in SEED.items():
            if langs and lang not in langs:
                continue
            for h in handles:
                stats["added" if add_channel(s, h, lang) else "failed"] += 1
                s.commit()
    return stats


def discover_all(langs: list[str] | None = None, per_lang: int = 8) -> dict:
    init_db()
    stats = {"added": 0, "failed": 0, "by_lang": {}}
    with Session() as s:
        for lang in DISCOVERY_QUERIES:
            if langs and lang not in langs:
                continue
            ok = 0
            for h in discover_language_channels(lang, limit=per_lang):
                if add_channel(s, h, lang):
                    stats["added"] += 1
                    ok += 1
                else:
                    stats["failed"] += 1
                time.sleep(0.2)
            s.commit()
            stats["by_lang"][lang] = ok
            time.sleep(1.0)
    return stats


# ------------------------------------------------------------- recording

def games_frame(s) -> pd.DataFrame:
    return pd.DataFrame(
        [(g.universe_id, g.name, g.root_place_id) for g in s.query(Game)],
        columns=["universe_id", "name", "root_place_id"])


def _record_video(s, u: dict, now, known: set | None = None,
                  rows: dict | None = None) -> bool:
    """One row per video; views only grow, so refresh in place.

    `rows` is a preloaded {video_id: YtVideo} map. Without it every existing
    video cost a SELECT - 1144 of them, roughly 68 s of pure latency on Neon.
    The ORM objects returned by the preload are live, so mutating one and
    committing is enough; no second query is needed.
    """
    if rows is not None:
        row = rows.get(u["video_id"])
        if row is None:
            rows[u["video_id"]] = YtVideo(
                video_id=u["video_id"], channel_id=u["channel_id"],
                title=u["title"][:512], published=u["published"],
                observed=now, views=u["views"],
                views_per_hour=u["views_per_hour"])
            s.add(rows[u["video_id"]])
            return True
        if u["views"] > (row.views or 0):
            row.views = u["views"]
            row.views_per_hour = u["views_per_hour"]
            row.observed = now
            return True
        return False

    if known is not None and u["video_id"] not in known:
        known.add(u["video_id"])
        s.add(YtVideo(
            video_id=u["video_id"], channel_id=u["channel_id"],
            title=u["title"][:512], published=u["published"],
            observed=now, views=u["views"], views_per_hour=u["views_per_hour"]))
        return True

    row = s.query(YtVideo).filter(YtVideo.video_id == u["video_id"]).first()
    if row is None:
        s.add(YtVideo(
            video_id=u["video_id"], channel_id=u["channel_id"],
            title=u["title"][:512], published=u["published"],
            observed=now, views=u["views"], views_per_hour=u["views_per_hour"]))
        return True
    if u["views"] > (row.views or 0):
        row.views = u["views"]
        row.views_per_hour = u["views_per_hour"]
        row.observed = now
        return True
    return False


def _record_mention(s, video_id: str, m: dict, now, channel_id, lang,
                    matched: set | None = None) -> bool:
    """Add a mention. `matched` is the preloaded (video_id, universe_id) set.

    The set is what makes this cheap. The previous version did a COUNT query
    per call and only consulted `matched` in a branch that called `pass` - so
    the set did nothing and every mention cost a full round trip. Measured on
    Neon: 59 ms per query, and a busy run produces thousands of matches, i.e.
    minutes of pure latency. The set is loaded once in the caller; a miss here
    means the pair is genuinely new, and the unique constraint still guards
    against a duplicate slipping through a race.
    """
    if matched is not None:
        key = (video_id, int(m["universe_id"]))
        if key in matched:
            return False
    elif s.query(Mention).filter(Mention.video_id == video_id,
                                 Mention.universe_id == m["universe_id"]).count():
        # Only reached when no preloaded set was supplied.
        return False
    if matched is not None:
        matched.add((video_id, int(m["universe_id"])))
    s.add(Mention(
        video_id=video_id, universe_id=int(m["universe_id"]), observed=now,
        containment=m.get("containment"), matched=(m.get("matched") or "")[:128],
        channel_id=channel_id, lang=lang, method=m.get("method"),
        score=m.get("score")))
    return True


def _already_matched(s, video_id: str) -> bool:
    return s.query(Mention).filter(Mention.video_id == video_id).count() > 0


def _resolve_place(place_id: int) -> int | None:
    d = rbx._get(f"https://apis.roblox.com/universes/v1/places/{place_id}/universe")
    uid = (d or {}).get("universeId")
    return int(uid) if uid else None


def _parse_dt(raw: str | None):
    if not raw or raw.startswith("0001"):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _ingest_new_games(s, universe_ids: list[int]) -> int:
    """Add games found via video links that no chart has shown us yet.

    When a game is genuinely new to the catalogue, videos that were already
    looked at had nothing to match against and were marked `embed_checked`
    anyway. Without a reset those videos can never be revisited, so the game
    would only ever be matched against future uploads.

    The reset is deliberately bounded, because an unbounded one defeats the
    entire point of `embed_checked`:

      * only on runs that actually added a game (most runs add none)
      * only videos from the last EMBED_RECHECK_DAYS days
      * at most EMBED_RECHECK_MAX rows, newest-viewed first

    Worst case that is one extra pass over 2000 recent videos, on top of the
    day's new arrivals - still inside the ~8000/day batch throughput. Resetting
    the full 7-day window on every ingest (~50-130 new games a day) would refill
    the queue to 8000 every single run and make the queue un-drainable again,
    which is exactly the bug being fixed here.
    """
    if not universe_ids:
        return 0
    added = 0
    for d in rbx.get_details(universe_ids):
        uid = d.get("id")
        if not uid or s.get(Game, uid) is not None:
            continue
        niche, tags = classify(d.get("name"), d.get("description"))
        cr = d.get("creator") or {}
        s.add(Game(
            universe_id=uid, root_place_id=d.get("rootPlaceId"),
            name=(d.get("name") or "")[:256],
            description=(d.get("description") or "")[:4000],
            creator_id=cr.get("id"), creator_name=cr.get("name"),
            creator_type=cr.get("type"), max_players=d.get("maxPlayers"),
            created=_parse_dt(d.get("created")), updated=_parse_dt(d.get("updated")),
            niche=niche, tags=tags))
        added += 1

    if added:
        s.execute(text("""
            update yt_videos set embed_checked = null
            where video_id in (
                select video_id from yt_videos
                where embed_checked is not null
                  and published > now() - make_interval(days => :days)
                order by views desc nulls last
                limit :lim
            )
        """), {"days": EMBED_RECHECK_DAYS, "lim": EMBED_RECHECK_MAX})
    return added


# ------------------------------------------------------------------ poll

def _select_channels(s, now, run_no: int) -> list:
    """Which channels to poll this run, by tier.

    The naive version polled every active channel every time. That looked fine
    until the database showed 70 of 177 with last_checked NULL - the run was
    being cut short and the same early channels were polled forever while the
    rest never got a turn.

    Tiering fixes both problems: hot channels are always fresh, cold ones are
    still covered but one eighth as often, and the whole set is reachable.
    """
    from datetime import timedelta

    channels = s.query(YtChannel).filter(YtChannel.active == 1).all()
    if not channels:
        return []

    # Last mention per channel tells us who is actually producing.
    last_hit: dict[str, object] = {}
    rows = s.execute(text(
        "select channel_id, max(observed) from mentions "
        "where channel_id is not null group by channel_id")).fetchall()
    for cid, ts in rows:
        last_hit[cid] = ts

    hot_cut = now - timedelta(days=TIER_HOT_DAYS)
    warm_cut = now - timedelta(days=TIER_WARM_DAYS)

    out = []
    cold = []
    for ch in channels:
        hit = last_hit.get(ch.channel_id)
        if hit is not None and hit >= hot_cut:
            out.append(ch)                      # hot: every run
        elif hit is not None and hit >= warm_cut:
            if run_no % 2 == 0:                 # warm: every other run
                out.append(ch)
            else:
                cold.append(ch)
        else:
            cold.append(ch)

    # Cold slice: deterministic rotation by run number.
    if cold:
        slot = run_no % COLD_EVERY
        step = max(1, len(cold) // COLD_EVERY)
        out.extend(cold[slot * step:(slot + 1) * step] or cold[:step])

    return out


def poll_channels(discover: bool = False, langs: list[str] | None = None,
                  use_embed: bool = True) -> dict:
    """Poll every active channel's RSS feed and match uploads to games."""
    init_db()
    now = utcnow().replace(microsecond=0)
    stats = {"channels": 0, "videos": 0, "mentions": 0, "new_games": 0,
             "channels_empty": 0, "by_method": {}, "by_lang": {}}

    def bump(m: dict, lang: str) -> None:
        k = m.get("method", "?")
        stats["by_method"][k] = stats["by_method"].get(k, 0) + 1
        stats["by_lang"][lang] = stats["by_lang"].get(lang, 0) + 1
        stats["mentions"] += 1

    # expire_on_commit=False: this loop commits per channel, and the default
    # behaviour expires every loaded object on commit. The next attribute
    # read then triggers a fresh SELECT - a profile showed 1436 of 1662
    # queries were exactly that re-load, costing ~105 s.
    with Session(expire_on_commit=False) as s:
        # Rotation counter: derived from the clock so it advances without a
        # stored cursor. 3-hour buckets means 8 distinct values per day.
        run_no = int(now.timestamp() // 10800)
        channels = _select_channels(s, now, run_no)
        if langs:
            channels = [c for c in channels if c.lang in langs]
        stats["selected"] = len(channels)
        games = games_frame(s)
        if games.empty:
            return {**stats, "error": "no games in DB - run collect.py first"}

        place_to_uid = {int(p): int(u) for u, p in
                        zip(games.universe_id, games.root_place_id) if pd.notna(p)}

        # Built ONCE. The old code sliced `games[["universe_id", "name"]]` inside
        # the per-video loop; each call produced a new DataFrame, which defeated
        # match.py's id()-keyed index cache and rebuilt the inverted index for
        # every title. That was the 190 s.
        match_frame = games[["universe_id", "name"]]

        pending: list[tuple[dict, str]] = []
        unknown_places: dict[int, tuple[dict, str]] = {}

        # Preload into memory what the per-video checks asked the database for.
        #
        # The old code did SELECT-per-video and COUNT-per-mention. On Neon every
        # round trip is ~60 ms, and a run touches ~900 videos, so roughly 150 s
        # of the runtime was network latency, not work. Two set loads replace
        # thousands of queries.
        video_rows = {v.video_id: v for v in s.query(YtVideo).all()}
        known_videos = set(video_rows)
        # (video_id, universe_id) pairs, not just video ids: `_record_mention`
        # needs to know whether THIS game is already attached to THIS video, and
        # a video can legitimately mention several games.
        matched_pairs = {(v, int(u)) for v, u in
                         s.query(Mention.video_id, Mention.universe_id).all()}
        matched_videos = {v for v, _ in matched_pairs}
        stats["known_videos"] = len(known_videos)

        # Fetch every RSS feed BEFORE touching the database, in parallel.
        #
        # The naive loop is 150 sequential HTTP calls. Measured at ~0.40 s each
        # that is a full minute of wall time per run, and almost all of it is
        # waiting on sockets. RSS is a static XML file per channel, so 16
        # concurrent connections bring 150 channels under 5 s with no rate
        # limiting - unlike the HTML endpoints, the feed host does not throttle.
        #
        # DB writes still happen sequentially below: SQLAlchemy sessions are not
        # thread-safe, and the matching passes must stay ordered.
        live = [c for c in channels if c.channel_id]
        skipped = len(channels) - len(live)
        if skipped:
            stats["skipped_no_id"] = skipped

        feeds: dict[str, list[dict]] = {}
        if live:
            from concurrent.futures import ThreadPoolExecutor
            def fetch(cid: str) -> tuple[str, list[dict]]:
                return cid, yt.channel_uploads(cid)
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for cid, uploads in ex.map(fetch, [c.channel_id for c in live]):
                    feeds[cid] = uploads

        for ch in live:
            uploads = feeds.get(ch.channel_id, [])
            ch.last_checked = now
            stats["channels"] += 1
            if not uploads:
                stats["channels_empty"] += 1
                # Retire a feed that has been empty repeatedly. These are
                # channels that exist but post nothing, or whose id points at a
                # dead uploads playlist - retrying them every run is pure cost.
                ch.empty_polls = (ch.empty_polls or 0) + 1
                if ch.empty_polls >= MAX_EMPTY_POLLS:
                    ch.active = 0
                    stats["retired"] = stats.get("retired", 0) + 1
                continue
            ch.empty_polls = 0
            lang = ch.lang or "en"

            for u in uploads:
                u["channel_id"] = ch.channel_id
                if _record_video(s, u, now, known_videos, video_rows):
                    stats["videos"] += 1
                if u["video_id"] in matched_videos:
                    continue

                # pass 1: link in description - exact, language independent
                linked = False
                for pid in matcher.extract_place_ids(u.get("description")):
                    uid = place_to_uid.get(pid)
                    if uid:
                        m = {"universe_id": uid, "method": "link",
                             "containment": 1.0, "matched": f"place:{pid}",
                             "score": 2.0}
                        if _record_mention(s, u["video_id"], m, now, ch.channel_id, lang,
                                   matched_pairs):
                            bump(m, lang)
                        linked = True
                        break
                    unknown_places.setdefault(pid, (u, lang))
                if linked:
                    continue

                # pass 2: game name inside the title.
                #
                # Runs BEFORE the noise filter, and the order matters. The
                # filter keys on words like "all", "every" and "top", which
                # also appear in real game names - "Collect All Pets!",
                # "All Star Tower Defense", "Clean all the Cash",
                # "Harvest It All" are 7 games in this catalogue. Filtering
                # first meant a video naming one of them was discarded before
                # anything tried to match it, so those games could never
                # receive a mention. An exact name hit is stronger evidence
                # than a list-shaped title is evidence of noise.
                m = matcher.best_match(u["title"], match_frame)
                if m:
                    if _record_mention(s, u["video_id"], m, now, ch.channel_id, lang,
                                   matched_pairs):
                        bump(m, lang)
                    continue

                # Now the compilations/rankings filter, on titles that matched
                # no game by name - which is where list-shaped noise actually
                # lives.
                if matcher.suspicious_title(u["title"]):
                    continue
                pending.append((u, lang))
            # No commit here. Committing per channel cost ~180 ms each (92
            # round trips = 17 s) and bought nothing: every row is keyed, so a
            # partial run is simply re-run. One commit at the end.

        # ---- games never seen in charts, discovered through video links
        if unknown_places:
            resolved: dict[int, tuple[int, dict, str]] = {}
            for pid, (u, lang) in list(unknown_places.items())[:MAX_NEW_GAMES_PER_RUN]:
                uid = _resolve_place(pid)
                if uid:
                    resolved[pid] = (uid, u, lang)
            fresh = list(dict.fromkeys(
                uid for uid, _, _ in resolved.values() if s.get(Game, uid) is None))
            stats["new_games"] = _ingest_new_games(s, fresh)
            s.commit()
            for pid, (uid, u, lang) in resolved.items():
                if s.get(Game, uid) is None:
                    continue
                m = {"universe_id": uid, "method": "link", "containment": 1.0,
                     "matched": f"place:{pid}", "score": 2.0}
                if _record_mention(s, u["video_id"], m, now, u["channel_id"], lang,
                                   matched_pairs):
                    bump(m, lang)
            s.commit()

        # ---- pass 3: semantic match for what the first two passes missed
        if use_embed:
            import embed
            if embed.available():
                games = games_frame(s)
                lang_of = {c.channel_id: (c.lang or "en") for c in channels}

                # Queue = this run's fresh uploads PLUS the standing backlog.
                #
                # `pending` alone only ever held videos from the current RSS
                # poll (~15 uploads per channel). Anything that missed its single
                # chance was unreachable forever, because RSS never serves it
                # again. Measured 2026-10-28: 19842 videos had no mention and
                # 4847 of them were already older than three days - past the
                # feed window, permanently invisible. An earlier comment here
                # claimed they would be "picked up next run"; they were not.
                fresh = [(u.get("views") or 0, u, lang) for u, lang in pending]
                seen_ids = {u["video_id"] for _, u, _ in fresh}

                # Queue = "not yet looked at", not "no mention".
                #
                # "No mention" made the queue un-drainable: a video matching
                # nothing never gains a mention, so it returned every run until
                # it aged out - roughly 7 redundant passes each over a 7-day
                # window, and queue growth measured at 1.06x throughput, i.e.
                # it would grow forever. embed_checked is stamped after the
                # pass whether or not anything matched, so each video is
                # processed once.
                # Two conditions, not one.
                #
                # `embed_checked IS NULL` alone is not "needs embedding". A
                # video matched by pass 1 or 2 (link / exact name) already has
                # its game and needs no semantic pass - but `embed_checked` is
                # only stamped for the batch pass 3 actually processed, so those
                # videos stayed NULL and queued up as if untouched. Measured
                # 2026-10-28: 15800 pending, of which 7320 were already matched
                # (5404 token, 1164 link, 385 embed, 384 search) - nearly half
                # the queue was work that could not produce anything.
                #
                # The NOT EXISTS makes the queue mean "unmatched and unlooked-at"
                # regardless of whether a stamp was written, which is the
                # property that actually matters. 15800 -> 8465.
                rows = s.execute(text("""
                    select v.video_id, v.title, v.channel_id, v.views, v.published
                    from yt_videos v
                    where v.embed_checked is null
                      and v.published > now() - make_interval(days => :days)
                      and not exists (select 1 from mentions m
                                      where m.video_id = v.video_id)
                    order by v.views desc nulls last
                    limit :lim
                """), {"lim": MAX_EMBED_PER_RUN * 3,
                       "days": EMBED_MAX_AGE_DAYS}).fetchall()

                for vid, title, cid, views, pub in rows:
                    if vid in seen_ids:
                        continue
                    seen_ids.add(vid)
                    fresh.append((views or 0, {
                        "video_id": vid, "title": title, "channel_id": cid,
                        "published": pub,
                    }, lang_of.get(cid, "en")))

                # Most-significant-first, not first-come-first-served - but
                # significance is recency-weighted, not popularity alone.
                #
                # The previous order was "whichever channel happened to be
                # listed earlier", so a 10-view upload outranked a 49.9M-view
                # one. Ranking by raw views alone is not the fix either: tried
                # on this table it put a 520-day-old Minecraft animation and a
                # 1915-day-old Russian short at the top, because old videos have
                # had years to accumulate views. A trend indicator is worthless
                # if it is two years stale.
                #
                # views / (age_days + 1) ** 1.5 was chosen by comparing the
                # orderings on real data:
                #   raw views  -> 500-day Minecraft animations first
                #   decay only -> fresh, but still leaked unrelated shorts
                #   30d + decay-> Roblox shorts at 3.9M-10M, all <= 13 days
                # The window is a hard floor (nothing older than 30 days can be
                # evidence of what is trending now) and the decay ranks inside
                # it. 15578 videos sit in that window - 3.9x the batch - so the
                # ceiling never truncates the pool.
                #
                # `views_per_hour` would be sharper still, but it is null on
                # rows written before it was tracked and is inflated by a single
                # viral spike, so it is not used for ordering.
                def _rank(item):
                    views, u, _ = item
                    pub = u.get("published")
                    if pub is None:
                        return 0.0
                    age_days = max((now - pub).total_seconds() / 86400, 0.0)
                    return views / ((age_days + 1) ** 1.5)

                before_age = len(fresh)
                fresh = [x for x in fresh
                         if x[1].get("published") is None
                         or (now - x[1]["published"]).days <= EMBED_MAX_AGE_DAYS]
                dropped_age = before_age - len(fresh)
                fresh.sort(key=_rank, reverse=True)
                total = len(fresh)
                chosen = fresh[:MAX_EMBED_PER_RUN]
                if total > len(chosen):
                    print(f"  embed: {len(chosen)} of {total} titles "
                          f"(limit {MAX_EMBED_PER_RUN}, rest next run)",
                          flush=True)
                if dropped_age:
                    # Without this line, videos excluded by the age window
                    # simply vanish from the counts, which reads as data loss.
                    print(f"  embed: {dropped_age} title(s) older than "
                          f"{EMBED_MAX_AGE_DAYS}d skipped as stale", flush=True)

                idx = embed.build_index(
                    [(int(a), b) for a, b in zip(games.universe_id, games.name)],
                    [u["title"] for _, u, _ in chosen])
                for _, u, lang in chosen:
                    m = idx.match(u["title"])
                    if m and _record_mention(s, u["video_id"], m, now,
                                             u.get("channel_id"), lang,
                                             matched_pairs):
                        bump(m, lang)

                # One bulk UPDATE, not one per video: the row is already in
                # memory from the query above, but 4000 individual writes on
                # Neon would be ~4 minutes of round trips - the exact cost this
                # whole change exists to remove.
                if chosen:
                    s.execute(text("""
                        update yt_videos set embed_checked = :now
                        where video_id = any(:ids)
                    """), {"now": now,
                           "ids": [u["video_id"] for _, u, _ in chosen]})
                s.commit()
            else:
                stats["embed"] = "skipped: fastembed not installed"

        if discover:
            pool = games.sample(min(5, len(games)))["name"].tolist()
            for name in pool[:3]:
                for v in yt.search(f"roblox {name}", sort_by_date=True, limit=10):
                    if matcher.suspicious_title(v["title"]):
                        continue
                    m = matcher.best_match(v["title"], match_frame)
                    if m and _record_mention(s, v["video_id"], m, now, None, "en"):
                        bump(m, "en")
            s.commit()

    if stats["channels"] and stats["channels_empty"] == stats["channels"]:
        stats["note"] = (
            "Все каналы вернули пустой список. Это не баг парсера: значит "
            "YouTube не отдал RSS-ленты. Проверь, что youtube.com открывается "
            "в браузере, и запусти ещё раз."
        )
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", action="store_true",
                   help="add seed channels AND fetch their uploads")
    p.add_argument("--discover-lang", action="store_true",
                   help="find new channels per language")
    p.add_argument("--discover", action="store_true",
                   help="search for videos about tracked games")
    p.add_argument("--no-embed", action="store_true", help="skip the semantic pass")
    p.add_argument("--langs", help="comma separated, e.g. en,pt,ru")
    p.add_argument("--add", metavar="HANDLE")
    a = p.parse_args()
    langs = a.langs.split(",") if a.langs else None

    if a.add:
        init_db()
        with Session() as s:
            cid = add_channel(s, a.add, "en")
            s.commit()
        print(f"{a.add} -> {cid or 'not found'}")
    elif a.seed:
        # Seeding must also FETCH, otherwise the dashboard still shows zero
        # mentions and the user is told to run the command they just ran.
        print("seeding channels...")
        print(seed_channels(langs))
        print("fetching uploads...")
        print(poll_channels(discover=False, langs=langs,
                            use_embed=not a.no_embed))
    elif a.discover_lang:
        print(discover_all(langs))
    else:
        print(poll_channels(discover=a.discover, langs=langs,
                            use_embed=not a.no_embed))
