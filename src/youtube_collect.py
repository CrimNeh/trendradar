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


def _record_video(s, u: dict, now) -> bool:
    """One row per video; views only grow, so refresh in place."""
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


def _record_mention(s, video_id: str, m: dict, now, channel_id, lang) -> bool:
    if s.query(Mention).filter(Mention.video_id == video_id,
                               Mention.universe_id == m["universe_id"]).count():
        return False
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
    """Add games found via video links that no chart has shown us yet."""
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
    return added


# ------------------------------------------------------------------ poll

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

    with Session() as s:
        q = s.query(YtChannel).filter(YtChannel.active == 1)
        if langs:
            q = q.filter(YtChannel.lang.in_(langs))
        channels = q.all()
        games = games_frame(s)
        if games.empty:
            return {**stats, "error": "no games in DB - run collect.py first"}

        place_to_uid = {int(p): int(u) for u, p in
                        zip(games.universe_id, games.root_place_id) if pd.notna(p)}

        pending: list[tuple[dict, str]] = []
        unknown_places: dict[int, tuple[dict, str]] = {}

        for ch in channels:
            uploads = yt.channel_uploads(ch.channel_id)
            ch.last_checked = now
            stats["channels"] += 1
            if not uploads:
                stats["channels_empty"] += 1
            lang = ch.lang or "en"

            for u in uploads:
                u["channel_id"] = ch.channel_id
                if _record_video(s, u, now):
                    stats["videos"] += 1
                if _already_matched(s, u["video_id"]):
                    continue

                # pass 1: link in description - exact, language independent
                linked = False
                for pid in matcher.extract_place_ids(u.get("description")):
                    uid = place_to_uid.get(pid)
                    if uid:
                        m = {"universe_id": uid, "method": "link",
                             "containment": 1.0, "matched": f"place:{pid}",
                             "score": 2.0}
                        if _record_mention(s, u["video_id"], m, now, ch.channel_id, lang):
                            bump(m, lang)
                        linked = True
                        break
                    unknown_places.setdefault(pid, (u, lang))
                if linked:
                    continue

                if matcher.suspicious_title(u["title"]):
                    continue

                # pass 2: game name inside the title
                m = matcher.best_match(u["title"], games[["universe_id", "name"]])
                if m:
                    if _record_mention(s, u["video_id"], m, now, ch.channel_id, lang):
                        bump(m, lang)
                    continue
                pending.append((u, lang))
            s.commit()

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
                if _record_mention(s, u["video_id"], m, now, u["channel_id"], lang):
                    bump(m, lang)
            s.commit()

        # ---- pass 3: semantic, only for what is still unmatched
        if use_embed and pending:
            import embed
            if embed.available():
                games = games_frame(s)
                idx = embed.build_index(
                    [(int(a), b) for a, b in zip(games.universe_id, games.name)],
                    [u["title"] for u, _ in pending])
                for u, lang in pending:
                    m = idx.match(u["title"])
                    if m and _record_mention(s, u["video_id"], m, now,
                                             u["channel_id"], lang):
                        bump(m, lang)
                s.commit()
            else:
                stats["embed"] = "skipped: fastembed not installed"

        if discover:
            pool = games.sample(min(5, len(games)))["name"].tolist()
            for name in pool[:3]:
                for v in yt.search(f"roblox {name}", sort_by_date=True, limit=10):
                    if matcher.suspicious_title(v["title"]):
                        continue
                    m = matcher.best_match(v["title"], games[["universe_id", "name"]])
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
