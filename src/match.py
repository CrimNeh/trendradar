"""Match YouTube videos to tracked Roblox games.

Two passes, in this order:

1. LINK PASS (exact). Many creators paste the Roblox game URL into the video
   description. A link gives the placeId, and places resolve to universes -
   zero ambiguity, works in every language. Measured on live data: only 4% of
   videos carry a link, but when they do it is always correct, and it catches
   games whose name never appears in the title at all.

2. TOKEN PASS (fast, free). One-directional containment on content tokens.
   Handles the common case where the game name appears verbatim in the title,
   including inside a foreign-language title ("Я ПРОШЁЛ STEAL AN EGG").

3. EMBEDDING PASS (slow, optional). Only for titles the token pass missed.
   Semantic, multilingual, hubness-corrected - see embed.py for the numbers
   behind its threshold. This is what catches a Russian title that says
   "ЯЙЦО, НО Я ПОКУПАЮ ПИТОМЦЕВ" for "Steal A Monkey Egg".

Token noise words and the "one short token" guard are kept from the original
implementation because they were calibrated on live data.
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from templates import tokens  # same normaliser as the family detector

NOISE = {
    "roblox", "game", "play", "playing", "played", "video", "new", "best",
    "worst", "funny", "moment", "moments", "official", "trailer", "update",
    "live", "stream", "streaming", "part", "full", "movie", "edit", "song",
    "im", "my", "me", "we", "us", "you", "this", "that", "the", "and", "for",
    "with", "from", "about", "into", "get", "got", "make", "made", "doing",
    "did", "going", "went", "why", "how", "what", "when", "where", "who",
    "not", "yes", "all", "every", "ever", "never", "again", "first", "last",
    "time", "day", "days", "hour", "hours", "insane", "crazy",
}

# Roblox links inside video descriptions.
PLACE_LINK = re.compile(
    r"roblox\.com/(?:[a-z]{2}(?:-[a-z]{2})?/)?games/(\d+)", re.I)


_DECOR = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")
_EMO = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2B50\u2705\u274C]")


def _strip_decor(name: str) -> str:
    return _EMO.sub(" ", _DECOR.sub(" ", name or ""))


def title_tokens(text: str) -> set[str]:
    return {t for t in tokens(text, strip_brackets=False)
            if t not in NOISE and len(t) >= 3}


def extract_place_ids(description: str | None) -> list[int]:
    """placeIds referenced in a video description, in order, deduped."""
    if not description:
        return []
    seen: list[int] = []
    for raw in PLACE_LINK.findall(description):
        pid = int(raw)
        if pid not in seen:
            seen.append(pid)
    return seen


_INDEX_CACHE: dict[int, tuple] = {}


def _candidates(tt: set, games: pd.DataFrame):
    """Only games sharing at least one token with the title.

    The original loop compared every title against every game. With 863 videos
    and 452 games that is 390k set operations per run, and it dominated the
    runtime: RSS fetch took 0.9 s while matching took ~190 s.

    An inverted index keyed on token turns that into a few dict lookups. The
    cache is keyed on id() and holds a reference to the frame, so a rebuilt
    frame cannot be served stale entries.
    """
    key = id(games)
    cached = _INDEX_CACHE.get(key)
    if cached is None or cached[0] is not games:
        index: dict = {}
        for row in games.itertuples():
            toks = {t for t in tokens(row.name) if t not in NOISE and len(t) >= 3}
            for t in toks:
                index.setdefault(t, []).append(row)
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = (games, index)
        cached = _INDEX_CACHE[key]

    index = cached[1]
    seen: set = set()
    for t in tt:
        for row in index.get(t, ()):
            if row.Index in seen:
                continue
            seen.add(row.Index)
            yield row


def match_video(title: str, games: pd.DataFrame,
                min_containment: float = 0.75) -> list[dict]:
    """Games whose names are substantially covered by the video title.

    A lone matching token is accepted only if it is long (>=5 chars) and the
    game name is short (<=2 tokens). "DOORS" is legitimately one token;
    "Cat Battles" matching a video that only says "cat" is not.
    """
    tt = title_tokens(title)
    if not tt or games.empty:
        return []
    title_words = {w.lower() for w in re.findall(r"[^\W\d_][\w'’]*", title or "")}

    hits = []
    for row in _candidates(tt, games):
        gt = {t for t in tokens(row.name) if t not in NOISE and len(t) >= 3}
        if not gt:
            continue
        overlap = gt & tt
        if not overlap:
            continue
        containment = len(overlap) / len(gt)
        if containment < min_containment:
            continue
        if len(overlap) == 1:
            only = next(iter(overlap))
            # Words in the ORIGINAL name, before noise/stop words were removed.
            # "Money Tycoon" reduces to {money} but is a two-word name, so a
            # lone "money" in a title is not evidence. A single-word name
            # ("Jailbreak", "DOORS") may match on its word; a longer name may
            # match on one token only if that token is long enough to be a
            # proper noun ("Brookhaven").
            raw_words = len(re.findall(r"[^\W\d_]{2,}", _strip_decor(row.name)))
            if raw_words <= 1:
                # single-word name: the word must stand alone. "door" inside
                # "PICK A DOOR" is a common noun, not the game "DOORS".
                if len(only) < 4:
                    continue
                # compare SURFACE forms: stemming turns both "doors" and
                # "door" into "door", which is exactly the false match.
                surface = {w.lower() for w in re.findall(
                    r"[^\W\d_][\w'’]*", _strip_decor(row.name))}
                if not (surface & title_words):
                    continue
            elif len(only) < 8:
                continue
        hits.append({
            "universe_id": int(row.universe_id),
            "game_name": row.name,
            "containment": round(containment, 3),
            "matched": " ".join(sorted(overlap)),
            "specificity": len(overlap),
            "method": "token",
            "score": round(containment, 3),
        })

    hits.sort(key=lambda h: (-h["specificity"], -h["containment"]))
    return hits[:3]


def best_match(title: str, games: pd.DataFrame) -> dict | None:
    h = match_video(title, games)
    return h[0] if h else None


def suspicious_title(title: str) -> bool:
    """Compilations and rankings rarely indicate a specific game trend.

    Only list-shaped titles are noise. "1,000 Robux in Tower of Hell" is a
    real game video, so the rule keys on "N games", never on currency.
    """
    t = (title or "").lower()
    if re.search(r"\b(top|ranking|tier list|every|all)\b", t):
        return True
    if re.search(r"\b\d+\s+(new\s+)?(roblox\s+)?games\b", t):
        return True
    return False


if __name__ == "__main__":
    from db import Game, Session

    with Session() as s:
        games = pd.DataFrame(
            s.query(Game.universe_id, Game.name).all(),
                             columns=["universe_id", "name"])

    samples = [
        "ROBLOX SELL FRIES!",
        "Break and Steal an Egg",
        "I Built MY DREAM Base in Roblox 99 Nights in the Forest",
        "ROBLOX STEAL A CAR..",
        "Ranking Every GTA Main Character",
        "1 Floor = 1,000 Robux in Tower of Hell!",
        "PICK A DOOR",
    ]
    for t in samples:
        m = best_match(t, games)
        flag = " [compilation]" if suspicious_title(t) else ""
        print(f"{t[:48]:50s} -> "
              f"{(m['game_name'][:24] + ' (' + str(m['containment']) + ')') if m else '—'}{flag}")
