"""Bridge external signals to Roblox games.

A Steam chart entry or an upcoming anime is only actionable if it connects
to something you could build. This module answers: which Roblox niches do
these origin signals point at, and is there already a game proving demand?

Three linkages:

1. GENRE -> NICHE. Steam tags and anime genres map onto Roblox niches.
   "survival" on Steam maps to the `survival` niche in classify.py.

2. TOKEN OVERLAP. An anime title or Steam game sharing distinctive tokens
   with a Roblox game name is a direct IP-clone signal - exactly what the
   `blue lock` family showed us, caught after the fact. This catches it
   before, while the IP is still upcoming.

3. NICHE PRESSURE. For each niche, how many origin signals point at it,
   weighted by their score. A niche with rising external pressure and few
   young Roblox games is an open window.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from templates import tokens  # noqa: E402

# Steam tags / anime genres -> Roblox niche labels used by classify.py.
GENRE_MAP: dict[str, list[str]] = {
    "survival": ["survival", "building", "stealth"],
    "simulation": ["tycoon", "simulator", "farm_garden", "roleplay"],
    "casual": ["simulator", "puzzle", "dress_up", "farm_garden"],
    "adventure": ["story", "rpg", "climbing"],
    "action": ["arena", "fighting", "shooter"],
    "rpg": ["rpg", "story", "card_battle"],
    "strategy": ["tower_defense", "rpg"],
    "indie": ["puzzle", "story", "horror"],
    "massively multiplayer": ["rpg", "roleplay", "arena"],
    "racing": ["racing"],
    "sports": ["sports", "racing"],
    "horror": ["horror"],
    "mystery": ["social_deduction", "horror", "puzzle"],
    "psychological": ["horror", "story", "social_deduction"],
    "comedy": ["social_deduction", "roleplay"],
    "romance": ["roleplay", "dress_up", "story"],
    "fantasy": ["rpg", "story", "arena"],
    "sci-fi": ["rpg", "arena", "shooter"],
    "slice of life": ["roleplay", "farm_garden"],
    "music": ["dress_up", "roleplay"],
    "gaming": ["simulator", "arena"],
}

# Generic words that appear in both anime titles and Roblox names without
# implying any IP relationship. "dungeon", "infinity", "final", "quest" all
# produced false pairs, so the rule is: a shared token must look like a
# distinct proper noun, not a common genre word.
WEAK_TOKENS = {
    "season", "game", "story", "world", "night", "days", "life", "love",
    "dark", "star", "part", "movie", "online", "original", "edition", "episode",
    "dungeon", "infinity", "final", "quest", "treasure", "slayer", "hunter",
    "hunters", "piece", "blade", "sword", "hero", "king", "queen", "magic",
    "academy", "school", "city", "land", "saga", "chronicle", "adventure",
    "simulator", "chapter", "battle", "war", "fire", "water", "light", "shadow",
    "angel", "demon", "dragon", "ghost", "soul", "heart", "dream", "time",
}

# Minimum length for a shared token to count as an IP name.
MIN_IP_TOKEN = 6


def direct_ip_matches(sig: pd.DataFrame, games: pd.DataFrame,
                      limit: int = 25) -> pd.DataFrame:
    """Anime titles sharing a distinctive NAME with a Roblox game.

    The pattern is the `blue lock` family: an IP name appears in Roblox as
    clones. Catching it while the anime is still upcoming is the point.

    Deliberately strict. Sharing "dungeon" between "Delicious in Dungeon" and
    "Dungeon Quest" is a genre coincidence, not a clone signal, so weak
    tokens are dropped and a longer minimum length is required. Missing a
    real pair is cheaper than crying wolf on every RPG.
    """
    if sig.empty or games.empty:
        return pd.DataFrame()

    anime = sig[sig["source"] == "anime"]
    if anime.empty:
        return pd.DataFrame()

    game_tokens = {
        r.universe_id: {t for t in tokens(r.name)
                        if t not in WEAK_TOKENS and len(t) >= MIN_IP_TOKEN}
        for r in games.itertuples()
    }

    out = []
    for r in anime.itertuples():
        at = {t for t in tokens(r.title)
              if t not in WEAK_TOKENS and len(t) >= MIN_IP_TOKEN}
        if not at:
            continue
        for g in games.itertuples():
            gt = game_tokens.get(g.universe_id, set())
            if not gt:
                continue
            shared = at & gt
            if not shared:
                continue
            days = None
            try:
                days = json.loads(r.meta).get("days_to_premiere")
            except (json.JSONDecodeError, TypeError):
                pass
            out.append({
                "ip": r.title[:60],
                "game": g.name[:40],
                "shared": " ".join(sorted(shared)),
                "niche": g.niche,
                "days_to_premiere": days,
                "ip_score": r.score,
            })

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).sort_values(["ip_score", "days_to_premiere"],
                                       ascending=[False, True])
    return df.drop_duplicates(subset=["ip", "game"]).head(limit).reset_index(drop=True)


def _load_signals() -> pd.DataFrame:
    from db import OriginSignal, Session

    with Session() as s:
        rows = [(r.source, r.title, r.score, r.tag or "", r.meta or "{}")
                for r in s.query(OriginSignal)]
    return pd.DataFrame(rows, columns=["source", "title", "score", "tag", "meta"])


def _load_games() -> pd.DataFrame:
    from db import Game, Session

    with Session() as s:
        rows = [(g.universe_id, g.name, g.niche or "other", g.created)
                for g in s.query(Game)]
    return pd.DataFrame(rows, columns=["universe_id", "name", "niche", "created"])


def signals_by_niche(sig: pd.DataFrame) -> pd.DataFrame:
    """Weighted external pressure per Roblox niche."""
    if sig.empty:
        return pd.DataFrame()

    acc: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for r in sig.itertuples():
        tags = [t.strip().lower() for t in (r.tag or "").split(",") if t.strip()]
        for tag in tags:
            for niche in GENRE_MAP.get(tag, []):
                acc[niche] += r.score
                counts[niche] += 1

    out = pd.DataFrame({
        "niche": list(acc),
        "ext_pressure": [round(v, 1) for v in acc.values()],
        "ext_count": [counts[n] for n in acc],
    })
    return out.sort_values("ext_pressure", ascending=False).reset_index(drop=True)


def origin_brief(sig: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Everything the dashboard and report need, computed once."""
    if sig.empty:
        return {"niches": pd.DataFrame(), "ips": pd.DataFrame(),
                "top": pd.DataFrame()}

    top = sig.nlargest(30, "score")[["source", "title", "score", "tag"]].copy()

    niches = signals_by_niche(sig)
    if not niches.empty and not games.empty:
        live = games.groupby("niche").size().to_dict()
        niches["roblox_games"] = niches["niche"].map(lambda n: live.get(n, 0))
        # Pressure per existing game: high external interest with few games
        # already serving it is the open-window signal.
        niches["window"] = niches.apply(
            lambda r: round(r["ext_pressure"] / max(r["roblox_games"], 1), 1),
            axis=1)
        niches = niches.sort_values("window", ascending=False).reset_index(drop=True)

    return {"niches": niches, "ips": direct_ip_matches(sig, games), "top": top}


if __name__ == "__main__":
    sig, games = _load_signals(), _load_games()
    print(f"signals={len(sig)} games={len(games)}\n")
    b = origin_brief(sig, games)
    pd.set_option("display.width", 200)
    if not b["niches"].empty:
        print("=== niches by open window ===")
        print(b["niches"].head(12).to_string(index=False))
    print("\n=== direct IP matches ===")
    print(b["ips"].head(12).to_string(index=False) if not b["ips"].empty else "none yet")
