"""Template-family detection.

The actionable signal for someone who wants to BUILD a game is not "this
title is rising" but "this SHAPE of game is being cloned right now".

Roblox names are formulaic. When several unrelated studios launch games off
the same name skeleton within weeks, that is a demand pocket being validated
in public - and the window to enter is still open.

1. Normalise names, drop decorators, extract content tokens.
2. Drop tokens that appear in a large share of all games (generic glue).
3. Union-find over games sharing token pairs seen in 2+ games.
4. Evict chained members that do not share enough with the cluster core.
5. Score families: member count, aggregate CCU, mean age, growth.
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from collections import defaultdict
from itertools import combinations

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

STOP = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "my", "your",
    "new", "update", "upd", "beta", "alpha", "test", "testing", "early",
    "access", "official", "remastered", "reborn", "2", "3", "4", "5", "v2",
    "simulator", "sim", "game", "games", "experience", "obby", "tycoon",
    "roblox", "edit", "edition", "pro", "hd", "x", "vs",
}

EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u2B50\u2705\u274C\u2B06\u2197\u3030]"
)


def tokens(name: str | None, strip_brackets: bool = True) -> list[str]:
    """Content tokens for matching and family detection.

    strip_brackets: True for game names, where [UPD] / (BETA) is decoration.
    False for video titles, where "(ROBLOX Jailbreak)" IS the game name.


    Non-ASCII words are kept WHOLE and never stemmed. Splitting them at the
    first accented character is what caused a real false positive: the
    Portuguese "diversões" (amusement rides) fragmented into "diver" and
    matched the game "Crazy Diver". A word in another script is either an
    exact match or it is not a match at all.
    """
    if not name:
        return []
    s = EMOJI.sub(" ", name)
    if strip_brackets:
        s = re.sub(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}", " ", s)
    else:
        s = re.sub(r"[\[\](){}]", " ", s)
    s = s.lower()

    out = []
    for raw in s.split():
        if any(ord(ch) > 127 for ch in raw):
            # keep non-Latin / accented words intact
            w = re.sub(r"[^\w]", "", raw, flags=re.UNICODE)
            if len(w) >= 3 and w not in STOP:
                out.append(w)
            continue
        w = re.sub(r"[^a-z0-9]", "", raw)
        if not w or w in STOP or len(w) < 3 or w.isdigit():
            continue
        if w.endswith("ies") and len(w) > 5:
            w = w[:-3] + "y"
        elif w.endswith("es") and len(w) > 4 and w[-3] in "sxz":
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3:
            w = w[:-1]
        out.append(w)
    return out


def family_key(core_tokens: list[str]) -> str:
    """Deterministic id for a template: hash of its sorted core tokens."""
    import hashlib
    raw = "|".join(sorted(core_tokens))
    return "f-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def build_families(f: pd.DataFrame, min_shared: int = 2,
                   max_df: float = 0.06) -> pd.DataFrame:
    if f.empty:
        return pd.DataFrame()

    tok = {r.universe_id: tokens(r.name) for r in f.itertuples()}

    df_count: dict[str, int] = defaultdict(int)
    for toks in tok.values():
        for t in set(toks):
            df_count[t] += 1
    n = max(len(tok), 1)
    generic = {t for t, c in df_count.items() if c / n > max_df}
    tok = {u: [t for t in ts if t not in generic] for u, ts in tok.items()}

    pair_count: dict[tuple[str, str], int] = defaultdict(int)
    for toks in tok.values():
        for a, b in combinations(sorted(set(toks)), 2):
            pair_count[(a, b)] += 1
    strong = {p for p, c in pair_count.items() if c >= 2}

    parent: dict[int, int] = {u: u for u in tok}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_owner: dict[tuple[str, str], int] = {}
    for uid, toks in tok.items():
        for a, b in combinations(sorted(set(toks)), 2):
            p = (a, b)
            if p not in strong:
                continue
            if p in pair_owner:
                union(uid, pair_owner[p])
            else:
                pair_owner[p] = uid

    fam = {uid: find(uid) for uid in tok}

    # Evict chained members: a game must share >=min_shared tokens with the
    # modal signature of its own cluster, otherwise it becomes a singleton.
    clusters: dict[int, list[int]] = defaultdict(list)
    for uid, fid in fam.items():
        clusters[fid].append(uid)

    for fid, members in clusters.items():
        cnt: dict[str, int] = defaultdict(int)
        for uid in members:
            for t in set(tok[uid]):
                cnt[t] += 1
        core = {t for t, c in cnt.items() if c >= max(2, len(members) // 2)}
        if not core:
            continue
        for uid in members:
            if len(core & set(tok[uid])) < min_shared:
                fam[uid] = uid

    f = f.copy()
    f["family_id"] = f["universe_id"].map(fam)

    # The union-find root is just "whichever game was seen first", so it
    # changes between runs and cannot be used to track a template over time.
    # Key each family by its core tokens instead: same template -> same key.
    labels: dict[int, str] = {}
    keys: dict[int, str] = {}
    for fid, grp in f.groupby("family_id"):
        cnt: dict[str, int] = defaultdict(int)
        for nm in grp["name"]:
            for t in set(tokens(nm)):
                cnt[t] += 1
        top = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
        labels[fid] = " ".join(t for t, _ in top) if top else "misc"
        if len(grp) >= 2 and top:
            keys[fid] = family_key([t for t, _ in top])
        else:
            keys[fid] = f"solo-{int(grp['universe_id'].iloc[0])}"
    f["family"] = f["family_id"].map(labels)
    f["family_key"] = f["family_id"].map(keys)

    sizes = f.groupby("family_id").size()
    f["family_size"] = f["family_id"].map(sizes).fillna(1).astype(int)
    return f


def family_summary(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty or "family_id" not in f.columns:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agg = f.groupby("family_id").agg(
            family_key=("family_key", "first"),
            family=("family", "first"),
            games=("universe_id", "count"),
            total_ccu=("ccu", "sum"),
            max_ccu=("ccu", "max"),
            mean_age=("age_days", lambda x: (
                float(x[x >= 0].mean()) if (x >= 0).any() else -1.0)),
            mean_score=("score", "mean"),
            best_score=("score", "max"),
            members=("name", lambda x: " | ".join(list(x)[:4])),
        ).reset_index()

    gcols = [c for c in ("g_d1", "g_d3") if c in f.columns]
    if gcols:
        agg = agg.merge(f.groupby("family_id")[gcols].median().reset_index(),
                        on="family_id", how="left")

    agg = agg[agg["games"] >= 2].copy()

    def enter_score(r):
        age = r["mean_age"]
        # Age is a hard gate, not a bonus: a family averaging 3 years old is a
        # settled genre with incumbents, not an open window.
        if age < 0:
            age_f = 0.5
        elif age <= 45:
            age_f = 1.0
        elif age <= 120:
            age_f = 0.85
        elif age <= 240:
            age_f = 0.55
        elif age <= 500:
            age_f = 0.25
        else:
            age_f = 0.05

        base = min(r["games"] / 3.0, 2.0) * 15 + min(r["total_ccu"] / 15000.0, 2.0) * 20
        s = base * age_f
        g = r.get("g_d1")
        if g is not None and pd.notna(g) and g > 1.2:
            s += min((g - 1) * 12, 20)
        s += min(r["best_score"] / 100.0, 1.0) * 10 * age_f
        return round(min(s, 100.0), 1)

    agg["enter_score"] = agg.apply(enter_score, axis=1)
    return agg.sort_values("enter_score", ascending=False).reset_index(drop=True)
