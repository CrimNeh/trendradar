"""Feature engineering + rising score.

Works from snapshots alone, so it can be recomputed and backfilled cheaply.

Signals combined here:
  - Roblox momentum (6h/1d/3d/7d growth, acceleration, chart position)
  - YouTube pressure, split by language market
  - paid promotion flag (isSponsored) - a studio spending money is a bet
  - retention proxies (favourites per visit, like ratio)
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from db import Game, Mention, Session, Snapshot, YtVideo  # noqa: E402

H = {"h6": 6 / 24, "d1": 1, "d3": 3, "d7": 7}

# Language buckets reported separately: these are where Roblox is growing.
MAJOR_LANGS = ("en", "es", "pt", "ru", "id")


def _snapshots_df() -> pd.DataFrame:
    with Session() as s:
        rows = s.query(
            Snapshot.universe_id, Snapshot.ts, Snapshot.playing, Snapshot.visits,
            Snapshot.favorites, Snapshot.up_votes, Snapshot.down_votes,
            Snapshot.max_players, Snapshot.servers, Snapshot.avg_fill,
            Snapshot.chart_count, Snapshot.best_chart_rank, Snapshot.is_sponsored,
        ).all()
    df = pd.DataFrame(rows, columns=[
        "universe_id", "ts", "playing", "visits", "favorites", "up_votes",
        "down_votes", "max_players", "servers", "avg_fill", "chart_count",
        "best_chart_rank", "is_sponsored",
    ])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def _games_df() -> pd.DataFrame:
    with Session() as s:
        rows = s.query(
            Game.universe_id, Game.name, Game.niche, Game.tags, Game.creator_id,
            Game.creator_name, Game.created, Game.icon_url, Game.first_seen,
        ).all()
    df = pd.DataFrame(rows, columns=[
        "universe_id", "name", "niche", "tags", "creator_id", "creator_name",
        "created", "icon_url", "first_seen",
    ])
    if not df.empty:
        for c in ("created", "first_seen"):
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def youtube_features(now: pd.Timestamp) -> pd.DataFrame:
    """Per-game YouTube pressure over the last 7 days, split by language.

    yt_videos   : total videos mentioning the game
    yt_views    : sum of their view counts
    yt_velocity : views per hour, summed - catches a video going viral
    yt_age_h    : hours since the first tracked mention (low = fresh buzz)
    yt_langs    : how many distinct language markets covered it
    yt_non_en   : mentions outside English - where growth actually is now
    """
    cols = ["universe_id", "yt_videos", "yt_views", "yt_velocity", "yt_age_h",
            "yt_langs", "yt_non_en"]

    with Session() as s:
        m = s.query(Mention.universe_id, Mention.video_id, Mention.observed,
                    Mention.lang).all()
        v = s.query(YtVideo.video_id, YtVideo.views, YtVideo.views_per_hour,
                    YtVideo.published).all()

    if not m or not v:
        return pd.DataFrame(columns=cols)

    md = pd.DataFrame(m, columns=["universe_id", "video_id", "observed", "lang"])
    vd = pd.DataFrame(v, columns=["video_id", "views", "vph", "published"])
    vd["published"] = pd.to_datetime(vd["published"], utc=True)
    md["observed"] = pd.to_datetime(md["observed"], utc=True)

    vd_last = vd.sort_values("published").drop_duplicates("video_id", keep="last")
    merged = md.merge(vd_last, on="video_id", how="left")

    week_ago = now - timedelta(days=7)
    merged = merged[(merged["observed"] >= week_ago) | (merged["published"] >= week_ago)]
    if merged.empty:
        return pd.DataFrame(columns=cols)

    g = merged.groupby("universe_id").agg(
        yt_videos=("video_id", "nunique"),
        yt_views=("views", "sum"),
        yt_velocity=("vph", "sum"),
        yt_first=("published", "min"),
    ).reset_index()
    g["yt_age_h"] = ((now - g["yt_first"]).dt.total_seconds() / 3600).round(1)

    langs = merged.groupby("universe_id")["lang"].agg(
        lambda x: len({v for v in x if v and not pd.isna(v)}))
    non_en = merged[merged["lang"].fillna("en") != "en"].groupby("universe_id").size()
    g["yt_langs"] = g["universe_id"].map(langs).fillna(0).astype(int)
    g["yt_non_en"] = g["universe_id"].map(non_en).fillna(0).astype(int)

    return g[cols]


def _value_at(sub: pd.DataFrame, col: str, days_ago: float, now: pd.Timestamp):
    """Nearest value to `days_ago` back, or None if the match is too far off.

    A 3-day comparison must not silently use a 20-minute-old sample.
    """
    if sub.empty:
        return None
    target = now - timedelta(days=days_ago)
    delta = (sub["ts"] - target).abs()
    tol = timedelta(days=max(days_ago * 0.5, 0.02))
    if delta.min() > tol:
        return None
    return sub.loc[delta.idxmin(), col]


def _ratio(new, old, floor=30.0):
    if new is None or old is None or pd.isna(new) or pd.isna(old):
        return None
    return float(new) / max(float(old), floor)


def build_features() -> pd.DataFrame:
    snaps = _snapshots_df()
    games = _games_df()
    if snaps.empty or games.empty:
        return pd.DataFrame()

    now = snaps["ts"].max()
    latest = snaps.sort_values("ts").groupby("universe_id").tail(1).set_index("universe_id")
    span_h = snaps.groupby("universe_id")["ts"].agg(
        lambda x: (x.max() - x.min()).total_seconds() / 3600)
    counts = snaps.groupby("universe_id").size()
    gm = games.set_index("universe_id")

    rows = []
    for uid, cur in latest.iterrows():
        sub = snaps[snaps["universe_id"] == uid]
        rec = {
            "universe_id": uid,
            "ccu": int(cur["playing"] or 0),
            "visits": int(cur["visits"] or 0),
            "favorites": int(cur["favorites"] or 0),
            "chart_count": int(cur["chart_count"] or 0),
            "best_rank": cur["best_chart_rank"],
            "avg_fill": cur["avg_fill"],
            "servers": cur["servers"],
            "sponsored": int(cur["is_sponsored"] or 0),
            "history_hours": round(float(span_h.get(uid, 0)), 1),
            "snapshots": int(counts.get(uid, 0)),
        }
        for hname, days in H.items():
            rec[f"ccu_{hname}"] = _value_at(sub, "playing", days, now)
            rec[f"rank_{hname}"] = _value_at(sub, "best_chart_rank", days, now)
            rec[f"g_{hname}"] = _ratio(rec["ccu"], rec[f"ccu_{hname}"], floor=30)

        up, down = cur["up_votes"] or 0, cur["down_votes"] or 0
        rec["like_ratio"] = round(up / (up + down), 3) if (up + down) else None
        rec["fav_per_1k_visits"] = (
            round(1000 * rec["favorites"] / rec["visits"], 2) if rec["visits"] else None)
        rec["ccu_per_1k_visits"] = (
            round(1000 * rec["ccu"] / rec["visits"], 3) if rec["visits"] else None)

        created = gm["created"].get(uid) if uid in gm.index else None
        if pd.notna(created):
            c = pd.Timestamp(created)
            c = c.tz_localize("UTC") if c.tzinfo is None else c.tz_convert("UTC")
            rec["age_days"] = int((now - c).days)
        else:
            rec["age_days"] = -1

        # Day-one signal: current CCU vs lifetime average daily visits.
        # Raw ratio is misleading: an abandoned 2013 game has an enormous
        # denominator history and any spark looks explosive. So the raw value
        # is only kept, and the score uses its percentile inside the age
        # cohort (computed below), never the absolute number.
        if rec["age_days"] > 0 and rec["visits"] > 0:
            per_day = rec["visits"] / rec["age_days"]
            rec["lifetime_ratio"] = round(rec["ccu"] / per_day, 4) if per_day else None
        else:
            rec["lifetime_ratio"] = None

        if rec["best_rank"] and rec["rank_d1"] and not pd.isna(rec["rank_d1"]):
            rec["rank_delta_d1"] = float(rec["best_rank"] - rec["rank_d1"])
        else:
            rec["rank_delta_d1"] = None

        rows.append(rec)

    f = pd.DataFrame(rows).merge(
        games[["universe_id", "name", "niche", "tags", "creator_id",
               "creator_name", "created", "icon_url", "first_seen"]],
        on="universe_id", how="left",
    )

    yt = youtube_features(now)
    if not yt.empty:
        f = f.merge(yt, on="universe_id", how="left")
    for c in ("yt_videos", "yt_views", "yt_velocity", "yt_age_h", "yt_langs", "yt_non_en"):
        if c not in f.columns:
            f[c] = np.nan

    f = _add_cohorts(f)

    f["score"] = f.apply(rising_score, axis=1)
    f["confidence"] = f.apply(_confidence, axis=1)
    return f.sort_values("score", ascending=False).reset_index(drop=True)


AGE_BINS = [-1, 30, 90, 365, 1095, 100_000]
SIZE_BINS = [-1, 500, 5_000, 50_000, 10**9]


def _add_cohorts(f: pd.DataFrame) -> pd.DataFrame:
    """Percentile of each game inside its age x size cohort.

    Absolute scores are not comparable across eras: a rising 2026 game and a
    decaying 2014 game can share a number. Ranking inside a cohort makes
    "top 5% of its peer group" the meaningful statement.
    """
    if f.empty:
        return f
    f = f.copy()
    age = f["age_days"].clip(lower=0)
    f["age_cohort"] = pd.cut(age, AGE_BINS,
                             labels=["<1м", "1-3м", "3-12м", "1-3г", "3г+"])
    f["size_cohort"] = pd.cut(f["ccu"].clip(lower=0), SIZE_BINS,
                              labels=["<500", "500-5k", "5k-50k", "50k+"])
    grp = f.groupby(["age_cohort", "size_cohort"], observed=True)

    def pct(s: pd.Series) -> pd.Series:
        """Rank inside the cohort; NaN where the cohort is too small to rank."""
        if len(s.dropna()) < 3:
            return pd.Series(np.nan, index=s.index)
        return s.rank(pct=True)

    f["r_g6_pct"] = grp["g_h6"].transform(pct)
    f["r_g1_pct"] = grp["g_d1"].transform(pct)
    f["r_life_pct"] = grp["lifetime_ratio"].transform(pct)

    # Weighted rank over the components that actually exist. Missing inputs
    # are dropped and the weights renormalised - filling them with 0.5 made
    # every game in a cohort tie at the same value.
    weights = {"r_g1_pct": 0.45, "r_g6_pct": 0.25, "r_life_pct": 0.30}
    num = pd.Series(0.0, index=f.index)
    den = pd.Series(0.0, index=f.index)
    for col, w in weights.items():
        present = f[col].notna()
        num = num.add(f[col].fillna(0.0) * w * present, fill_value=0.0)
        den = den.add(pd.Series(np.where(present, w, 0.0), index=f.index),
                      fill_value=0.0)
    f["cohort_rank"] = ((num / den.replace(0, np.nan)) * 100).round(1)
    return f


def rising_score(r: pd.Series) -> float:
    """0-100. Returns 0 when there is no momentum evidence at all."""
    parts, weights = [], []
    g1, g3, g6 = r.get("g_d1"), r.get("g_d3"), r.get("g_h6")
    has_growth = any(x is not None and not pd.isna(x) for x in (g1, g3, g6))

    if not has_growth:
        # No deltas yet: fall back to the peer-group percentile, which is
        # scale-free and does not reward ancient abandoned games.
        pct = r.get("cohort_rank")
        if pct is None or pd.isna(pct):
            return 0.0
        # A pure percentile gives every cohort leader exactly 100, so "best of
        # a dull cohort" looks as strong as a real surge. Blend in the absolute
        # level (log-scaled, bounded) to separate them and break ties.
        lr = r.get("lifetime_ratio")
        absolute = 0.0
        if lr is not None and not pd.isna(lr) and lr > 0:
            absolute = float(np.clip((np.log10(lr) + 2.0) / 3.0, 0.0, 1.0)) * 100
        blended = 0.7 * float(np.clip(pct, 0, 100)) + 0.3 * absolute
        return round(blended * 0.6, 1)   # capped at 60: no history, low trust

    if g1 is not None and not pd.isna(g1):
        parts.append(np.log2(max(g1, 0.01)) + 1.0)
        weights.append(0.26)
    if g3 is not None and not pd.isna(g3):
        parts.append(np.log2(max(g3, 0.01)) / 1.58 + 0.5)
        weights.append(0.18)
    if g6 is not None and not pd.isna(g6):
        parts.append(np.log2(max(g6, 0.01)) * 1.5 + 0.5)
        weights.append(0.13)
    if all(x is not None and not pd.isna(x) for x in (g1, g3)):
        acc = np.log2(max(g1, 0.01)) - np.log2(max(g3, 0.01)) / 3
        parts.append(float(np.clip(acc, 0, 3)))
        weights.append(0.11)

    # YouTube pressure: normalised view velocity, log-scaled.
    vph = r.get("yt_velocity")
    if vph is not None and not pd.isna(vph) and vph > 0:
        parts.append(float(np.clip(np.log10(vph) - 2.0, 0, 3)))
        weights.append(0.14)
    elif r.get("yt_videos") and not pd.isna(r.get("yt_videos")):
        parts.append(float(min(r["yt_videos"], 5) / 5.0))
        weights.append(0.08)

    # Multi-market coverage: the same game trending in 4 languages at once is
    # a stronger signal than one English creator with a big number.
    langs = r.get("yt_langs")
    if langs is not None and not pd.isna(langs) and langs >= 2:
        parts.append(float(min(langs, 5) / 5.0) * 1.5)
        weights.append(0.09)

    # Paid promotion: a studio spending money is a forward-looking bet.
    if r.get("sponsored"):
        parts.append(1.2)
        weights.append(0.05)

    if r.get("chart_count"):
        parts.append(min(r["chart_count"] / 3.0, 2.0))
        weights.append(0.05)
    if r.get("rank_delta_d1") is not None and not pd.isna(r["rank_delta_d1"]):
        parts.append(float(np.clip(r["rank_delta_d1"] / 10.0, -1.5, 2.0)))
        weights.append(0.03)

    lr = r.get("like_ratio")
    if lr is not None and not pd.isna(lr) and (r.get("visits") or 0) > 500:
        parts.append(float(np.clip((lr - 0.6) * 4, -1.5, 1.5)))
        weights.append(0.02)

    if not parts:
        return 0.0
    raw = float(np.average(parts, weights=weights))
    return round(100.0 / (1.0 + np.exp(-raw)), 1)


def _confidence(r: pd.Series) -> float:
    """0-1: how much independent evidence actually backs this score."""
    have = sum(1 for k in ("g_h6", "g_d1", "g_d3")
               if r.get(k) is not None and not pd.isna(r.get(k)))
    base = {0: 0.0, 1: 0.35, 2: 0.7, 3: 1.0}[have]
    span = r.get("history_hours") or 0
    conf = base * min(span / 24.0, 1.0)

    # independent corroboration, each capped so no single source dominates
    if r.get("yt_videos") and not pd.isna(r.get("yt_videos")) and r["yt_videos"] > 0:
        conf = min(conf + 0.12, 1.0)
    if r.get("yt_langs") and not pd.isna(r.get("yt_langs")) and r["yt_langs"] >= 2:
        conf = min(conf + 0.08, 1.0)
    if r.get("sponsored"):
        conf = min(conf + 0.05, 1.0)
    return round(conf, 2)


def flag_suspect(f: pd.DataFrame) -> pd.DataFrame:
    """Spike with no retention support = likely bots or a paid push."""
    if f.empty:
        return f
    f = f.copy()
    f["suspect_bot"] = (
        f["g_d1"].notna() & (f["g_d1"] > 3)
        & (f["fav_per_1k_visits"].fillna(0) < 5)
    )
    return f


if __name__ == "__main__":
    df = build_features()
    if df.empty:
        print("no snapshots yet")
    else:
        df = flag_suspect(df)
        pd.set_option("display.width", 240)
        cols = ["name", "niche", "ccu", "g_d1", "yt_videos", "yt_langs", "yt_non_en",
                "sponsored", "score", "confidence"]
        print(df[cols].head(25).to_string(index=False))
