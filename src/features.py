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
import pickle
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from db import Game, Mention, Session, Snapshot, YtVideo  # noqa: E402
from sqlalchemy import text, func  # noqa: E402

H = {"h6": 6 / 24, "d1": 1, "d3": 3, "d7": 7}

# --- transfer budget -------------------------------------------------------
#
# Neon free tier allows 5 GB of network transfer per project per month. Measured
# 2026-10-29: the run was pulling ~51 MB per tick, 8 ticks a day, i.e. 11.1 GB a
# month - 221% of the allowance, with the project already at 81%. Two causes,
# both fixed here:
#
# 1. NO TIME WINDOW. `_snapshots_df()` read the whole table every time. The
#    table is 4.02 MB today and covers only 9.8 days, so the number grows
#    linearly and forever: at one year of history the same query is ~150 MB, and
#    the monthly transfer would be ~50 GB. Nothing in this module needs more
#    than WINDOW_DAYS back - see the long note on the constant.
#
# 2. NO CACHING ACROSS PROCESSES. `build_features()` is called by six separate
#    scripts (report, opportunity, clone_targets, digest_html, predictions) that
#    run as six separate processes in the same workflow job, each transferring
#    the same rows again. A module-level cache cannot help across processes, so
#    the cache is written to disk - the job's filesystem is shared between
#    steps, which is exactly the lifetime that is needed.
#
# 21 days, not 7: the widest window any consumer reads is 7 days (`H` below),
# but `derive_threshold()` measures growth over a 7-day horizon and needs a
# second 7-day window before it to compute a distribution - at a 7-day window
# it would find zero comparable pairs and always report "not enough history".
# 21 days gives three full measurement windows and costs 92% -> 62% of the
# current table, growing much slower than "everything".
WINDOW_DAYS = int(os.environ.get("TRENDRADAR_WINDOW_DAYS", "21"))

_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "features_cache.pkl")


def _cache_key() -> tuple:
    """Identity of the snapshot set: (max ts, row count).

    Both change on every collection tick (a new ts appears, rows are added), so
    the key invalidates itself. Deliberately not a TTL: a TTL long enough to
    help would also survive a new ingest and serve stale features into a page
    that claims to be current.
    """
    with Session() as s:
        ts = s.query(func.max(Snapshot.ts)).scalar()
        n = s.query(func.count()).select_from(Snapshot).scalar()
    return (str(ts), int(n or 0))

# Language buckets reported separately: these are where Roblox is growing.
MAJOR_LANGS = ("en", "es", "pt", "ru", "id")


def _snapshots_df() -> pd.DataFrame:
    """Snapshots for the last WINDOW_DAYS, newest window only.

    The cutoff is computed in Python rather than SQL so the same code works on
    SQLite and Postgres without a dialect branch - `Snapshot.ts >= cutoff` with
    a real datetime is portable, whereas `now() - interval '21 days'` is not.
    """
    cutoff = None
    with Session() as s:
        newest = s.query(func.max(Snapshot.ts)).scalar()
        if newest is not None:
            cutoff = newest - timedelta(days=WINDOW_DAYS)
        q = s.query(
            Snapshot.universe_id, Snapshot.ts, Snapshot.playing, Snapshot.visits,
            Snapshot.favorites, Snapshot.up_votes, Snapshot.down_votes,
            Snapshot.max_players, Snapshot.servers, Snapshot.avg_fill,
            Snapshot.chart_count, Snapshot.best_chart_rank, Snapshot.is_sponsored,
        )
        if cutoff is not None:
            q = q.filter(Snapshot.ts >= cutoff)
        rows = q.all()
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
        # Mentioned videos only - see the same change in digest_html.
        # yt_videos is 38724 rows here and only 15773 of them are ever joined
        # to a mention; the rest are transferred and dropped. 2.36 MB -> 0.96 MB
        # per call on Neon (measured 2026-10-29).
        v = s.execute(text("""
            select v.video_id, v.views, v.views_per_hour, v.published
            from yt_videos v
            where exists (select 1 from mentions m where m.video_id = v.video_id)
        """)).all()

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

    # Tolerance was 50% of the window, which is far too loose. At 50% a
    # "7-day" comparison accepted a 4-day-old snapshot (43% short) and a
    # "1-day" comparison accepted a 13-hour-old one (46% short). Both were
    # silently reported as d1/d7 growth.
    #
    # The bound is now max(3 h, 12% of window) - 3 h because sampling is
    # every 3 hours so nothing closer is available, 12% because beyond that
    # the window being measured is misdescribed by its own label.
    tol = timedelta(hours=max(3.0, days_ago * 24 * 0.12))
    if delta.min() > tol:
        return None
    return sub.loc[delta.idxmin(), col]


def _span_hours(sub: pd.DataFrame, now) -> float:
    """Hours between the oldest snapshot for this game and now."""
    if sub.empty:
        return 0.0
    oldest = sub["ts"].min()
    return max((now - oldest).total_seconds() / 3600, 0.0)


def _value_at_with_span(sub: pd.DataFrame, col: str, days_ago: float,
                        now: pd.Timestamp):
    """Value nearest `days_ago` back, plus the actual hours between it and now.

    The span is returned because the caller needs it to turn a difference into a
    rate. Using the game's whole history instead of the matched window silently
    rescales every rate by however much history happens to exist.
    """
    if sub.empty:
        return None, None
    target = now - timedelta(days=days_ago)
    delta = (sub["ts"] - target).abs()
    tol = timedelta(hours=max(3.0, days_ago * 24 * 0.12))
    if delta.min() > tol:
        return None, None
    idx = delta.idxmin()
    matched_ts = sub.loc[idx, "ts"]
    hours = (now - matched_ts).total_seconds() / 3600
    return sub.loc[idx, col], hours


def _ratio(new, old, floor=30.0):
    if new is None or old is None or pd.isna(new) or pd.isna(old):
        return None
    return float(new) / max(float(old), floor)


def build_features() -> pd.DataFrame:
    """Cached wrapper around _build_features().

    The cache exists because six separate processes call this in one workflow
    job and none of them can see another's memory. Keyed on (max ts, row count)
    rather than a TTL - see _cache_key() - so a new collection tick always
    invalidates it and a stale frame can never reach the page.

    On any cache error the function simply recomputes: a corrupt or unreadable
    pickle must not be able to break the run, and the whole point of a cache is
    that losing it is cheap.
    """
    try:
        key = _cache_key()
    except Exception:
        return _build_features()

    try:
        if os.path.exists(_CACHE):
            with open(_CACHE, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("key") == key and isinstance(blob.get("df"), pd.DataFrame):
                return blob["df"]
    except Exception:
        pass

    f = _build_features()
    try:
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        with open(_CACHE, "wb") as fh:
            pickle.dump({"key": key, "df": f}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return f


def _build_features() -> pd.DataFrame:
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

        # --- conversion signals: who WANTS to come back, not who is here now
        #
        # favorites and up_votes are cumulative counters, and unlike CCU they
        # move in one direction: monotonically, all day. Measured on 17
        # snapshots over 21 hours, favorites rose for 414 of 453 games and
        # up_votes for 392. CCU during the same window follows the market cycle
        # and says nothing about a single game.
        #
        # So the derivative is the signal, not the level. fav_rate is
        # favourites gained per hour; it answers "are people saving this to
        # come back", which is what an active audience looks like.
        fav_now = cur["favorites"] or 0
        up_now = cur["up_votes"] or 0
        fav_prev, fav_span = _value_at_with_span(sub, "favorites", H["h6"], now)
        up_prev, up_span = _value_at_with_span(sub, "up_votes", H["h6"], now)

        # The rate must divide by the window the gain was measured over, not by
        # the game's whole history. The first version divided by total history
        # span: on 23 hours of data that understated a 6-hour gain by ~3.8x, and
        # games with longer histories were penalised against newer ones inside
        # the same cohort percentile. Measured: most games had 20-24h of span
        # against a 6h gain window, so the bias was uniform but real.
        if fav_prev is not None and not pd.isna(fav_prev) and fav_span and fav_span >= 1:
            gained = fav_now - float(fav_prev)
            rec["fav_gain_6h"] = int(gained) if gained > 0 else 0
            rec["fav_rate"] = round(max(gained, 0) / fav_span, 1)
        else:
            rec["fav_gain_6h"] = None
            rec["fav_rate"] = None

        if up_prev is not None and not pd.isna(up_prev) and up_span and up_span >= 1:
            gained = up_now - float(up_prev)
            rec["vote_gain_6h"] = int(gained) if gained > 0 else 0
            rec["vote_rate"] = round(max(gained, 0) / up_span, 1)
        else:
            rec["vote_gain_6h"] = None
            rec["vote_rate"] = None

        # Conversion efficiency: what share of people who ever played came back
        # to save the game. Age-normalised by cohort below, like lifetime_ratio.
        rec["fav_per_visit"] = (
            round(fav_now / rec["visits"], 4) if rec["visits"] else None)

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

    # Conversion percentiles. fav_per_visit is raw favourites divided by raw
    # visits, so a 2014 game with 4 billion visits scores near zero forever and
    # a brand-new game scores high on its first 200 players. Ranking inside the
    # age x size cohort removes both distortions, which is what makes this
    # comparable at all - the same reasoning as lifetime_ratio above.
    f["r_fav_pct"] = grp["fav_per_visit"].transform(pct)
    f["r_favrate_pct"] = grp["fav_rate"].transform(pct)

    # Weighted rank over the components that actually exist. Missing inputs
    # are dropped and the weights renormalised - filling them with 0.5 made
    # every game in a cohort tie at the same value.
    # Conversion gets real weight (0.25 of 1.25) because it is the only signal
    # here that is not derived from CCU. Momentum can be bought with ads;
    # somebody pressing "favourite" cannot. Measured fill: fav_rate 95%,
    # fav_per_visit 100%, against yt_videos at 5%.
    weights = {"r_g1_pct": 0.38, "r_g6_pct": 0.20, "r_life_pct": 0.22,
               "r_favrate_pct": 0.12, "r_fav_pct": 0.08}
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
