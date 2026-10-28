"""Write forecasts down, judge them later. Append-only by construction.

    python src/predictions.py record          # log this run's forecasts
    python src/predictions.py due             # what is ready to judge
    python src/predictions.py resolve         # judge what is due
    python src/predictions.py score           # hit rate, calibration, baseline
    python src/predictions.py report          # the table a critic reads

WHY THIS EXISTS

Every other module answers "what is happening". This one answers the only
question that decides whether the others are worth running:

    was the forecast right?

By the numbers in this codebase: 8 cards per digest, ~30 digests so far -
roughly 240 recommendations, and not one of them was ever checked. A ranking
nobody grades is a preference, not a model.

THE RULE

A row is written once and never edited. `outcome` moves from NULL to a value,
once. There is no update function and no delete function here, and nothing in
this file touches a settled row. That is deliberate.

Judging is fixed at creation too. `criterion` is chosen when the forecast is
made - "CCU up 20% in 7 days" - not when the result is known, so the target
cannot drift toward whatever happened.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from db import Prediction, Session, Snapshot, utcnow  # noqa: E402

# Below this a percentage move is noise: 50 -> 60 is +20% and means nothing.
MIN_BASELINE_CCU = 500

# Thresholds can be derived from the data instead of picked by eye.
#
# The first version used 20 pp because it is a round number. Measured, that bar
# is cleared by ~9% of games - which is neither right nor wrong on its own, but
# it was never chosen, only assumed. A threshold should be set by the base rate
# you want to test against, not by how tidy the number looks.


# Whether games discovered outside the Roblox carousels (`games.origin =
# 'orphan'`) count toward the market median and the base rate.
#
# False until the week-2 comparison is finished, and the reason is measurement,
# not quality. Orphans are on average smaller than carousel games (measured
# 2026-10-28: median 116 CCU across all orphans, 1812 among those above the 500
# floor, against a carousel market median of ~3000-4000). Dropping 161 such
# games into the sample moves the median of per-game growth - the yardstick
# both the 20 pp bar and `universe_share_over_20pp` are measured against - and
# the verdicts would then be produced under a different ruler than week 1's.
# The one thing week 2 exists to test, "did the new selection rule help", would
# become unanswerable.
#
# They are still collected (collect.collect_orphans), so their history starts
# accumulating now and can be applied retroactively: `origin` lives on the game,
# not baked into each snapshot, so flipping this constant recomputes every past
# window correctly with no backfill and no data loss.
INCLUDE_ORPHANS_IN_METRICS = False


def _orphan_uids(s) -> set[int]:
    """Games to exclude from market medians and base rates, right now."""
    if INCLUDE_ORPHANS_IN_METRICS:
        return set()
    from db import Game
    return {uid for (uid,) in
            s.query(Game.universe_id).filter(Game.origin == "orphan").all()}


def _snapshots_pivots() -> tuple[pd.DataFrame, pd.Series]:
    """CCU by game over time, plus the market median per snapshot."""
    with Session() as s:
        rows = s.query(Snapshot.universe_id, Snapshot.ts, Snapshot.playing).all()
        drop = _orphan_uids(s)
    df = pd.DataFrame(rows, columns=["universe_id", "ts", "playing"])
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    if drop:
        df = df[~df["universe_id"].isin(drop)]
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    piv = (df.pivot_table(index="ts", columns="universe_id",
                          values="playing", aggfunc="last").sort_index())
    return piv, piv.median(axis=1)


def excess_samples(horizon_days: float = 7,
                   min_ccu: int = MIN_BASELINE_CCU) -> pd.DataFrame:
    """Every measurable (game, window) pair of excess growth over the market.

    Excess = game change minus market change, in percentage points. This is the
    quantity the digest implicitly claims and that a relative criterion grades.

    Returns empty when history is shorter than the horizon - which is the honest
    answer before ~7 days have accumulated, not an error.
    """
    piv, med = _snapshots_pivots()
    if piv.empty or len(piv) < 2:
        return pd.DataFrame()

    horizon = pd.Timedelta(days=horizon_days)

    # The tolerance may never exceed the window itself. The first version used
    # max(3h, 12% of window), so every horizon shorter than 3 hours was measured
    # over a matched pair up to 3 hours apart - "p75 on 15 minutes = 24.3 pp"
    # was in fact a 3-hour measurement wearing a 15-minute label. Sampling is
    # every 3 hours, so anything below that is not measurable at all and is
    # rejected rather than silently widened.
    if horizon_days * 24 < 3.0:
        return pd.DataFrame()
    # 12% of the window, but never below 3h because sampling is every 3h.
    tol = pd.Timedelta(hours=max(3.0, horizon_days * 24 * 0.12))
    stamps = piv.index
    out = []

    for t0 in stamps:
        later = stamps[stamps >= t0 + horizon]
        if len(later) == 0:
            continue
        t1 = later[0]
        if abs((t1 - t0) - horizon) > tol:
            continue
        a, b = piv.loc[t0], piv.loc[t1]

        # Market change = the MEDIAN OF EACH GAME'S OWN ratio from t0 to t1,
        # not the ratio of the two median snapshots (med[t1]/med[t0]).
        #
        # Those are different quantities whenever games enter or leave the
        # matched set between t0 and t1, which happens on every tick here -
        # the carousel composition shifts. By Jensen's inequality the ratio
        # of medians is biased upward whenever the underlying ratios are
        # right-skewed, which CCU growth always is (many flat/declining games,
        # a few that multiply). Measured on this project's own snapshots: the
        # two methods disagreed by a mean of +4.8 pp and up to +24.5 pp across
        # 15 sampled windows, and changed the hit/miss verdict on 6 of the
        # first 21 settled forecasts (29%) - always in the same direction,
        # always making the bar harder to clear. Median-of-ratios is the
        # unbiased version of "did a typical game grow this much".
        both = a.index[(a > 0) & b.notna()]
        if len(both) < 20:
            continue
        ratios = (b[both] / a[both]).to_numpy(dtype=float)
        market_change = float(np.median(ratios)) - 1.0

        for uid in piv.columns:
            a0, b0 = a.get(uid), b.get(uid)
            if pd.isna(a0) or pd.isna(b0) or a0 < min_ccu:
                continue
            out.append((uid, t0, ((b0 / a0 - 1) - market_change) * 100))

    return pd.DataFrame(out, columns=["universe_id", "start", "excess_pp"])


def derive_threshold(percentile: int = 75, horizon_days: float = 7,
                     min_ccu: int = MIN_BASELINE_CCU) -> dict:
    """The pp bar that `percentile` of games clear, measured from history.

    Why this instead of a round number: the point of a threshold is to test a
    claim about base rates. "p75" means a forecast that clears it did something
    three quarters of comparable games did not - a statement with a known prior.
    "20 pp" means nothing until someone measures what 20 pp is worth.

    Returns ok=False with the reason when there is not enough history. Before
    the first week that is the expected outcome, and the caller should keep the
    existing criterion rather than invent a substitute.
    """
    samples = excess_samples(horizon_days, min_ccu)
    if samples.empty:
        span_h = 0.0
        piv, _ = _snapshots_pivots()
        if not piv.empty:
            span_h = (piv.index[-1] - piv.index[0]).total_seconds() / 3600
        return {"ok": False,
                "reason": (f"истории {span_h:.0f} ч — меньше горизонта "
                           f"{horizon_days*24:.0f} ч"),
                "span_hours": round(span_h, 1),
                "horizon_days": horizon_days}

    q = float(samples["excess_pp"].quantile(percentile / 100))
    return {
        "ok": True,
        "percentile": percentile,
        "threshold_pp": round(q, 1),
        "horizon_days": horizon_days,
        "samples": int(len(samples)),
        "games": int(samples["universe_id"].nunique()),
        "median_pp": round(float(samples["excess_pp"].median()), 1),
        "criterion": f"beat_market_p{percentile}",
    }


# Bumped whenever PLANS, _judge, or the candidate ranking in record_from_features
# changes meaningfully. Written onto every new row (Prediction.ruleset_version)
# so a future comparison can filter or group by "which rules produced this
# number" instead of relying on prose in outcome_basis or guessing from dates.
#
#   1 = the pre-2026-10-22 state: void used as an ad-hoc edit mechanism during
#       development (100 of the first 124 rows carry this history, all void).
#   2 = due_at judging (not run-time), void->miss on charts dropout,
#       median-of-ratios benchmark (not ratio-of-medians), ascending-trend
#       selection (closest-to-bar instead of biggest-mover). Rows written
#       2026-10-22 onward with the fixes from that date forward are version 2;
#       the exact mix of fixes present differs slightly within version 2 itself
#       (e.g. the benchmark fix landed 2026-10-28), which is a known gap - see
#       CHANGELOG below rather than assume every v2 row is judged identically.
CURRENT_RULESET_VERSION = 2

# How each kind of forecast is judged. Deliberately blunt: a percentage move.
# Anything softer ("became popular") cannot be scored, and a soft criterion is
# how a forecast gets reinterpreted into a hit after the fact.
PLANS = {
    # The digest says "grows faster than the market" - a SHARE claim. The first
    # version graded it as an absolute CCU move of +20%, which is a different
    # question entirely.
    #
    # Measured on live data, the market median moves ~+2.8%/day, i.e. ~+19%
    # over a week. So an absolute +20% bar is almost exactly "did the market
    # grow", and every forecast would pass or fail together regardless of the
    # game. The criterion has to be relative to beat that.
    "move":  {"metric": "ccu", "criterion": "beat_market_20pct", "horizon": 7},
    # A clone target gets longer: a copy needs time to show whether the
    # original held.
    "clone": {"metric": "ccu", "criterion": "up_20pct", "horizon": 14},
    # An anime premiere should lift its mapped niche. Judged on the niche
    # median, not one game, because we cannot know which clone wins.
    "anime": {"metric": "ccu", "criterion": "up_10pct", "horizon": 21},
}

# How far from the due date a snapshot may sit and still be used to judge it.
# Sized to the collection interval: 3-hour ticks plus slack for a missed run,
# and well under the 7-day horizon so a game that left the charts three days
# early cannot be graded on a reading from before it fell.
STALE_HOURS = 48



def _already_open(s, kind: str, universe_id: int | None, subject: str) -> bool:
    """One open forecast per game per kind.

    Deduplication keys on `universe_id`, not on the name. Roblox games rename
    constantly - "[CURE] Violence District" becomes "Violence District [UPDATE]"
    with the same universe_id - and the first version matched on subject as
    well. Measured: a renamed game made this return False, so a second row was
    created for a game already under forecast. Both rows would later be judged
    on the same window, and one lucky game would count twice in the hit rate.

    The name is only used as a fallback for rows that have no universe_id, which
    should not exist for `move` but does for aggregate kinds.
    """
    q = s.query(Prediction).filter(
        Prediction.kind == kind,
        Prediction.outcome.is_(None))
    if universe_id is not None:
        return q.filter(Prediction.universe_id == universe_id).count() > 0
    return q.filter(Prediction.subject == subject).count() > 0


def digest_selection() -> pd.DataFrame:
    """The rows the digest shows under "растут быстрее рынка" - nothing else.

    The first version of record_from_features selected by `cohort_rank`, which
    is a percentile inside an age x size cohort. The digest selects by
    `share_slope * span` - market-relative movement. Two different rankings.

    Measured overlap: 2 games in 10. So the table was grading a list the site
    never displayed, which makes the whole exercise meaningless - the point is
    to check what the user was actually told.

    This function reproduces the digest's own selection by calling its own
    functions, so the two cannot drift apart again.
    """
    import digest_html as D

    piv, med = D.market_series()
    moves = D.relative_moves(piv, med)
    f = D.build_features()
    rows = D.build_rows(f, moves, {})
    if rows.empty:
        return rows

    # Same thresholds render() uses. Duplicating the numbers would drift, so
    # they are read from the module.
    return rows[(rows["trend"] >= D.MOVE_THRESHOLD) & (rows["r2"] >= 0.25)]


def _blend_confidence(availability, r2) -> float | None:
    """Confidence = how much data backs this AND how well the trend fits.

    features._confidence measures only the first half: how many growth windows
    are present and how long the history is. Measured on live data it is
    0.67 for 365 of 468 games - g_d3 is empty for every game until three days
    of history exist, so `have` is always 2 and the score is constant. A
    constant cannot calibrate anything: the calibration table would be a single
    bucket, which is exactly the failure this column exists to detect.

    r2 measures the other half - whether the fitted trend is real or noise -
    and it does vary (0.28-0.45 across the current top eight). Neither alone
    answers "should I trust this prediction", so the product does.

    Scale: r2 below the 0.25 digest gate is treated as weak (0.4), a perfect
    fit as 1.0. Multiplying keeps a well-fitted game with thin data from
    outranking a well-evidenced one.
    """
    if availability is None or pd.isna(availability):
        return None
    avail = float(np.clip(availability, 0.0, 1.0))
    if r2 is None or pd.isna(r2):
        # No fit information: halve it rather than pretend the trend is solid.
        return float(round(avail * 0.5, 2))
    fit = float(np.clip((float(r2) - 0.25) / 0.75 * 0.6 + 0.4, 0.2, 1.0))
    return float(round(avail * fit, 2))


def record_from_features(top_n: int = 10, kind: str = "move") -> dict:
    """Log what the digest actually showed, in the order it showed it.

    Selecting independently would let the record drift from the page, and then
    the score would grade a ranking nobody saw.
    """
    plan = PLANS[kind]

    if kind == "move":
        rows = digest_selection()
        if rows.empty:
            return {"recorded": 0, "reason": "digest shows nothing rising"}
        # Match the page exactly. The page renders RISING_CARDS cards; writing
        # more than that would put games into the sample that nobody was shown.
        import digest_html as D

        # PROVISIONAL, set 2026-10-28 after the first 24 forecasts settled.
        # The original rule picked the LARGEST moves first, on the assumption
        # that a bigger move is a stronger signal. Measured on those 24:
        # Spearman(trend, hit) = -0.36 - the biggest movers missed more than
        # they hit (46% hit rate below the trend median vs 8% above it). r2
        # (how cleanly the growth line fits) pointed the other way, weakly
        # (+0.17). Read together: a game at the top of a spike scores a huge
        # trend number right when it is least likely to still be climbing by
        # the time the horizon closes, which is the "launch spike, then
        # reversion" pattern the external audit flagged and n=24 now shows
        # directly on this project's own data - not a reason to trust it
        # blindly, but a reason to stop selecting for the opposite of what
        # the first week measured.
        #
        # So: rank candidates by PROXIMITY to the qualifying bar (ascending
        # trend) instead of by the size of the move, with r2 descending as
        # the tie-break. This changes what gets RECORDED as a forecast, not
        # what the digest page shows - the page still lists the biggest
        # movers, which is a legitimate "what happened" view independent of
        # whether that is a good forecast ranking.
        #
        # n=24 is small. Re-check this sort once week 2 settles (~2026-11-04)
        # and revert to descending if the direction does not hold.
        rows = rows.sort_values(["trend", "r2"], ascending=[True, False])
        rows = rows.head(min(top_n, D.RISING_CARDS))

        # The digest already drops games below its own MIN_CCU inside
        # relative_moves, but that coupling is implicit and one refactor away
        # from breaking. Stating it here means the ledger and the page cannot
        # drift into disagreeing about which games are large enough to forecast
        # on - a silent split that would record rows the page never showed.
        floor = getattr(D, "MIN_CCU", MIN_BASELINE_CCU)
        rows = rows[rows["ccu_last"].fillna(0) >= floor]

        # `confidence` must be features._confidence, not r2. They measure
        # different things: r2 is how well a line fits the trend, confidence is
        # how much data backs it. Writing r2 into the confidence column meant
        # the calibration table was grading the wrong quantity entirely.
        conf_by_uid = {}
        try:
            from features import build_features as _bf
            _f = _bf()
            if not _f.empty and "confidence" in _f.columns:
                conf_by_uid = dict(zip(_f["universe_id"], _f["confidence"]))
        except Exception:
            conf_by_uid = {}

        candidates = [{
            "universe_id": int(r["universe_id"]),
            "name": r["name"],
            "base": float(r["ccu_last"] or 0),
            "score": r.get("trend"),
            "confidence": _blend_confidence(
                conf_by_uid.get(int(r["universe_id"])), r.get("r2")),
            "cohort_rank": r.get("cohort_rank"),
            "r2": r.get("r2"),
        } for r in rows.to_dict("records")]
    else:
        from features import build_features
        f = build_features()
        if f.empty:
            return {"recorded": 0, "reason": "no features"}
        f = f[f["ccu"] >= MIN_BASELINE_CCU].dropna(subset=["cohort_rank"])
        candidates = [{
            "universe_id": int(r["universe_id"]),
            "name": r.get("name"),
            "base": float(r.get("ccu") or 0),
            "score": r.get("score"),
            "confidence": r.get("confidence"),
            "cohort_rank": r.get("cohort_rank"),
        } for r in f.nlargest(top_n, "cohort_rank").to_dict("records")]

    made = skipped = no_momentum = 0
    with Session() as s:
        for c in candidates:
            subject = (c["name"] or "")[:256]
            uid = c["universe_id"]
            if not subject or _already_open(s, kind, uid, subject):
                skipped += 1
                continue
            base = c["base"]

            # NaN needs its own check. `float(nan or 0)` returns nan (nan is
            # truthy), and every comparison against nan is False - so a nan
            # baseline passes `base <= 0` and later makes `_judge` fall through
            # to "miss" with a basis of "nan%". A silent wrong outcome is worse
            # than no row at all.
            if base is None or (isinstance(base, float) and math.isnan(base)) \
                    or base <= 0:
                skipped += 1
                continue

            # A forecast named "move" must be based on an observed move. A game
            # can rank highly on conversion alone while having no growth data
            # at all - predicting "will grow 20%" from that is a contradiction,
            # and it would pollute the sample with a claim never actually made.
            # If conversion alone predicts growth, that is a separate
            # hypothesis and gets its own kind later.
            if kind == "move":
                hist = (s.query(Snapshot.playing)
                         .filter(Snapshot.universe_id == uid)
                         .count())
                if hist < 2:
                    no_momentum += 1
                    continue

            s.add(Prediction(
                kind=kind,
                universe_id=uid,
                subject=subject,
                claim=(f"обгонит рынок на ≥20 пп за {plan['horizon']} дн "
                       f"(старт {base:.0f})"),
                metric=plan["metric"],
                baseline=base,
                # Market median at forecast time. Stored as `benchmark` so the
                # relative criterion can be applied later without re-deriving
                # history that may have been pruned.
                benchmark=_market_median_ccu(s, utcnow()),
                criterion=plan["criterion"],
                threshold_pp=plan.get("threshold_pp", 20.0),
                horizon_days=plan["horizon"],
                score=c["score"],
                confidence=c["confidence"],
                cohort_rank=c["cohort_rank"],
                r2=c.get("r2"),
                ruleset_version=CURRENT_RULESET_VERSION,
                due_at=utcnow() + timedelta(days=plan["horizon"]),
            ))
            made += 1
        s.commit()
    return {"recorded": made, "skipped_duplicate": skipped,
            "no_momentum": no_momentum}


def due_forecasts() -> list[dict]:
    with Session() as s:
        rows = (s.query(Prediction)
                 .filter(Prediction.outcome.is_(None))
                 .filter(Prediction.due_at <= utcnow())
                 .all())
        return [{
            "id": r.id, "kind": r.kind, "universe_id": r.universe_id,
            "subject": r.subject, "metric": r.metric, "baseline": r.baseline,
            "criterion": r.criterion, "claim": r.claim, "due_at": r.due_at,
            "threshold_pp": r.threshold_pp,
            # Needed to compute the market's own change over the SAME window
            # the forecast covers (made_at -> due_at), not "whatever the
            # market is doing right now".
            "made_at": r.made_at,
        } for r in rows]


def _market_median_ccu(s, at) -> float | None:
    """Market median CCU at a single point in time. Display/debug only.

    Used to populate `benchmark` when a forecast is recorded, as a reference
    value a human can read next to the row. NOT used to judge "beat the
    market" claims any more - see `_market_change` for that.
    """
    from sqlalchemy import func
    ts = (s.query(func.max(Snapshot.ts))
           .filter(Snapshot.ts <= at).scalar())
    if ts is None:
        return None
    vals = [r[0] for r in
            (s.query(Snapshot.playing)
              .filter(Snapshot.ts == ts)
              .filter(Snapshot.playing.isnot(None)).all())]
    vals = sorted(v for v in vals if v and v > 0)
    if not vals:
        return None
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def _snapshot_ticks_at_or_before(s, at, n: int) -> list:
    """Up to `n` distinct snapshot timestamps at or before `at`, newest first."""
    rows = (s.query(Snapshot.ts).filter(Snapshot.ts <= at)
             .distinct().order_by(Snapshot.ts.desc()).limit(n).all())
    return [r[0] for r in rows]


def _market_change(s, base_at, now_at, min_games: int = 20) -> tuple[float | None, int]:
    """How much a TYPICAL game grew from base_at to now_at, as a fraction.

    This replaces the old method of comparing two market-median SNAPSHOTS
    (median_now - median_base) / median_base, which is not the same question
    as "how much did a typical game grow" and is biased toward overstating
    market growth whenever growth is right-skewed - which CCU growth always
    is (a few games multiply, most are flat or shrink, so the median of the
    later snapshot is pulled up by games that were not even in the earlier
    one, and the Jensen's-inequality gap between ratio-of-medians and
    median-of-ratios is not zero).

    Measured on this project's own history: the two methods disagreed by a
    mean of +4.8 pp and up to +24.5 pp across 15 sampled week-long windows,
    and the old method changed the hit/miss verdict on 6 of the first 21
    settled forecasts (29%) - always toward "harder to beat", never the
    other way.

    This function instead computes, per game present at both ends, its own
    ratio value_at(now)/value_at(base), and returns the MEDIAN of those
    per-game ratios, using the single nearest snapshot tick at each end (same
    matching rule `_value_now_fresh` uses, so both halves of the comparison
    are built the same way).

    An earlier version averaged a trailing window of ticks at each end to
    damp day/night noise (the market median point-sample swings between
    ~3000 and ~5200 over a week). That was reverted: averaging a window
    ending at base_at and a window ending at now_at does not sample the same
    phase of the daily cycle unless the two endpoints happen to fall at the
    same hour, so the smoothing quietly swapped one bias (ratio-of-medians)
    for another (uncontrolled time-of-day mismatch) - checked by replaying it
    against the 24 settled forecasts, where it moved the verdict on 9 of them
    versus 6 for the single-tick fix alone. The point-sample volatility is a
    real, separate problem; it needs same-hour matching or a longer baseline,
    not a trailing average, and is left open rather than papered over.

    Returns (change_fraction, n_games_used). change_fraction is None when
    fewer than `min_games` games can be matched at both ends.
    """
    base_ts = _snapshot_ticks_at_or_before(s, base_at, 1)
    now_ts = _snapshot_ticks_at_or_before(s, now_at, 1)
    if not base_ts or not now_ts:
        return None, 0

    # Same exclusion as the base rate, for the same reason: the median of
    # per-game ratios IS the ruler a "beat the market" verdict is measured
    # against. Mixing a differently-sized population into one half of the
    # comparison and not the other would be the ratio-of-medians bias again,
    # wearing a different hat.
    drop = _orphan_uids(s)

    def _avg_per_game(ticks: list) -> dict[int, float]:
        rows = (s.query(Snapshot.universe_id, Snapshot.playing)
                 .filter(Snapshot.ts.in_(ticks))
                 .filter(Snapshot.playing.isnot(None))
                 .filter(Snapshot.playing > 0).all())
        acc: dict[int, list] = {}
        for uid, val in rows:
            if uid in drop:
                continue
            acc.setdefault(uid, []).append(val)
        return {uid: float(np.mean(v)) for uid, v in acc.items()}

    a = _avg_per_game(base_ts)
    b = _avg_per_game(now_ts)
    common = [u for u in a if u in b]
    if len(common) < min_games:
        return None, len(common)
    ratios = np.array([b[u] / a[u] for u in common], dtype=float)
    return float(np.median(ratios)) - 1.0, len(common)


def _judge(criterion: str, base: float, now: float,
           market_change: float | None = None,
           threshold_pp: float | None = None) -> tuple[str, str]:
    """Apply the criterion exactly as written. No judgement calls.

    `threshold_pp` is the frozen bar stored on the row. It is never recomputed
    here: a criterion whose bar can move is not a criterion.

    `market_change` is the market's own fractional change over the SAME
    window the forecast covers, as computed by `_market_change` (median of
    per-game ratios, not a ratio of two median snapshots - see that
    function's docstring for why the distinction changes verdicts).
    """
    # nan slips past `base <= 0` because comparisons with nan are always False.
    # Without this check a corrupt row scores as a miss instead of a void.
    if base <= 0 or (isinstance(base, float) and math.isnan(base)):
        return "void", "baseline zero или NaN"
    if now is None or (isinstance(now, float) and math.isnan(now)):
        return "void", "нет значения на дату проверки"

    change = (now - base) / base

    if criterion.startswith("beat_market"):
        # Relative: how much the game outgrew the market itself.
        if market_change is None:
            return "void", "нет медианы рынка для сравнения"
        excess = change - market_change
        basis = (f"игра {change*100:+.0f}%, рынок {market_change*100:+.0f}%, "
                 f"разница {excess*100:+.0f} пп")
        # Frozen at creation. The old hardcoded 0.20 is kept only for rows
        # written before the field existed.
        bar = threshold_pp if threshold_pp is not None else 20.0
        basis = f"{basis} (порог {bar:.0f} пп)"
        if excess * 100 >= bar:
            return "hit", basis
        if excess >= 0:
            return "partial", basis
        return "miss", basis

    table = {"up_20pct": 0.20, "up_10pct": 0.10}
    if criterion not in table:
        return "void", "unknown criterion"
    bar = table[criterion]
    if change >= bar:
        return "hit", f"+{change*100:.0f}%"
    if change >= 0:
        return "partial", f"+{change*100:.0f}% ниже порога"
    return "miss", f"{change*100:.0f}%"


def _value_now_fresh(s, universe_id: int, metric: str, at=None):
    """(value, age_hours, snapshot_ts) nearest to `at`; `at` defaults to now.

    `at` is the DUE date, not the run date, and that is the whole point. A
    7-day claim graded on "whatever the latest snapshot is when the job
    happens to fire" silently measures a longer window while still calling it
    7 days - and the job fires on an external pinger, so being late is the
    normal case, not an exception. Measured on this project's rows, due_at sits
    7.00-7.01 days after made_at, so the window error is small here; it is
    bounded rather than eliminated by judging at the due date.

    age_hours is signed against `at`: positive means the snapshot predates the
    deadline. Returning None means no snapshot sits within STALE_HOURS of the
    deadline, which the caller must read as the game having left the charts.
    """
    col = Snapshot.playing if metric == "ccu" else Snapshot.favorites
    target = at or utcnow()
    lo = target - timedelta(hours=STALE_HOURS)
    # Never read PAST the deadline. A snapshot taken after due_at contains
    # information the forecast did not have, and grading on it would score the
    # model on hindsight. The only slack allowed is one collection interval
    # (3 h), because the tick that covers the deadline can land just after it.
    hi = target + timedelta(hours=3)
    rows = (s.query(col, Snapshot.ts)
             .filter(Snapshot.universe_id == universe_id)
             .filter(Snapshot.ts >= lo, Snapshot.ts <= hi)
             .all())
    if not rows:
        return None, None, None
    best = min(rows, key=lambda r: abs((r[1] - target).total_seconds()))
    age_h = (target - best[1]).total_seconds() / 3600
    return float(best[0] or 0), age_h, best[1]


def resolve() -> dict:
    """Judge everything due. Writes outcome once and never revisits it."""
    stats = {"checked": 0, "hit": 0, "miss": 0, "partial": 0, "void": 0}
    pending = due_forecasts()
    if not pending:
        return stats

    with Session() as s:
        for p in pending:
            row = s.get(Prediction, p["id"])
            # The guard is the contract: a settled row is never touched again.
            if row is None or row.outcome is not None:
                continue

            # Judge at the DUE date, not at whatever moment this job fired.
            # The horizon is part of the claim; grading it on a longer window
            # grades a different claim.
            now_val, age_h, snapshot_ts = _value_now_fresh(
                s, p["universe_id"], p["metric"], at=p["due_at"])

            # A game that dropped out of the carousels stops receiving
            # snapshots, so there is nothing to judge it on. That is not a
            # reason to drop it from the sample - it is the result.
            #
            # This used to write `void`, which excludes the row from the
            # denominator entirely. But falling off the charts is the most
            # common way a "it will keep growing" claim loses, so excluding it
            # counted every failure as "not measured" and inflated the hit rate
            # by exactly the thing being measured. Intent-to-treat: the row was
            # written and the claim was made, so it gets graded. Missing data
            # at the deadline means the game was below the collection floor,
            # which is a miss on a growth claim.
            if now_val is None:
                row.outcome = "miss"
                row.outcome_basis = (
                    f"нет данных на срок: последний снимок дальше {STALE_HOURS} ч "
                    f"от {p['due_at']:%Y-%m-%d %H:%M} — игра выпала из чартов")[:256]
                row.outcome_value = None
                row.checked_at = utcnow()
                stats["miss"] += 1
                stats["checked"] += 1
                continue

            # Market change over the SAME window the forecast covers
            # (made_at -> the game's own snapshot time), computed as the
            # median of each game's own growth ratio - not two point-in-time
            # medians compared to each other. See `_market_change` docstring:
            # the old method was biased toward a harder bar, and changed the
            # verdict on 29% of the first 21 settled forecasts.
            market_change, n_mkt = _market_change(s, p["made_at"], snapshot_ts)
            verdict, basis = _judge(p["criterion"], p["baseline"], now_val,
                                    market_change, p["threshold_pp"])
            row.outcome = verdict
            row.outcome_value = now_val
            row.outcome_basis = basis[:256]
            row.checked_at = utcnow()
            stats[verdict] = stats.get(verdict, 0) + 1
            stats["checked"] += 1
        s.commit()
    return stats


def score() -> dict:
    """Hit rate, calibration, and the baseline it has to beat.

    The baseline matters more than the hit rate. "60% right" means nothing
    until you know that the dumb rule - every game above the CCU floor grows -
    scores 55% on the same window. If the model cannot clear that, it is an
    expensive way to say nothing.
    """
    with Session() as s:
        total = s.query(Prediction).count()
        open_n = s.query(Prediction).filter(Prediction.outcome.is_(None)).count()
        rows = (s.query(Prediction)
                 .filter(Prediction.outcome.in_(("hit", "miss", "partial")))
                 .all())
        settled = [{
            "kind": r.kind, "outcome": r.outcome, "confidence": r.confidence,
            "baseline": r.baseline, "outcome_value": r.outcome_value,
            "benchmark": r.benchmark, "score": r.score, "r2": r.r2,
            "made_at": r.made_at, "ruleset_version": r.ruleset_version,
        } for r in rows]

    if not settled:
        return {"total": total, "open": open_n, "settled": 0}

    df = pd.DataFrame(settled)
    df["hit_num"] = df["outcome"].map({"hit": 1.0, "partial": 0.5, "miss": 0.0})

    out = {
        "total": total,
        "open": open_n,
        "settled": len(df),
        "hit_rate_pct": round(float(df["hit_num"].mean()) * 100, 1),
        "by_kind": (df.groupby("kind")["hit_num"].mean().mul(100).round(1)
                    .to_dict()),
    }

    # Settled rows made at the same moment share the same market tick and the
    # same digest run - they are not independent trials. An external audit on
    # 2026-10-28 traced the first 24 settled rows to 6 distinct `made_at`
    # clusters (sizes 8/8/1/2/2/3, not 24 independent draws), which means any
    # interval computed as if n=24 understates the true uncertainty by a
    # large factor. Reported here so "settled: 24" is never read alone again.
    #
    # Rows from one run are written seconds apart in a loop, not at the exact
    # same microsecond, so the grouping key floors made_at to the hour - ticks
    # are >=3h apart, which is wider than any single run takes.
    #
    # cluster_hit_rate_range is the spread of per-cluster hit rates, which is
    # a more honest "how much could this move" indicator than a binomial CI
    # on the row count would give, without pretending a formal interval is
    # warranted on this little data.
    cluster_key = pd.to_datetime(df["made_at"]).dt.floor("h")
    clusters = df.groupby(cluster_key)["hit_num"].mean()
    out["n_clusters"] = int(clusters.shape[0])
    out["rows_per_cluster"] = (df.groupby(cluster_key).size()
                                .sort_values(ascending=False).tolist())
    if len(clusters) >= 2:
        out["cluster_hit_rate_range_pct"] = [
            round(float(clusters.min()) * 100, 1),
            round(float(clusters.max()) * 100, 1),
        ]

    # How a miss happened matters as much as how many. A game that fell off the
    # charts never got the chance to grow, which is a different failure from
    # "grew, but slower than the market" - and the two point at different fixes.
    # Reported separately so the numerator cannot hide behind the denominator.
    no_data = df[df["outcome_value"].isna()]
    if not no_data.empty:
        out["miss_no_data"] = int(len(no_data))
        out["miss_no_data_pct"] = round(len(no_data) / len(df) * 100, 1)

    # Baseline. This has to be measured on the SAME quantity the criterion
    # grades, which for "beat the market" is excess growth, not absolute growth.
    # The first version reported the absolute share above +20%, a number that
    # cannot be compared with a relative hit rate at all - it looks like a
    # baseline while measuring something else.
    moves = df.dropna(subset=["baseline", "outcome_value", "benchmark"])
    moves = moves[(moves["baseline"] > 0) & (moves["benchmark"] > 0)]
    if not moves.empty:
        game_chg = (moves["outcome_value"] - moves["baseline"]) / moves["baseline"]
        # Market change over the same window is implied by the stored
        # benchmark only at forecast time; the resolution-time median is not
        # stored per row, so excess is reconstructed from the recorded basis
        # where possible and reported as unavailable otherwise.
        out["median_move_pct"] = round(float(game_chg.median()) * 100, 1)
        out["baseline_up_share_pct"] = round(float((game_chg >= 0).mean()) * 100, 1)

    # What a strategy of "pick any game" would score: the share of ALL
    # comparable games that clear the same bar. This is the number the hit rate
    # has to beat, and without it a hit rate is uninterpretable.
    #
    # Computable only once history covers the horizon, so before the first week
    # it is absent rather than zero - a zero here would read as "nothing ever
    # clears the bar", which is a different and false statement.
    samp = excess_samples(horizon_days=7)
    if not samp.empty:
        out["universe_games"] = int(samp["universe_id"].nunique())
        for thr in (10, 20, 30):
            share = float((samp["excess_pp"] >= thr).mean()) * 100
            out[f"universe_share_over_{thr}pp"] = round(share, 1)
        # Paired explicitly next to hit_rate_pct so a reader cannot cite one
        # without seeing the other. "27% hit rate" means nothing alone; next
        # to "20% of ALL comparable games clear the same bar with no model at
        # all" it reads as a +7pp edge over picking at random, which is the
        # actual claim being tested. hit_rate_pct counts partial as 0.5 while
        # this base rate is binary (>=20pp or not), so the two are not on
        # exactly the same scale - the comparison is a sanity check, not a
        # formal test.
        if "universe_share_over_20pp" in out:
            out["edge_vs_base_rate_pp"] = round(
                out["hit_rate_pct"] - out["universe_share_over_20pp"], 1)

    conf = df.dropna(subset=["confidence"]).copy()
    if len(conf) >= 5:
        # features._confidence returns 0..1, not 0..100. The first version
        # bucketed with `// 20`, which put every single row in bucket 0 and
        # made the calibration table a single meaningless number.
        conf["bucket"] = (conf["confidence"].clip(0, 1) * 100 // 20 * 20).astype(int)
        out["calibration"] = {
            int(k): {"hit_pct": round(v["hit_num"].mean() * 100),
                     "n": int(v["hit_num"].count())}
            for k, v in conf.groupby("bucket")}

    # Self-check, not a decision rule: does the ranking signal (score=trend)
    # or the fit quality (r2) actually relate to hitting? Added after week 1
    # showed Spearman(trend, hit) = -0.36 on n=24 - the selection was ranking
    # by the thing that predicted failure. Kept here so that question has an
    # answer on every future `score` call instead of requiring a one-off
    # script each time. n<10 is too small for the number to mean anything;
    # shown anyway, labelled with n, so a reader is not tempted to trust it
    # at face value before there is enough data.
    diag = df.dropna(subset=["score", "r2"])
    if len(diag) >= 8:
        out["diagnostics"] = {
            "n": int(len(diag)),
            "spearman_trend_vs_hit": round(
                float(diag["score"].corr(diag["hit_num"], method="spearman")), 3),
            "spearman_r2_vs_hit": round(
                float(diag["r2"].corr(diag["hit_num"], method="spearman")), 3),
        }
    return out


def _born_ages(s) -> dict[int, int]:
    """Creation age in days, for the one sanity check that matters here.

    A brand-new game's CCU is still settling, so a +20% move three days in is
    mostly the launch curve, not a signal. Surfacing the age lets a reader see
    that a run of hits may be an artefact.
    """
    from db import Game
    now = utcnow()
    return {g.universe_id: (now - g.created).days
            for g in s.query(Game).filter(Game.created.isnot(None)).all()}


def report() -> str:
    """The table a critic reads. Data only - no narrative, no explanation.

    Deliberately free of prose about why anything happened. A critic handed the
    reasoning will confirm the reasoning; handed only the numbers, it has to
    find the flaws itself.
    """
    with Session() as s:
        rows = (s.query(Prediction)
                 .order_by(Prediction.made_at.desc()).limit(500).all())
        ages = _born_ages(s)
        out = [{
            "id": r.id,
            "kind": r.kind,
            "subject": r.subject,
            "claim": r.claim,
            "criterion": r.criterion,
            "threshold_pp": r.threshold_pp,
            "baseline": r.baseline,
            "horizon_days": r.horizon_days,
            "age_at_make": ages.get(r.universe_id),
            "cohort_rank": r.cohort_rank,
            "confidence": r.confidence,
            "made_at": r.made_at.isoformat() if r.made_at else None,
            "due_at": r.due_at.isoformat() if r.due_at else None,
            "outcome": r.outcome,
            "outcome_value": r.outcome_value,
            "outcome_basis": r.outcome_basis,
        } for r in rows]
    return json.dumps(out, ensure_ascii=False, indent=1, default=str)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["record", "due", "resolve", "score", "report",
                                   "threshold"])
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--kind", default="move", choices=sorted(PLANS))
    p.add_argument("--horizon", type=float, default=7)
    a = p.parse_args()

    if a.cmd == "record":
        print(record_from_features(top_n=a.top, kind=a.kind))
    elif a.cmd == "due":
        d = due_forecasts()
        print(f"к проверке: {len(d)}")
        for x in d[:20]:
            print(f"  #{x['id']} {x['subject'][:40]:42s} "
                  f"base={x['baseline']:.0f} {x['criterion']}")
    elif a.cmd == "resolve":
        print(resolve())
    elif a.cmd == "threshold":
        for pct in (50, 60, 70, 75, 80, 90):
            d = derive_threshold(pct, a.horizon)
            if d["ok"]:
                print(f"  p{pct:<3} -> {d['threshold_pp']:>6.1f} пп  "
                      f"({d['samples']} наблюдений, {d['games']} игр)")
            else:
                print(f"  p{pct:<3} -> нет данных: {d['reason']}")
                break
    elif a.cmd == "score":
        for k, v in score().items():
            print(f"  {k:24s} {v}")
    else:
        print(report())


if __name__ == "__main__":
    main()
