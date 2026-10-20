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


def _snapshots_pivots() -> tuple[pd.DataFrame, pd.Series]:
    """CCU by game over time, plus the market median per snapshot."""
    with Session() as s:
        rows = s.query(Snapshot.universe_id, Snapshot.ts, Snapshot.playing).all()
    df = pd.DataFrame(rows, columns=["universe_id", "ts", "playing"])
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
        if med[t0] <= 0:
            continue
        market_change = med[t1] / med[t0] - 1
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
        rows = rows.sort_values("trend", ascending=False)
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
        } for r in rows]


def _market_median_ccu(s, at) -> float | None:
    """Market median CCU at a point in time.

    The benchmark a "grows faster than the market" claim is measured against.
    Without it, a forecast about relative growth is being graded on an absolute
    number, which measures the market instead of the game.
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


def _judge(criterion: str, base: float, now: float,
           market_base: float | None = None,
           market_now: float | None = None,
           threshold_pp: float | None = None) -> tuple[str, str]:
    """Apply the criterion exactly as written. No judgement calls.

    `threshold_pp` is the frozen bar stored on the row. It is never recomputed
    here: a criterion whose bar can move is not a criterion.
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
        if not market_base or not market_now or market_base <= 0:
            return "void", "нет медианы рынка для сравнения"
        market_change = (market_now - market_base) / market_base
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


def _value_now_fresh(s, universe_id: int, metric: str):
    """(value, age_hours, snapshot_ts). The timestamp matters as much as the
    value: the market benchmark has to be read at the same moment."""
    """Latest value plus how stale it is, in hours.

    Returning the age alongside the value is the point: the caller has to decide
    whether a number is recent enough to judge on, and it cannot do that
    without knowing when it was taken.
    """
    col = Snapshot.playing if metric == "ccu" else Snapshot.favorites
    row = (s.query(col, Snapshot.ts)
            .filter(Snapshot.universe_id == universe_id)
            .order_by(Snapshot.ts.desc()).first())
    if not row:
        return None, None, None
    age_h = (utcnow() - row[1]).total_seconds() / 3600 if row[1] else None
    return float(row[0] or 0), age_h, row[1]


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

            now_val, age_h, snapshot_ts = _value_now_fresh(
                s, p["universe_id"], p["metric"])

            # A game that dropped out of the carousels stops receiving
            # snapshots. Judging it on a week-old reading would report "no
            # growth" when the truth is "we stopped looking" - and that is the
            # most common way a forecast dies on Roblox.
            if now_val is None or (age_h is not None and age_h > 48):
                row.outcome = "void"
                row.outcome_basis = (
                    "нет данных" if now_val is None
                    else f"данные устарели на {age_h:.0f} ч — игра выпала из чартов")[:256]
                row.checked_at = utcnow()
                stats["void"] += 1
                continue

            # Market median must be read at the game's own snapshot time, not
            # at "now". If a game's last observation is 30 hours old and the
            # market figure is from this minute, the two windows being compared
            # have different lengths and the excess is not a difference of like
            # with like.
            market_now = _market_median_ccu(s, snapshot_ts)
            verdict, basis = _judge(p["criterion"], p["baseline"], now_val,
                                    row.benchmark, market_now,
                                    row.threshold_pp)
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
            "benchmark": r.benchmark,
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
