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
import os
import sys
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from db import Prediction, Session, Snapshot, utcnow  # noqa: E402

# How each kind of forecast is judged. Deliberately blunt: a percentage move.
# Anything softer ("became popular") cannot be scored, and a soft criterion is
# how a forecast gets reinterpreted into a hit after the fact.
PLANS = {
    # A game the digest flagged as rising should still be up a week later.
    # Short horizon because Roblox momentum decays fast.
    "move":  {"metric": "ccu", "criterion": "up_20pct", "horizon": 7},
    # A clone target gets longer: a copy needs time to show whether the
    # original held.
    "clone": {"metric": "ccu", "criterion": "up_20pct", "horizon": 14},
    # An anime premiere should lift its mapped niche. Judged on the niche
    # median, not one game, because we cannot know which clone wins.
    "anime": {"metric": "ccu", "criterion": "up_10pct", "horizon": 21},
}

# Below this a percentage move is noise: 50 -> 60 is +20% and means nothing.
MIN_BASELINE_CCU = 500


def _ccu_at_latest(s, universe_id: int) -> float | None:
    row = (s.query(Snapshot.playing)
            .filter(Snapshot.universe_id == universe_id)
            .order_by(Snapshot.ts.desc()).first())
    return float(row[0] or 0) if row else None


def _already_open(s, kind: str, universe_id: int | None, subject: str) -> bool:
    """One open forecast per subject per kind.

    Without this, eight runs a day would create eight overlapping forecasts of
    the same game, and one hit would count eight times in the score. The sample
    would grow without the evidence growing.
    """
    q = s.query(Prediction).filter(
        Prediction.kind == kind,
        Prediction.outcome.is_(None),
        Prediction.subject == subject)
    if universe_id is not None:
        q = q.filter(Prediction.universe_id == universe_id)
    return q.count() > 0


def record_from_features(top_n: int = 10, kind: str = "move") -> dict:
    """Log what this run recommends, using the same rank the digest sorts by.

    Called right after digest_html so the record matches what was actually
    shown. Re-deriving the selection independently would let the two drift, and
    then the score would grade a ranking nobody saw.
    """
    from features import build_features

    plan = PLANS[kind]
    f = build_features()
    if f.empty:
        return {"recorded": 0, "reason": "no features"}

    f = f[f["ccu"] >= MIN_BASELINE_CCU].dropna(subset=["cohort_rank"])
    if f.empty:
        return {"recorded": 0, "reason": "nothing above floor"}

    made = skipped = 0
    with Session() as s:
        for r in f.nlargest(top_n, "cohort_rank").to_dict("records"):
            subject = (r.get("name") or "")[:256]
            uid = int(r["universe_id"])
            if not subject or _already_open(s, kind, uid, subject):
                skipped += 1
                continue
            base = float(r.get("ccu") or 0)
            if base <= 0:
                skipped += 1
                continue

            s.add(Prediction(
                kind=kind,
                universe_id=uid,
                subject=subject,
                claim=(f"CCU вырастет ≥20% за {plan['horizon']} дн "
                       f"(от {base:.0f})"),
                metric=plan["metric"],
                baseline=base,
                criterion=plan["criterion"],
                horizon_days=plan["horizon"],
                score=r.get("score"),
                confidence=r.get("confidence"),
                cohort_rank=r.get("cohort_rank"),
                due_at=utcnow() + timedelta(days=plan["horizon"]),
            ))
            made += 1
        s.commit()
    return {"recorded": made, "skipped_duplicate": skipped}


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
        } for r in rows]


def _judge(criterion: str, base: float, now: float) -> tuple[str, str]:
    """Apply the criterion exactly as written. No judgement calls."""
    if base <= 0:
        return "void", "baseline zero"
    change = (now - base) / base
    table = {"up_20pct": 0.20, "up_10pct": 0.10}
    if criterion not in table:
        return "void", "unknown criterion"
    bar = table[criterion]
    if change >= bar:
        return "hit", f"+{change*100:.0f}%"
    if change >= 0:
        return "partial", f"+{change*100:.0f}% ниже порога"
    return "miss", f"{change*100:.0f}%"


def _value_now(s, universe_id: int, metric: str) -> float | None:
    col = Snapshot.playing if metric == "ccu" else Snapshot.favorites
    row = (s.query(col)
            .filter(Snapshot.universe_id == universe_id)
            .order_by(Snapshot.ts.desc()).first())
    return float(row[0] or 0) if row else None


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

            now_val = (_value_now(s, p["universe_id"], p["metric"])
                       if p["universe_id"] else None)
            if now_val is None:
                row.outcome = "void"
                row.outcome_basis = "нет данных на дату проверки"
                row.checked_at = utcnow()
                stats["void"] += 1
                continue

            verdict, basis = _judge(p["criterion"], p["baseline"], now_val)
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

    moves = df.dropna(subset=["baseline", "outcome_value"])
    moves = moves[moves["baseline"] > 0]
    if not moves.empty:
        pct = (moves["outcome_value"] - moves["baseline"]) / moves["baseline"]
        out["median_move_pct"] = round(float(pct.median()) * 100, 1)
        # Share of all judged forecasts that merely went up, ignoring the 20%
        # bar. That is what "just pick anything big" would have scored.
        out["baseline_up_share_pct"] = round(float((pct >= 0).mean()) * 100, 1)
        out["baseline_20pct_share_pct"] = round(float((pct >= 0.20).mean()) * 100, 1)

    conf = df.dropna(subset=["confidence"]).copy()
    if len(conf) >= 5:
        conf["bucket"] = (conf["confidence"] // 20 * 20).astype(int)
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
    p.add_argument("cmd", choices=["record", "due", "resolve", "score", "report"])
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--kind", default="move", choices=sorted(PLANS))
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
    elif a.cmd == "score":
        for k, v in score().items():
            print(f"  {k:24s} {v}")
    else:
        print(report())


if __name__ == "__main__":
    main()
