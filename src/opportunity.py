"""Decision engine: what game to make next, and why.

    python src/opportunity.py

Reads the same feature table the dashboard uses, but instead of sorting games
it groups them into *opportunities* - a niche, or a mechanic template inside a
niche - and answers one question per row:

    if I build in here now, is the door open?

The rules are deliberately strict. A niche only makes the list when all four
of these hold at once:

  DEMAND      people are already playing in it (total CCU is not noise)
  FRESHNESS   young games in it are winning, not just old incumbents
  HEADROOM    the top game is not so far ahead that a copy cannot reach it
  MOMENTUM    it is growing now, not five months ago

Anything that fails one of those is printed in the rejected section with the
reason, because "why not" is as useful as "why".
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, flag_suspect  # noqa: E402
from templates import build_families, family_summary, tokens  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports")

# A niche smaller than this cannot pay for the work of building in it.
MIN_NICHE_CCU = 20_000
# The top game must not be more than this multiple of the median, or you are
# competing with a monopoly rather than an open market.
MAX_TOP_RATIO = 60.0
# Below this share of young games, the niche is settled and closed.
MIN_YOUNG_SHARE = 0.15


def _fmt(n: float | None, digits: int = 0) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    if digits:
        return f"{n:,.{digits}f}"
    return f"{int(round(n)):,}"


def _pct(x: float | None) -> str:
    """Growth multiplier as a readable percentage change."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or x <= 0:
        return "—"
    return f"{(x - 1) * 100:+.0f}%"


def niche_table(f: pd.DataFrame) -> pd.DataFrame:
    """One row per niche with the four gates as explicit columns."""
    rows = []
    for niche, g in f.groupby("niche"):
        if niche in ("other", None) or pd.isna(niche):
            continue
        ccu = g["ccu"].fillna(0)
        total = float(ccu.sum())
        if total < 1:
            continue
        ages = g["age_days"][g["age_days"] >= 0]
        young = float((ages <= 60).mean()) if len(ages) else 0.0
        top = float(ccu.max())
        med = float(ccu.median()) if len(ccu) else 0.0
        top_ratio = top / med if med > 0 else 999.0

        # Momentum: median daily growth across games in the niche. Median, not
        # mean - one rocket must not drag a dead niche up.
        #
        # BUT a 24h comparison is only valid with a full day of history. Roblox
        # CCU follows a hard daily cycle: 08:00 UTC is ~40% below 20:00 UTC. With
        # 13h of data, g_d1 compares morning against last evening and every niche
        # reads as "falling", which is an artefact, not a trend. So the column is
        # reported as unknown until there is enough span, and the gate below
        # treats unknown as "not disqualifying".
        gm = g["g_d1"].dropna() if "g_d1" in g.columns else pd.Series(dtype=float)
        span_h = float(g["history_hours"].max()) if "history_hours" in g.columns else 0.0
        momentum = float(gm.median()) if (len(gm) and span_h >= 30) else np.nan

        # Freshness proof: are the young games actually earning, or merely alive?
        # Mask the frame itself - `ages` was filtered, so its index no longer
        # lines up with `ccu` and selecting by it silently sums everything.
        young_mask = g["age_days"] >= 0
        young_mask &= g["age_days"] <= 60
        young_ccu = float(ccu[young_mask].sum())
        young_share_ccu = young_ccu / total if total else 0.0

        rows.append({
            "niche": niche,
            "games": len(g),
            "total_ccu": total,
            "median_ccu": med,
            "top_ccu": top,
            "top_ratio": top_ratio,
            "median_age": float(ages.median()) if len(ages) else np.nan,
            "young_share": young,
            "young_ccu_share": young_share_ccu,
            "momentum": momentum,
            "growing": int((gm > 1.15).sum()) if len(gm) else 0,
            "yt_videos": int(g["yt_videos"].fillna(0).sum()) if "yt_videos" in g else 0,
        })
    return pd.DataFrame(rows)


def score_opportunities(nt: pd.DataFrame) -> pd.DataFrame:
    """Rank niches, and record which gate each one fails."""
    if nt.empty:
        return nt

    out = nt.copy()
    fails: list[list[str]] = []
    scores: list[float] = []

    for r in out.itertuples():
        why: list[str] = []
        s = 0.0

        # --- DEMAND: log-scaled, saturates around 1M CCU
        demand = np.clip(np.log10(max(r.total_ccu, 1) / MIN_NICHE_CCU) / 1.7, 0, 1)
        if r.total_ccu < MIN_NICHE_CCU:
            why.append(f"мало спроса ({_fmt(r.total_ccu)} CCU)")
        s += demand * 32

        # --- FRESHNESS: can a new entrant still win here?
        fresh = np.clip(r.young_share / 0.5, 0, 1)
        if r.young_share < MIN_YOUNG_SHARE:
            why.append(f"ниша закрыта: молодых игр {r.young_share:.0%}")
        s += fresh * 26

        # --- HEADROOM: is the leader beatable?
        head = np.clip(1.0 - np.log10(max(r.top_ratio, 1)) / np.log10(MAX_TOP_RATIO), 0, 1)
        if r.top_ratio > MAX_TOP_RATIO:
            why.append(f"монополия: лидер в {r.top_ratio:.0f}× от медианы")
        s += head * 24

        # --- MOMENTUM: growth right now
        if pd.notna(r.momentum):
            mom = np.clip((r.momentum - 1.0) / 0.5, 0, 1)
            if r.momentum < 1.0:
                why.append(f"падает ({_pct(r.momentum)})")
        else:
            mom = 0.0
        s += mom * 18

        scores.append(round(s, 1))
        fails.append(why)

    out["score"] = scores
    out["fails"] = fails
    out["verdict"] = out["fails"].map(lambda x: "делать" if not x else "пропустить")
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def template_ideas(f: pd.DataFrame, opp: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    """Mechanic templates inside the niches that passed, with a copy target.

    A niche is a direction; a template is a buildable thing. This is what the
    user actually acts on: "make an X-style game in niche Y, the closest thing
    that works already does N CCU and is M days old."
    """
    if opp.empty:
        return pd.DataFrame()
    good = set(opp[opp["verdict"] == "делать"]["niche"])
    if not good:
        return pd.DataFrame()

    # build_families RETURNS a copy with family_id attached - the original frame
    # has no such column, so the niche lookup must read from the returned one.
    fb = build_families(f)
    summ = family_summary(fb)
    if summ.empty:
        return pd.DataFrame()

    fam_niche = (fb.dropna(subset=["family_id"])
                   .groupby("family_id")["niche"]
                   .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None))
    summ["niche"] = summ["family_id"].map(fam_niche)
    sub = summ[summ["niche"].isin(good)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["headroom"] = sub.apply(
        lambda r: 1.0 if r["max_ccu"] < 5_000 else (0.6 if r["max_ccu"] < 50_000 else 0.3),
        axis=1)
    sub["idea_score"] = (sub["enter_score"] * 0.6 + sub["headroom"] * 100 * 0.4).round(1)
    return sub.sort_values("idea_score", ascending=False).head(limit).reset_index(drop=True)


def winner_dna(f: pd.DataFrame, top_n: int = 60) -> dict:
    """What the current winners have in common - the shared build traits.

    Not a score, just the vocabulary: which mechanics keep showing up among the
    games that are growing. If "steal" appears in half of them, that is the
    shape of the market right now.
    """
    if f.empty:
        return {"tokens": [], "median_age": None, "median_ccu": None, "n": 0}
    gm = f["g_d1"].dropna()
    winners = f.loc[gm[gm > 1.2].index] if len(gm) else f.nlargest(top_n, "ccu")
    winners = winners[winners["ccu"] >= 500]
    if winners.empty:
        winners = f.nlargest(top_n, "ccu")

    counts: dict[str, int] = {}
    for t in winners.get("tags", pd.Series(dtype=object)).dropna():
        for tok in str(t).split(","):
            tok = tok.strip()
            if tok:
                counts[tok] = counts.get(tok, 0) + 1

    return {
        "tokens": sorted(counts.items(), key=lambda x: -x[1])[:10],
        "median_age": float(winners["age_days"].median()) if len(winners) else None,
        "median_ccu": float(winners["ccu"].median()) if len(winners) else None,
        "n": len(winners),
    }


def render(f: pd.DataFrame, opp: pd.DataFrame, ideas: pd.DataFrame,
           dna: dict) -> str:
    L: list[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    hist_h = float(f["history_hours"].max()) if len(f) else 0.0

    L.append(f"# Что делать — {stamp}\n")

    if hist_h < 24:
        L.append(f"> ⚠️ Истории {hist_h:.0f} ч — меньше суток. Оценки роста пока "
                 f"ненадёжны, через 3 дня цифры сдвинутся. Ниши ниже выбраны "
                 f"по спросу, возрасту и концентрации, а не по динамике.\n")

    # ---------- 1. the decision
    act = opp[opp["verdict"] == "делать"] if not opp.empty else pd.DataFrame()
    L.append("## Куда идти\n")
    if act.empty:
        L.append("_Открытых окон нет. Все ниши либо малы, либо закрыты лидером._\n")
    else:
        for i, r in enumerate(act.head(5).itertuples(), 1):
            young_ccu = r.young_ccu_share * 100
            L.append(f"### {i}. {r.niche} — {r.score:.0f}/100\n")
            L.append(f"Игр в нише **{r.games}**, суммарно **{_fmt(r.total_ccu)}** CCU. "
                     f"Лидер {_fmt(r.top_ccu)} при медиане {_fmt(r.median_ccu)} "
                     f"— разрыв {r.top_ratio:.0f}×.\n")
            L.append(f"- Медианный возраст: **{_fmt(r.median_age)} дн**")
            L.append(f"- Молодых игр (до 60 дней): **{r.young_share:.0%}**, "
                     f"и они держат **{young_ccu:.0f}%** всего онлайна ниши")
            if pd.notna(r.momentum):
                L.append(f"- Рост за сутки (медиана): **{_pct(r.momentum)}**, "
                         f"растут {r.growing} из {r.games}")
            L.append(f"- Упоминаний на YouTube: {r.yt_videos}\n")
            L.append(f"**Почему сейчас:** {_why(r)}\n")

    # ---------- 2. concrete builds
    if not ideas.empty:
        L.append("## Что именно строить\n")
        L.append("| Шаблон | Ниша | Игр | CCU ниши | Ср. возраст | Вход | Примеры |")
        L.append("|---|---|---|---|---|---|---|")
        for r in ideas.itertuples():
            L.append(f"| **{r.family}** | {r.niche} | {r.games} | "
                     f"{_fmt(r.total_ccu)} | {_fmt(r.mean_age)} дн | "
                     f"{r.idea_score:.0f} | {r.members} |")
        L.append("")

    # ---------- 3. winner DNA
    if dna["n"]:
        L.append("## Из чего состоит успех сейчас\n")
        L.append(f"По {dna['n']} играм, которые растут прямо сейчас:\n")
        if dna["median_age"] is not None and not pd.isna(dna["median_age"]):
            L.append(f"- Медианный возраст: **{dna['median_age']:.0f} дней**")
        if dna["median_ccu"] is not None and not pd.isna(dna["median_ccu"]):
            L.append(f"- Медианный онлайн: **{_fmt(dna['median_ccu'])}** CCU")
        if dna["tokens"]:
            toks = ", ".join(f"`{t}` ({n})" for t, n in dna["tokens"][:8])
            L.append(f"- Частые механики: {toks}")
        L.append("")

    # ---------- 4. rejected, with reasons
    skip = opp[opp["verdict"] == "пропустить"] if not opp.empty else pd.DataFrame()
    if not skip.empty:
        L.append("## Куда не идти и почему\n")
        L.append("| Ниша | CCU | Игр | Причина |")
        L.append("|---|---|---|---|")
        for r in skip.head(12).itertuples():
            L.append(f"| {r.niche} | {_fmt(r.total_ccu)} | {r.games} | "
                     f"{'; '.join(r.fails)} |")
        L.append("")

    L.append("---\n")
    L.append(_methodology())
    return "\n".join(L)


def _why(r) -> str:
    bits = []
    if r.top_ratio < 15:
        bits.append("лидера можно догнать, он не держит рынок")
    if r.young_share >= 0.3:
        bits.append("треть ниши — игры младше двух месяцев")
    if pd.notna(r.momentum) and r.momentum > 1.2:
        bits.append(f"ниша растёт на {_pct(r.momentum)} в сутки")
    if r.yt_videos == 0:
        bits.append("на YouTube ещё не освоено")
    elif r.yt_videos < 5:
        bits.append("YouTube только начал замечать")
    return "; ".join(bits) or "спрос есть, вход не закрыт"


def _methodology() -> str:
    return (
        "## Как это считается\n\n"
        "Ниша проходит, только если выполнены четыре условия одновременно:\n\n"
        "1. **Спрос** — суммарный онлайн не ниже 20 000. Меньше — не окупит разработку.\n"
        "2. **Свежесть** — доля игр младше 60 дней не ниже 15%. Если ниша состоит "
        "из ветеранов, новичку там нечего делать.\n"
        "3. **Пространство** — лидер не более чем в 60 раз больше медианной игры. "
        "Выше — это монополия, копия не пробьётся.\n"
        "4. **Динамика** — медианный суточный рост не отрицательный.\n\n"
        "Вес: спрос 32, свежесть 26, пространство 24, динамика 18.\n\n"
        "**Чего здесь нет:** базовой частоты («сколько новых игр в нише провалилось»). "
        "Сборщик видит только игры, попавшие в карусели Roblox, поэтому знаменатель "
        "неполный. Пока это не исправлено, считай score ориентиром, а не гарантией.\n"
    )


def main() -> None:
    f = build_features()
    if f.empty:
        print("нет данных")
        return
    f = flag_suspect(f)
    f = f[~f["suspect_bot"]]

    nt = niche_table(f)
    opp = score_opportunities(nt)
    ideas = template_ideas(f, opp)
    dna = winner_dna(f)

    md = render(f, opp, ideas, dna)
    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(OUT, f"{stamp}_opportunity.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
