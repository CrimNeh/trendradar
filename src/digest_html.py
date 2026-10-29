"""Presentation-grade HTML digest: which games are actually moving.

    python src/digest_html.py

Writes docs/index.html, which GitHub Pages serves as the dashboard.

The hard part is not the markup, it is the comparison. Roblox CCU follows a
hard daily cycle - the market median swings from ~3,750 at 20:00 UTC to ~2,340
at 08:00 UTC, a 38% drop that has nothing to do with any game. Ranking raw
deltas therefore says "everything is falling" every morning.

So every game is compared against the market median at the same instant:

    share(t) = ccu(t) / median_ccu_across_all_games(t)

A game whose share rose while the whole market fell is genuinely gaining. This
removes the cycle without needing a full day of history, which matters because
the dataset is only hours old.
"""
from __future__ import annotations

import html
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from db import Game, Session, Mention, Snapshot, YtVideo  # noqa: E402
from sqlalchemy import text  # noqa: E402
from features import build_features, flag_suspect  # noqa: E402
import forecast  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs")

# Below this a gain is noise, not a trend.
MIN_CCU = 500
# Share change needed to call something "rising" or "falling".
# Share growth per day, in log2 units: >0.09 means the game's slice of the
# market is growing ~6.4%/day faster than the market itself.
# Was 0.15 while `trend` was slope * hours-of-history; at the 41 h archive
# of 21 Oct those were the same bar. Rescaled once, so the gate keeps the
# same strictness now that `trend` no longer grows with the archive.
MOVE_THRESHOLD = 0.09

# How many rising cards the page renders. predictions.py reads this so the
# ledger records exactly what was shown - it previously wrote 10 while the page
# rendered 8, which put two unevaluated games into the hit-rate sample.
RISING_CARDS = 8


def _esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def _fmt(n, digits: int = 0) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    return f"{n:,.{digits}f}" if digits else f"{int(round(n)):,}"


def _pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{(x - 1) * 100:+.0f}%"


# Orphan games (games.origin = 'orphan': discovered outside the Roblox
# carousels, see collect.collect_orphans) are snapshotted but kept OUT of the
# market median and the per-game share calculations while this is False.
#
# The reason is the ruler, not the quality of those games. Orphans are
# systematically smaller (measured 2026-10-28: median 116 CCU overall, 1812
# among those above the 500 floor, against a carousel market median of
# ~3000-4000), so folding 161 of them in shifts the median that every "did this
# game beat the market" number is divided by. The page and the forecast ledger
# both read this function - patching only predictions.py would leave the digest
# ranking games against one market while the ledger graded them against another.
#
# Flip to True once the week-2 comparison is finished and the sample can move
# without invalidating the very comparison it exists to make.
INCLUDE_ORPHANS = False


def _orphan_uids(s) -> set[int]:
    """Games excluded from market medians and share calculations right now."""
    if INCLUDE_ORPHANS:
        return set()
    return {uid for (uid,) in
            s.query(Game.universe_id).filter(Game.origin == "orphan").all()}


_MARKET_CACHE: tuple | None = None


def market_series() -> tuple[pd.DataFrame, pd.Series]:
    """Snapshot matrix plus the per-timestamp market median.

    The median across ~430 games is a stable proxy for the platform-wide daily
    cycle, and it is immune to one giant game moving the average.

    Memoised per process. The page is built twice per deployment (once in the
    snapshot job, once in deploy) and each call transfers the whole snapshots
    table; within one process the data cannot change.
    """
    global _MARKET_CACHE
    if _MARKET_CACHE is not None:
        return _MARKET_CACHE
    with Session() as s:
        rows = s.query(Snapshot.universe_id, Snapshot.ts, Snapshot.playing).all()
        drop = _orphan_uids(s)
    df = pd.DataFrame(rows, columns=["universe_id", "ts", "playing"])
    if df.empty:
        _MARKET_CACHE = (pd.DataFrame(), pd.Series(dtype=float))
        return _MARKET_CACHE
    if drop:
        df = df[~df["universe_id"].isin(drop)]
        if df.empty:
            _MARKET_CACHE = (pd.DataFrame(), pd.Series(dtype=float))
            return _MARKET_CACHE
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    piv = df.pivot_table(index="ts", columns="universe_id",
                         values="playing", aggfunc="last").sort_index()
    _MARKET_CACHE = (piv, piv.median(axis=1))
    return _MARKET_CACHE


def relative_moves(piv: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    """Per-game share trend, fitted across every snapshot rather than endpoints.

    Comparing only the first and last snapshot labelled 154 of 414 games as
    "rising" - an artefact, because the window starts at the evening peak and
    ends at the morning trough, so anything that merely fell less than the
    market looks like a winner. A least-squares slope over all points plus a
    consistency score separates a real climb from a single lucky reading.
    """
    if piv.empty or len(piv) < 2:
        return pd.DataFrame()

    share = piv.div(med.replace(0, np.nan), axis=0)
    hours = (piv.index - piv.index[0]).total_seconds() / 3600
    x = np.asarray(hours, dtype=float)

    first = share.iloc[0]
    last = share.iloc[-1]
    ok = (first > 0) & (last > 0) & last.notna() & first.notna()

    rows = []
    for uid in last[ok].index:
        y = share[uid].to_numpy(dtype=float)
        good = ~np.isnan(y) & (y > 0)
        if good.sum() < 3:
            continue
        ly = np.log2(y[good])
        xx = x[good]
        try:
            slope, intercept = np.polyfit(xx, ly, 1)
        except Exception:
            continue

        # R^2: how well a straight line describes the path. A game that jumped
        # once and sat still scores low even if the endpoints differ.
        pred = slope * xx + intercept
        ss_res = float(((ly - pred) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Directional consistency: share of consecutive steps that agree.
        diffs = np.diff(ly)
        agree = float((np.sign(diffs) == np.sign(slope)).mean()) if len(diffs) else 0.0

        rows.append({
            "universe_id": uid,
            "share_change": float(last[uid] / first[uid]),
            "share_slope": float(slope),
            "r2": float(max(r2, 0.0)),
            "consistency": agree,
            "points": int(good.sum()),
            "ccu_first": float(piv.iloc[0][uid]),
            "ccu_last": float(piv.iloc[-1][uid]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ccu_delta"] = out["ccu_last"] - out["ccu_first"]
    out["span_h"] = (piv.index[-1] - piv.index[0]).total_seconds() / 3600
    return out


def sparkline(values: list[float], w: int = 132, h: int = 34) -> str:
    """Inline SVG of the raw series - the daily dip is the point, not a bug."""
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    step = w / (len(vals) - 1)
    pts = [(i * step, h - 3 - (v - lo) / rng * (h - 6)) for i, v in enumerate(vals)]

    d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f} " + " ".join(
        f"L {x:.1f} {y:.1f}" for x, y in pts[1:])
    area = d + f" L {pts[-1][0]:.1f} {h} L {pts[0][0]:.1f} {h} Z"
    rising = (vals[-1] >= vals[0])
    stroke = "#3fb950" if rising else "#f85149"
    uid = abs(hash(tuple(round(v, 1) for v in vals))) % 10000

    return (
        f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
        f'<defs><linearGradient id="g{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{stroke}" stop-opacity=".28"/>'
        f'<stop offset="100%" stop-color="{stroke}" stop-opacity="0"/>'
        f'</linearGradient></defs>'
        f'<path d="{area}" fill="url(#g{uid})"/>'
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="1.8" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'</svg>'
    )


def yt_evidence() -> dict[int, dict]:
    """YouTube activity per game: how many videos, how fast they are gaining."""
    with Session() as s:
        m = s.query(Mention.universe_id, Mention.video_id, Mention.lang).all()
        # Only videos that actually carry a mention. This used to load every
        # row in yt_videos (38724 / 4.81 MB on Neon, 2026-10-29) and then merge
        # them onto the mention table - so ~23000 rows were transferred and
        # discarded by the `how="left"` join, whose only effect they could have
        # had was to fill in a video_id the mention table already guarantees.
        #
        # Restricting to the mentioned ids is behaviourally identical (the
        # merge is keyed on mention.video_id, so a video with no mention can
        # never contribute a row) and halves the transfer to 2.03 MB. Scoped by
        # a subquery rather than a materialised id list so the DB does the
        # filtering and only matching rows cross the wire.
        v = s.execute(text("""
            select v.video_id, v.views, v.views_per_hour, v.title, v.published
            from yt_videos v
            where exists (select 1 from mentions m where m.video_id = v.video_id)
        """)).all()

    if not m or not v:
        return {}
    md = pd.DataFrame(m, columns=["universe_id", "video_id", "lang"])
    vd = pd.DataFrame(v, columns=["video_id", "views", "vph", "title", "published"])
    merged = md.merge(vd, on="video_id", how="left")

    out: dict[int, dict] = {}
    for uid, g in merged.groupby("universe_id"):
        views = g["views"].fillna(0)
        vph = g["vph"].fillna(0)
        top = g.loc[views.idxmax()] if len(g) else None
        out[int(uid)] = {
            "videos": int(len(g)),
            "views": int(views.sum()),
            "vph": float(vph.sum()),
            "langs": int(g["lang"].dropna().nunique()),
            "top_title": (str(top["title"])[:70] if top is not None else ""),
            "top_views": int(top["views"]) if top is not None and pd.notna(top["views"]) else 0,
        }
    return out


def build_rows(f: pd.DataFrame, moves: pd.DataFrame,
               yt: dict[int, dict]) -> pd.DataFrame:
    if moves.empty or f.empty:
        return pd.DataFrame()
    m = moves.merge(
        f[["universe_id", "name", "niche", "age_days", "tags", "chart_count",
           "best_rank", "rank_delta_d1", "history_hours",
           # Conversion signals. These exist for ~95% of games, against
           # yt_videos at 5%, so they carry the "is it alive" question that the
           # YouTube panel could never answer.
           "fav_rate", "fav_gain_6h", "fav_per_visit", "r_fav_pct",
           "r_favrate_pct", "vote_rate", "like_ratio", "cohort_rank"]],
        on="universe_id", how="inner")
    m = m[m["ccu_last"].fillna(0) >= MIN_CCU]
    m["yt"] = m["universe_id"].map(lambda u: yt.get(int(u), {}))
    m["yt_videos"] = m["yt"].map(lambda d: d.get("videos", 0) if d else 0)
    m["yt_views"] = m["yt"].map(lambda d: d.get("views", 0) if d else 0)
    m["yt_vph"] = m["yt"].map(lambda d: d.get("vph", 0.0) if d else 0.0)
    m["yt_langs"] = m["yt"].map(lambda d: d.get("langs", 0) if d else 0)
    # Normalised to a per-day rate, NOT multiplied by the length of the whole
    # history. The first version used `share_slope * span_h`, where span_h is
    # every hour ever collected. The slope is per hour, so the product grew with
    # the archive: measured, the same slope scored 1.0 at 41 h of history, 17.5
    # at 720 h and 52.6 at 2160 h. MOVE_THRESHOLD is a constant, so after a
    # month the gate would have been ~17x weaker than the day it was chosen and
    # almost the whole market would have qualified.
    #
    # Ranking was never affected - span_h is identical for every game, so it
    # cancelled out of the sort - but the threshold was. A per-day rate means
    # the number keeps its meaning as the archive grows.
    m["trend"] = m["share_slope"] * 24
    return m.sort_values("trend", ascending=False).reset_index(drop=True)


def why(row) -> str:
    """One line of plain reasoning, built from the numbers next to it."""
    bits = []
    # Wording follows `trend`, not `share_change`: trend is what orders the
    # cards, so a sentence built from a different measure can make the 8th card
    # read stronger than the 1st. Measured, they disagree on rank for 6 of 8.
    tr = row.get("trend")
    if tr is not None and not pd.isna(tr):
        if tr >= 1.5:
            bits.append("доля рынка растёт очень быстро")
        elif tr >= 0.85:
            bits.append("уверенно забирает долю у остальных")
        elif tr >= 0.0:
            bits.append("доля растёт, но умеренно")

    d = row.get("ccu_delta", 0) or 0
    if d > 0:
        bits.append(f"плюс {_fmt(d)} игроков за {row['span_h']:.0f} ч")

    age = row.get("age_days", -1)
    if 0 <= age <= 30:
        bits.append(f"вышла {int(age)} дн назад — окно входа ещё открыто")
    elif 0 <= age <= 90:
        bits.append(f"возраст {int(age)} дн, формула ещё не устоялась")

    vids = int(row.get("yt_videos") or 0)
    if vids >= 20:
        bits.append(f"{vids} роликов на YouTube — игра живая")
    elif vids >= 5:
        bits.append(f"{vids} роликов, интерес растёт")
    elif vids > 0:
        bits.append("YouTube только начал замечать")

    if (row.get("yt_langs") or 0) >= 3:
        bits.append(f"ролики на {row['yt_langs']} языках — выходит за пределы одного рынка")

    rd = row.get("rank_delta_d1")
    if rd is not None and not pd.isna(rd) and abs(rd) >= 3:
        bits.append(f"в чартах {'+' if rd > 0 else ''}{int(rd)} позиций")

    # Conversion. This replaces YouTube as the "is it alive" evidence: it is
    # available for 95% of games against 5%, and it is a person choosing to
    # come back rather than a view on a video.
    fr = row.get("fav_rate")
    if fr is not None and not pd.isna(fr) and fr >= 20:
        bits.append(f"+{_fmt(fr)} в избранное за час — возвращаются")
    elif fr is not None and not pd.isna(fr) and fr >= 3:
        bits.append(f"+{_fmt(fr)} в избранное за час")

    fv = row.get("fav_per_visit")
    if fv is not None and not pd.isna(fv) and fv >= 0.04:
        bits.append(f"конверсия в избранное {fv*100:.1f}% — высокая")

    lr = row.get("like_ratio")
    if lr is not None and not pd.isna(lr) and lr >= 0.92:
        bits.append(f"лайков {lr*100:.0f}%")

    return "; ".join(bits) or "держится на месте"


def card(row, piv: pd.DataFrame) -> str:
    sc = row["share_change"]
    up = sc >= 1.0
    cls = "up" if up else "down"
    arrow = "▲" if up else "▼"

    series = piv[row["universe_id"]].dropna().tolist() if row["universe_id"] in piv.columns else []
    spark = sparkline(series)

    vids = row.get("yt_videos", 0)
    yt_html = ""
    if vids:
        yt_html = (
            f'<span class="badge yt" title="видео, найдённые по названию игры">'
            f'▶ {vids} видео · {_fmt(row["yt_views"])} просмотров</span>'
        )
    else:
        yt_html = '<span class="badge dim">нет роликов на YouTube</span>'

    # The single strongest video is the most convincing evidence of life: a
    # count says "someone mentioned it", a title with a view number says people
    # actually watched.
    # Conversion badge. cohort_rank is the age x size percentile, so "топ 6%"
    # means this game converts better than 94% of games its own age and size -
    # the only comparison that means anything here.
    #
    # Wording matters here. cohort_rank is "better than N% of peers", so 56 is
    # the middle. The first version rendered that as "топ 44%", which reads like
    # a top-half result when it is not - 100-cr inverts the meaning.
    cr = row.get("cohort_rank")
    conv_badge = ""
    if cr is not None and not pd.isna(cr):
        cls = "conv-hi" if cr >= 85 else ("conv-mid" if cr >= 55 else "conv-low")
        conv_badge = (f'<span class="badge {cls}">конверсия лучше '
                      f'{cr:.0f}% сверстников</span>')

    ytv = row.get("yt") or {}
    yt_top = ""
    if ytv.get("top_title"):
        yt_top = (f'<span class="badge vid" title="{_esc(ytv["top_title"])}">'
                  f'«{_esc(ytv["top_title"][:46])}» · '
                  f'{_fmt(ytv.get("top_views"))}</span>')

    age = row.get("age_days", -1)
    age_txt = f"{int(age)} дн" if age is not None and age >= 0 else "?"

    # Confidence from the fit: a straight climb through most snapshots is worth
    # more than the same endpoint reached by one spike.
    r2 = float(row.get("r2") or 0)
    cons = float(row.get("consistency") or 0)
    if r2 >= 0.6 and cons >= 0.7:
        conf_cls, conf_txt = "conf-hi", "уверенный тренд"
    elif r2 >= 0.35 and cons >= 0.55:
        conf_cls, conf_txt = "conf-mid", "тренд намечается"
    else:
        conf_cls, conf_txt = "conf-low", "пока шум"

    return f"""
    <article class="card {cls}">
      <div class="card-head">
        <div class="rank">{arrow}</div>
        <div class="title">
          <h3>{_esc(row['name'])}</h3>
          <div class="meta">
            <span class="niche">{_esc(row.get('niche') or '—')}</span>
            <span class="dot">·</span>
            <span>{age_txt}</span>
            <span class="dot">·</span>
            <span>чартов: {int(row['chart_count']) if pd.notna(row.get('chart_count')) else 0}</span>
          </div>
        </div>
        <div class="delta {cls}">
          <div class="big">{_pct(sc)}</div>
          <div class="small">доля рынка</div>
        </div>
      </div>
      <div class="card-body">
        <div class="numbers">
          <div><span class="k">Сейчас</span><span class="v">{_fmt(row['ccu_last'])}</span></div>
          <div><span class="k">Было</span><span class="v">{_fmt(row['ccu_first'])}</span></div>
          <div><span class="k">Изм.</span><span class="v {'pos' if (row['ccu_delta'] or 0) >= 0 else 'neg'}">{'+' if (row['ccu_delta'] or 0) >= 0 else ''}{_fmt(row['ccu_delta'])}</span></div>
          <div><span class="k">В избранное/ч</span><span class="v pos">{_fmt(row.get('fav_rate')) if row.get('fav_rate') is not None and not pd.isna(row.get('fav_rate')) else '—'}</span></div>
        </div>
        {spark}
      </div>
      <div class="why">{_esc(why(row))}</div>
      <div class="badges">
        <span class="badge {conf_cls}">{conf_txt} · R² {r2:.2f}</span>
        {conv_badge}
        {yt_html}
        {yt_top}
      </div>
    </article>"""


def render(f: pd.DataFrame, piv: pd.DataFrame, med: pd.Series,
           rows: pd.DataFrame, yt: dict[int, dict] | None = None,
           fc: dict | None = None) -> str:
    yt = yt or {}
    fc = fc or {"anime": [], "changes": [], "steam": [], "counts": {}}
    stamp = datetime.now(timezone.utc)
    span_h = rows["span_h"].iloc[0] if len(rows) else 0.0
    market_now = float(med.iloc[-1]) if len(med) else 0
    market_first = float(med.iloc[0]) if len(med) else 0

    if len(rows):
        rising = rows[(rows["trend"] >= MOVE_THRESHOLD) & (rows["r2"] >= 0.25)]
        falling = rows[(rows["trend"] <= -MOVE_THRESHOLD) & (rows["r2"] >= 0.25)]
        falling = falling.sort_values("trend")
    else:
        rising = falling = rows

    total_ccu = int(f["ccu"].sum()) if len(f) else 0
    yt_games = int((f["yt_videos"].fillna(0) > 0).sum()) if "yt_videos" in f else 0

    stale = (datetime.now(timezone.utc) - med.index[-1]).total_seconds() / 60 if len(med) else 9999
    fresh_cls = "ok" if stale < 45 else ("warn" if stale < 180 else "bad")
    fresh_txt = (f"данные {int(stale)} мин назад" if stale < 180
                 else f"данные устарели: {stale/60:.1f} ч назад")

    thin = ""
    if span_h < 24:
        thin = (f'<div class="notice">Наблюдаем {span_h:.0f} ч. Сравнение идёт '
                f'с рынком в тот же момент, поэтому суточный цикл не мешает, '
                f'но устойчивость тренда станет видна через сутки.</div>')

    rising_html = "".join(card(r, piv) for r in rising.head(RISING_CARDS).to_dict("records")) \
        if len(rising) else '<p class="empty">Нет игр, растущих быстрее рынка.</p>'

    fall_rows = "".join(
        f"<tr><td class='name'>{_esc(r['name'])}</td>"
        f"<td>{_esc(r.get('niche') or '—')}</td>"
        f"<td class='num'>{_fmt(r['ccu_first'])} → {_fmt(r['ccu_last'])}</td>"
        f"<td class='num neg'>{_pct(r['share_change'])}</td>"
        f"<td class='num'>{_fmt(r['ccu_delta'])}</td>"
        f"<td>{int(r['yt_videos']) if r.get('yt_videos') else '—'}</td></tr>"
        for r in falling.head(12).to_dict("records")
    ) if len(falling) else "<tr><td colspan='6' class='empty'>Нет падающих.</td></tr>"

    # YouTube coverage panel. The collector tracks 34 channels, so only ~23
    # games have any video at all. Showing them by view totals answers "is this
    # game alive outside Roblox" directly, instead of leaving it blank on cards
    # that happen to have no videos yet.
    yt_rows = []
    for uid, d in yt.items():
        if not d.get("videos"):
            continue
        nm = f[f["universe_id"] == uid]
        if nm.empty:
            continue
        yt_rows.append({
            "name": nm.iloc[0]["name"],
            "niche": nm.iloc[0]["niche"],
            "ccu": int(nm.iloc[0]["ccu"] or 0),
            "videos": d["videos"],
            "views": d["views"],
            "langs": d["langs"],
            "top_title": d.get("top_title", ""),
            "top_views": d.get("top_views", 0),
        })
    yt_rows.sort(key=lambda r: -r["views"])
    yt_html_rows = "".join(
        f"<tr><td class='name'>{_esc(r['name'])}</td>"
        f"<td>{_esc(r['niche'] or '—')}</td>"
        f"<td class='num'>{_fmt(r['ccu'])}</td>"
        f"<td class='num'>{r['videos']}</td>"
        f"<td class='num'>{_fmt(r['views'])}</td>"
        f"<td class='num'>{r['langs'] or '—'}</td>"
        f"<td class='vt'>{_esc(r['top_title'][:56]) or '—'}</td></tr>"
        for r in yt_rows[:12]
    ) or "<tr><td colspan='7' class='empty'>Роликов пока нет.</td></tr>"

    # ---------------- forward-looking sections ----------------
    anime_rows = "".join(
        f"<tr><td class='name'>{_esc(a['title'][:52])}</td>"
        f"<td class='num'><span class='in {('hot' if a['days'] <= 7 else ('soon' if a['days'] <= 30 else 'far'))}'>"
        f"{a['days']} дн</span></td>"
        f"<td>{_esc(a['format'])}</td>"
        f"<td class='num'>{a['pop']:,}</td>"
        f"<td class='niches'>{' · '.join(f'<span class=nt>{_esc(n)}</span>' for n in a['niches'][:2])}</td>"
        f"<td class='num'>{a['live']:.0f}</td>"
        f"<td class='num sc'>{a['impact']:.0f}</td></tr>"
        for a in fc["anime"][:10]
    ) or "<tr><td colspan='7' class='empty'>Нет премьер в ближайшие 4 месяца.</td></tr>"

    change_rows = "".join(
        f"<tr><td class='name'>{_esc(c['title'][:62])}</td>"
        f"<td><span class='kind {c['kind']}'>{c['kind']}</span></td>"
        f"<td class='num'>{c['days']} дн</td></tr>"
        for c in fc["changes"][:10]
    ) or "<tr><td colspan='3' class='empty'>Нет свежих изменений.</td></tr>"

    steam_rows = "".join(
        f"<tr><td class='name'>{_esc(st['title'][:40])}</td>"
        f"<td class='num'>{st['peak']:,}</td>"
        f"<td class='vt'>{_esc(st['genre'] or '—')}</td>"
        f"<td class='num'>{st['clone']:.2f}</td></tr>"
        for st in fc["steam"][:8]
    ) or "<tr><td colspan='4' class='empty'>В Steam ничего не растёт подходящего.</td></tr>"

    # The "read this first" strip: three sentences, no tables.
    bullets = forecast.summary_line(fc["anime"], fc["changes"], fc["steam"])
    bullets_html = "".join(f"<li>{_esc(b)}</li>" for b in bullets) or "<li>Сигналов нет.</li>"

    top = f.nlargest(10, "ccu") if len(f) else f
    top_rows = "".join(
        f"<tr><td class='name'>{_esc(r['name'])}</td><td>{_esc(r.get('niche') or '—')}</td>"
        f"<td class='num'>{_fmt(r['ccu'])}</td>"
        f"<td>{int(r['yt_videos']) if r.get('yt_videos') and pd.notna(r['yt_videos']) else '—'}</td></tr>"
        for r in top.to_dict("records")
    )

    return f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roblox Pulse — {stamp:%d.%m %H:%M} UTC</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0b0e14; --panel:#141922; --panel2:#1a212c; --line:#232c3a;
  --txt:#e8eef7; --dim:#8b98ab; --up:#3fb950; --down:#f85149; --accent:#58a6ff;
}}
body{{background:var(--bg);color:var(--txt);line-height:1.5;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  padding:26px 16px 60px}}
.wrap{{max-width:1080px;margin:0 auto}}

header{{display:flex;justify-content:space-between;align-items:flex-start;
  flex-wrap:wrap;gap:14px;padding-bottom:20px;border-bottom:1px solid var(--line);margin-bottom:22px}}
h1{{font-size:26px;font-weight:680;letter-spacing:-.5px}}
.sub{{color:var(--dim);font-size:13px;margin-top:5px}}
.pulse{{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--up);margin-right:7px;animation:blink 2s infinite}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.35}}}}
.fresh{{font-size:12px;padding:6px 13px;border-radius:20px;white-space:nowrap}}
.fresh.ok{{background:rgba(63,185,80,.12);color:var(--up);border:1px solid rgba(63,185,80,.3)}}
.fresh.warn{{background:rgba(210,153,34,.12);color:#d29922;border:1px solid rgba(210,153,34,.3)}}
.fresh.bad{{background:rgba(248,81,73,.12);color:var(--down);border:1px solid rgba(248,81,73,.3)}}

.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:12px;margin-bottom:24px}}
.kpi{{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 17px}}
.kpi .lab{{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.6px}}
.kpi .val{{font-size:23px;font-weight:660;margin-top:5px;letter-spacing:-.5px}}
.kpi .sub2{{font-size:12px;color:var(--dim);margin-top:3px}}

.notice{{background:rgba(88,166,255,.08);border:1px solid rgba(88,166,255,.25);
  border-radius:10px;padding:12px 16px;font-size:13px;color:#a9c8f0;margin-bottom:22px}}

h2{{font-size:16px;font-weight:640;margin:28px 0 13px;display:flex;align-items:center;gap:9px}}
h2 .cnt{{font-size:12px;color:var(--dim);font-weight:500;
  background:var(--panel2);padding:2px 9px;border-radius:11px}}

.cards{{display:grid;gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:15px 17px;border-left:3px solid var(--line)}}
.card.up{{border-left-color:var(--up)}}
.card.down{{border-left-color:var(--down)}}
.card-head{{display:flex;align-items:flex-start;gap:11px}}
.rank{{font-size:15px;line-height:1.3}}
.card.up .rank{{color:var(--up)}}
.card.down .rank{{color:var(--down)}}
.title{{flex:1;min-width:0}}
.title h3{{font-size:15px;font-weight:600;letter-spacing:-.2px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.meta{{color:var(--dim);font-size:11.5px;margin-top:3px;display:flex;
  gap:6px;flex-wrap:wrap;align-items:center}}
.niche{{background:var(--panel2);padding:1px 7px;border-radius:5px;color:#9fb0c7}}
.dot{{opacity:.4}}
.delta{{text-align:right;flex-shrink:0}}
.delta .big{{font-size:19px;font-weight:680;letter-spacing:-.5px}}
.delta.up .big{{color:var(--up)}}
.delta.down .big{{color:var(--down)}}
.delta .small{{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px}}

.card-body{{display:flex;justify-content:space-between;align-items:center;
  gap:14px;margin-top:13px;flex-wrap:wrap}}
.numbers{{display:flex;gap:19px}}
.numbers .k{{display:block;font-size:10px;color:var(--dim);
  text-transform:uppercase;letter-spacing:.5px}}
.numbers .v{{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums}}
.numbers .v.pos{{color:var(--up)}}
.numbers .v.neg{{color:var(--down)}}
.spark{{flex-shrink:0}}

.why{{font-size:12.5px;color:#b6c4d6;margin-top:11px;padding-top:11px;
  border-top:1px solid var(--line)}}
.badges{{display:flex;gap:7px;flex-wrap:wrap;margin-top:9px}}
.badge{{font-size:11px;padding:3px 9px;border-radius:6px;white-space:nowrap}}
.badge.yt{{background:rgba(248,81,73,.1);color:#ff8b84;border:1px solid rgba(248,81,73,.22)}}
.badge.dim{{background:var(--panel2);color:var(--dim)}}
.badge.vid{{background:var(--panel2);color:#9fb0c7;max-width:100%;
  overflow:hidden;text-overflow:ellipsis}}
.badge.conf-hi{{background:rgba(63,185,80,.1);color:#6fdc7f;border:1px solid rgba(63,185,80,.22)}}
.badge.conf-mid{{background:rgba(210,153,34,.1);color:#d29922;border:1px solid rgba(210,153,34,.22)}}
.badge.conf-low{{background:var(--panel2);color:var(--dim)}}
.badge.conv-hi{{background:rgba(88,166,255,.13);color:#7fb3f0;border:1px solid rgba(88,166,255,.28)}}
.badge.conv-mid{{background:var(--panel2);color:#9fb0c7}}
.badge.conv-low{{background:var(--panel2);color:var(--dim)}}

table{{width:100%;border-collapse:collapse;font-size:12.5px;
  background:var(--panel);border-radius:11px;overflow:hidden;
  border:1px solid var(--line)}}
th{{text-align:left;padding:10px 13px;color:var(--dim);font-weight:550;
  font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;
  border-bottom:1px solid var(--line);background:var(--panel2)}}
td{{padding:9px 13px;border-bottom:1px solid rgba(35,44,58,.55)}}
tr:last-child td{{border-bottom:0}}
td.name{{font-weight:550}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.neg{{color:var(--down)}}
td.vt{{color:var(--dim);font-size:11.5px;max-width:280px;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}}
.empty{{color:var(--dim);text-align:center;padding:22px}}

.lead{{background:linear-gradient(180deg,rgba(88,166,255,.10),rgba(88,166,255,.03));
  border:1px solid rgba(88,166,255,.28);border-radius:12px;padding:15px 18px;margin-bottom:6px}}
.lead-h{{font-size:11px;text-transform:uppercase;letter-spacing:.7px;
  color:#7fb3f0;font-weight:600;margin-bottom:9px}}
.lead-list{{list-style:none;font-size:13.5px;line-height:1.75}}
.lead-list li{{padding-left:17px;position:relative}}
.lead-list li:before{{content:"→";position:absolute;left:0;color:#58a6ff;opacity:.75}}

.sect-note{{color:var(--dim);font-size:12px;margin:-6px 0 11px;max-width:760px;line-height:1.6}}

.in{{padding:2px 8px;border-radius:6px;font-size:11.5px;font-weight:600}}
.in.hot{{background:rgba(248,81,73,.14);color:#ff8b84}}
.in.soon{{background:rgba(210,153,34,.14);color:#d29922}}
.in.far{{background:var(--panel2);color:var(--dim)}}
.niches{{font-size:11.5px;color:var(--dim)}}
.nt{{background:var(--panel2);padding:1px 6px;border-radius:5px;color:#9fb0c7}}
.kind{{font-size:10.5px;padding:2px 7px;border-radius:5px}}
.kind.beta{{background:rgba(210,153,34,.14);color:#d29922}}
.kind.feature{{background:rgba(63,185,80,.12);color:#6fdc7f}}
.kind.update{{background:var(--panel2);color:var(--dim)}}
td.sc{{font-weight:640;color:#9ecbff}}

footer{{margin-top:34px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--dim);font-size:11.5px;line-height:1.7}}
footer b{{color:#a9b8cc;font-weight:560}}
</style></head><body><div class="wrap">

<header>
  <div>
    <h1><span class="pulse"></span>Roblox Pulse</h1>
    <div class="sub">Кто набирает обороты, кто теряет · обновлено {stamp:%d.%m.%Y %H:%M} UTC</div>
  </div>
  <div class="fresh {fresh_cls}">{fresh_txt}</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="lab">Игр в наблюдении</div>
    <div class="val">{len(f):,}</div><div class="sub2">в каруселях Roblox</div></div>
  <div class="kpi"><div class="lab">Суммарный онлайн</div>
    <div class="val">{total_ccu:,}</div><div class="sub2">прямо сейчас</div></div>
  <div class="kpi"><div class="lab">Рынок за период</div>
    <div class="val">{_pct(market_now/market_first if market_first else None)}</div>
    <div class="sub2">медиана игрока {_fmt(market_first)} → {_fmt(market_now)}</div></div>
  <div class="kpi"><div class="lab">С роликами на YouTube</div>
    <div class="val">{yt_games}</div><div class="sub2">из {len(f)} игр</div></div>
</div>

{thin}

<div class="lead">
  <div class="lead-h">Что повлияет в ближайшие недели</div>
  <ul class="lead-list">{bullets_html}</ul>
</div>

<h2>Растут быстрее рынка <span class="cnt">{len(rising)}</span></h2>
<div class="cards">{rising_html}</div>

<h2>Теряют позиции <span class="cnt">{len(falling)}</span></h2>
<table>
  <tr><th>Игра</th><th>Ниша</th><th>Онлайн</th><th>Доля</th><th>Изм.</th><th>YT</th></tr>
  {fall_rows}
</table>

<h2>Активность на YouTube <span class="cnt">{len(yt_rows)} игр с роликами</span></h2>
<table>
  <tr><th>Игра</th><th>Ниша</th><th>Онлайн</th><th>Видео</th><th>Просмотры</th><th>Яз.</th><th>Топ-ролик</th></tr>
  {yt_html_rows}
</table>

<h2>Крупнейшие сейчас <span class="cnt">топ-10</span></h2>
<table>
  <tr><th>Игра</th><th>Ниша</th><th>Онлайн</th><th>YT</th></tr>
  {top_rows}
</table>

<h2>Аниме-премьеры <span class="cnt">{len(fc["anime"])}</span></h2>
<p class="sect-note">Волна клонов под популярный тайтл начинается за 1–4 недели
до первого эпизода, а не после. Колонка «игр в нишах» показывает, насколько
ниша уже занята: мало игр — больше места.</p>
<table>
  <tr><th>Тайтл</th><th>До премьеры</th><th>Формат</th><th>Интерес</th><th>Ниши Roblox</th><th>Игр</th><th>Сила</th></tr>
  {anime_rows}
</table>

<h2>Изменения Roblox <span class="cnt">{len(fc["changes"])}</span></h2>
<p class="sect-note">Новые возможности платформы удешевляют одни жанры и
устаревают другие. Beta-фича — повод присмотреться раньше конкурентов.</p>
<table>
  <tr><th>Изменение</th><th>Тип</th><th>Когда</th></tr>
  {change_rows}
</table>

<h2>Растёт в Steam <span class="cnt">{len(fc["steam"])}</span></h2>
<p class="sect-note">Проверенная формула, которая ещё не перенесена в Roblox.
Высокая «скопируемость» — простые механики, которые реально повторить.</p>
<table>
  <tr><th>Игра</th><th>Пик онлайн</th><th>Жанры</th><th>Скоп.</th></tr>
  {steam_rows}
</table>

<footer>
  <b>Как считается рост.</b> Онлайн в Roblox подчиняется суточному циклу: медиана
  по всем играм падает с ~3 750 в 20:00 UTC до ~2 340 в 08:00 UTC. Поэтому каждая
  игра сравнивается не с собой вчера, а с медианой рынка в тот же момент.
  Показанная «доля рынка» — отношение этих двух величин: рост доли означает, что
  игра забирает игроков у остальных, даже если абсолютный онлайн просел вместе
  со всем рынком.<br><br>
  <b>Порог.</b> В списки попадают игры с онлайном от {MIN_CCU} и изменением доли
  более ±{int(MOVE_THRESHOLD*100)}%.<br><br>
  <b>YouTube.</b> Сборщик следит за 34 каналами, поэтому ролики есть только у
  части игр — пустая строка значит «не найдено», а не «не существует».
  Ролики находятся по названию игры в заголовке или описании.
  Счётчик показывает охват и активность: много свежих видео с растущими
  просмотрами означает живое сообщество, а не разовую вспышку.<br><br>
  <b>Чего здесь нет.</b> Базовой частоты: сколько похожих игр вышло и провалилось.
  Сбор видит только карусели Roblox, поэтому знаменатель неполный — воспринимай
  ranking как приоритет для проверки, а не как гарантию.
</footer>

</div></body></html>"""


def main() -> None:
    f = build_features()
    if f.empty:
        print("нет данных")
        return
    f = flag_suspect(f)
    f = f[~f["suspect_bot"]]

    piv, med = market_series()
    moves = relative_moves(piv, med)
    yt = yt_evidence()
    rows = build_rows(f, moves, yt)

    fc = forecast.build()
    page = render(f, piv, med, rows, yt, fc)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"-> {path}  ({len(page)/1024:.1f} KB)")
    print(f"   rising={len(rows[rows['trend'] >= MOVE_THRESHOLD]) if len(rows) else 0} "
          f"falling={len(rows[rows['trend'] <= -MOVE_THRESHOLD]) if len(rows) else 0} "
          f"(before r2 filter)")
    if len(rows):
        print("   top movers:")
        for r in rows.head(5).itertuples():
            print(f"     {r.name[:38]:40s} {_pct(r.share_change):>7s}  "
                  f"{_fmt(r.ccu_first)} → {_fmt(r.ccu_last)}  yt={r.yt_videos}")


if __name__ == "__main__":
    main()
