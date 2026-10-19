"""Markdown digest across all four signal families.

    python src/report.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, flag_suspect  # noqa: E402
from link import _load_games, _load_signals, origin_brief  # noqa: E402
from origins import trends as origin_trends  # noqa: E402
from templates import build_families, family_summary  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports")


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.{digits}f}"


def main() -> str:
    f = build_features()
    if f.empty:
        print("no data")
        return ""

    f = flag_suspect(f)
    summ = family_summary(build_families(f))
    sig, games = _load_signals(), _load_games()
    brief = origin_brief(sig, games)

    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(OUT, f"{stamp}.md")

    f.to_csv(os.path.join(OUT, f"{stamp}_games.csv"), index=False)
    if not summ.empty:
        summ.to_csv(os.path.join(OUT, f"{stamp}_templates.csv"), index=False)
    if not brief["niches"].empty:
        brief["niches"].to_csv(os.path.join(OUT, f"{stamp}_niches.csv"), index=False)

    yt_hits = f[f["yt_videos"].fillna(0) > 0]
    multi = f[f["yt_langs"].fillna(0) >= 2]

    L: list[str] = []
    L.append(f"# TrendRadar — {stamp}\n")
    L.append(f"Игр: **{len(f)}** · CCU: **{int(f['ccu'].sum()):,}** · "
             f"шаблонов: **{len(summ)}** · YouTube-упоминаний: **{len(yt_hits)}** · "
             f"в 2+ языках: **{len(multi)}** · внешних сигналов: **{len(sig)}**\n")

    if f["g_d1"].notna().sum() < 5:
        L.append("\n> ⚠️ Истории роста мало. Рейтинг опирается на `lifetime_ratio`. "
                 "Дельты появятся через 1–2 дня сбора.\n")

    # ---- origins: what is coming from outside Roblox
    if not brief["niches"].empty:
        L.append("\n## Внешнее давление на ниши\n")
        L.append("Сколько внимания ниша получает на Steam и в аниме-календаре, "
                 "делённое на число игр в Roblox. Высокое окно = спрос есть, "
                 "конкурентов мало.\n")
        L.append("| Ниша | Давление | Сигналов | Игр в Roblox | Окно |")
        L.append("|---|---|---|---|---|")
        for r in brief["niches"].head(12).itertuples():
            L.append(f"| {r.niche} | {fmt(r.ext_pressure,0)} | {r.ext_count} | "
                     f"{r.roblox_games} | **{r.window}** |")

    if not brief["ips"].empty:
        L.append("\n## IP-волна: аниме → Roblox\n")
        L.append("Аниме, чьё имя уже встречается в названиях игр. Премьера "
                 "впереди — значит волна клонов ещё не накрыла платформу.\n")
        L.append("| IP | Игра | Общее слово | До премьеры |")
        L.append("|---|---|---|---|")
        for r in brief["ips"].head(10).itertuples():
            d = "—" if r.days_to_premiere is None or r.days_to_premiere == 999 \
                else f"{int(r.days_to_premiere)} дн"
            L.append(f"| {r.ip[:44]} | {r.game[:30]} | {r.shared} | {d} |")

    if not brief["top"].empty:
        steam = brief["top"][brief["top"]["source"] == "steam"].head(6)
        if not steam.empty:
            L.append("\n### Steam: что могут склонировать\n")
            for r in steam.itertuples():
                L.append(f"- **{r.title[:56]}** — интерес {fmt(r.score,0)} ({r.tag[:30]})")

    # ---- external signals that are actually climbing (needs 2+ days of data)
    climbing = [t for t in origin_trends(7) if t["delta"] > 0][:8]
    if climbing:
        L.append("\n### Внешние сигналы, которые растут\n")
        L.append("| Источник | Сигнал | Было | Стало | Δ | Дней данных |")
        L.append("|---|---|---|---|---|---|")
        for t in climbing:
            L.append(f"| {t['source']} | {t['title'][:44]} | {t['from']} | "
                     f"{t['to']} | **+{t['delta']}** | {t['days']} |")
    else:
        L.append("\n> Динамика внешних сигналов появится со второго дня сбора.\n")

    # ---- templates
    if not summ.empty:
        L.append("\n## Шаблоны: куда входить\n")
        L.append("| Шаблон | Игр | CCU | Ср. возраст | Вход | Примеры |")
        L.append("|---|---|---|---|---|---|")
        for r in summ.head(12).itertuples():
            L.append(f"| **{r.family}** | {r.games} | {int(r.total_ccu):,} | "
                     f"{fmt(r.mean_age,0)} дн | {r.enter_score} | {r.members[:80]} |")

    # ---- youtube
    if not yt_hits.empty:
        L.append("\n## YouTube опережает Roblox\n")
        L.append("| Игра | Видео | Просмотры | ×час | Языков | Не-en | CCU | Скор |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in yt_hits.sort_values("yt_velocity", ascending=False).head(12).itertuples():
            L.append(f"| {r.name[:28]} | {int(r.yt_videos)} | {int(r.yt_views):,} | "
                     f"{fmt(r.yt_velocity,0)} | {int(r.yt_langs or 0)} | "
                     f"{int(r.yt_non_en or 0)} | {int(r.ccu):,} | {r.score} |")

    # ---- top games
    L.append("\n## Топ игр по скору\n")
    L.append("| Игра | Ниша | CCU | ×1д | YT | Яз | Скор | Когорта | Conf | Реклама |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in f.head(20).itertuples():
        L.append(f"| {r.name[:30]} | {r.niche} | {int(r.ccu):,} | {fmt(r.g_d1)} | "
                 f"{fmt(r.yt_videos,0)} | {fmt(r.yt_langs,0)} | {r.score} | "
                 f"{fmt(r.cohort_rank,0)} | {r.confidence} | {'да' if r.sponsored else ''} |")

    g = f.groupby("niche").agg(games=("universe_id", "count"),
                               ccu=("ccu", "sum")).sort_values("ccu", ascending=False)
    L.append("\n## Ниши в Roblox\n")
    L.append("| Ниша | Игр | CCU |")
    L.append("|---|---|---|")
    for niche, r in g.head(12).iterrows():
        L.append(f"| {niche} | {int(r['games'])} | {int(r['ccu']):,} |")

    text = "\n".join(L) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"written {path}")
    return path


if __name__ == "__main__":
    main()
