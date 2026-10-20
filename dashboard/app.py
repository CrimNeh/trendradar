"""Streamlit dashboard. Run: streamlit run dashboard/app.py"""
from __future__ import annotations

import os
import sys
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from features import build_features, flag_suspect  # noqa: E402
from link import _load_games, _load_signals, origin_brief  # noqa: E402
from origins import trends as origin_trends  # noqa: E402
from templates import build_families, family_summary  # noqa: E402

st.set_page_config(page_title="TrendRadar", page_icon="📡", layout="wide")


@st.cache_data(ttl=300)
def load():
    f = build_features()
    if f.empty:
        return f, pd.DataFrame(), {"niches": pd.DataFrame(),
                                   "ips": pd.DataFrame(), "top": pd.DataFrame()}
    f = flag_suspect(f)
    summ = family_summary(build_families(f))
    brief = origin_brief(_load_signals(), _load_games())
    return f, summ, brief


st.title("📡 TrendRadar — Roblox")


def _freshness() -> None:
    """Show when data was last written, in UTC, with age in minutes.

    Neon's table view sorts by id, so a stale table can look fresh. This line
    answers "is it collecting?" directly, without opening the database console.
    """
    try:
        from datetime import datetime, timezone
        from sqlalchemy import func
        from db import Session as _S, Snapshot as _Snap
        with _S() as _s:
            last = _s.query(func.max(_Snap.ts)).scalar()
    except Exception:
        return
    if last is None:
        st.warning("В базе пока нет ни одного снимка.")
        return
    age_min = (datetime.now(timezone.utc).replace(tzinfo=None) - last).total_seconds() / 60
    if age_min < 30:
        st.success(f"Последнее обновление: {last:%H:%M} UTC ({age_min:.0f} мин назад)")
    elif age_min < 120:
        st.warning(f"Данные отстают: последнее обновление {last:%H:%M} UTC "
                   f"({age_min:.0f} мин назад). GitHub задерживает запуски.")
    else:
        st.error(f"Сбор остановлен: последнее обновление {last:%d.%m %H:%M} UTC "
                 f"({age_min/60:.1f} ч назад). Проверь Actions.")


_freshness()
f, summ, brief = load()

if f.empty:
    st.warning("Нет данных. Запусти `python src/collect.py` дважды с интервалом 15+ мин.")
    st.stop()

yt_count = int((f["yt_videos"].fillna(0) > 0).sum())
multi = int((f["yt_langs"].fillna(0) >= 2).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Игр", f"{len(f):,}")
c2.metric("CCU", f"{int(f['ccu'].sum()):,}")
c3.metric("Шаблонов", f"{len(summ):,}")
c4.metric("YouTube-упоминаний", yt_count)
c5.metric("В 2+ языках", multi)

tabs = st.tabs(["🚀 Растущие", "🧬 Шаблоны", "📺 YouTube",
                "🌍 Внешние сигналы", "📊 Ниши"])

with tabs[0]:
    st.caption("Скор = момент + качество + давление YouTube + реклама.")
    min_conf = st.slider("Мин. confidence", 0.0, 1.0, 0.0, 0.05)
    d = f[f["confidence"] >= min_conf].copy()
    if st.checkbox("Скрыть подозрительные (боты)", value=True):
        d = d[~d["suspect_bot"]]
    cols = [c for c in ["name", "niche", "ccu", "g_h6", "g_d1", "yt_videos",
                        "yt_langs", "sponsored", "score", "cohort_rank",
                        "confidence", "age_days"] if c in d.columns]
    st.dataframe(
        d[cols].head(60), width="stretch", height=540,
        column_config={
            "name": st.column_config.TextColumn("Игра", width="large"),
            "ccu": st.column_config.NumberColumn("CCU", format="%d"),
            "g_h6": st.column_config.NumberColumn("×6ч", format="%.2f"),
            "g_d1": st.column_config.NumberColumn("×1д", format="%.2f"),
            "yt_videos": st.column_config.NumberColumn("YT", format="%d"),
            "yt_langs": st.column_config.NumberColumn("Яз", format="%d"),
            "sponsored": st.column_config.CheckboxColumn("Реклама"),
            "score": st.column_config.ProgressColumn("Скор", min_value=0, max_value=100),
            "cohort_rank": st.column_config.NumberColumn("Когорта %", format="%.0f"),
        },
    )

with tabs[1]:
    st.caption("Молодой шаблон с несколькими участниками = спрос подтверждён, "
               "окно входа открыто.")
    if summ.empty:
        st.info("Появятся после накопления данных.")
    else:
        st.dataframe(
            summ[["family", "games", "total_ccu", "mean_age",
                  "enter_score", "members"]].head(40),
            width="stretch", height=520,
            column_config={
                "family": st.column_config.TextColumn("Шаблон", width="medium"),
                "games": st.column_config.NumberColumn("Игр", format="%d"),
                "total_ccu": st.column_config.NumberColumn("CCU", format="%d"),
                "mean_age": st.column_config.NumberColumn("Возраст, дн", format="%.0f"),
                "enter_score": st.column_config.ProgressColumn(
                    "Вход", min_value=0, max_value=100),
                "members": st.column_config.TextColumn("Примеры", width="large"),
            },
        )

with tabs[2]:
    if yt_count == 0:
        from db import Session as _S, YtChannel as _C, YtVideo as _V
        with _S() as _s:
            n_ch = _s.query(_C).count()
            n_vid = _s.query(_V).count()

        # Say exactly what is missing. "run --seed" was shown even after the
        # user ran it, which is how this confusion started: seeding adds
        # channels, polling downloads videos.
        if n_ch == 0:
            st.info("Каналы ещё не добавлены. Запусти:\n\n"
                    "`python src/youtube_collect.py --seed`")
        elif n_vid == 0:
            st.info(f"Каналов в базе: {n_ch}. Заголовки их видео ещё не прочитаны. "
                    "Запусти:\n\n`python src/youtube_collect.py`")
        else:
            st.info(f"Заголовков прочитано: {n_vid}, но ни один не привязался к игре. "
                    "Запусти `python src/youtube_collect.py` ещё раз — "
                    "привязка идёт по названию игры в заголовке и по ссылкам "
                    "в описании.")
    else:
        st.caption("Скорость = просмотры в час. Языки = сколько рынков подхватили. "
                   "Хранятся только название, дата и просмотры — сами видео "
                   "не скачиваются.")
        yt = f[f["yt_videos"].fillna(0) > 0].sort_values("yt_velocity", ascending=False)
        st.dataframe(
            yt[["name", "yt_videos", "yt_views", "yt_velocity", "yt_langs",
                "yt_non_en", "ccu", "score"]],
            width="stretch", height=400,
            column_config={
                "name": st.column_config.TextColumn("Игра", width="large"),
                "yt_videos": st.column_config.NumberColumn("Видео", format="%d"),
                "yt_views": st.column_config.NumberColumn("Просмотры", format="%d"),
                "yt_velocity": st.column_config.NumberColumn("×час", format="%.0f"),
                "yt_langs": st.column_config.NumberColumn("Языков", format="%d"),
                "yt_non_en": st.column_config.NumberColumn("Не-en", format="%d"),
            },
        )
        from db import Session, YtVideo
        with Session() as s:
            raw = s.query(YtVideo.title, YtVideo.views, YtVideo.views_per_hour,
                          YtVideo.published).order_by(
                YtVideo.views_per_hour.desc()).limit(50).all()
        if raw:
            st.subheader("Сырые видео")
            st.dataframe(pd.DataFrame(raw, columns=["Заголовок", "Просмотры",
                                                   "×час", "Опубликовано"]),
                         width="stretch", height=360)

with tabs[3]:
    st.caption("Что приходит в Roblox снаружи: Steam (импорт механик), "
               "аниме-календарь (IP-волна), DevForum (фичи платформы).")
    if not brief["niches"].empty:
        st.subheader("Давление на ниши")
        st.caption("Внешнее внимание / число игр в Roblox. Высокое окно = "
                   "спрос есть, конкурентов мало.")
        st.dataframe(
            brief["niches"].head(20), width="stretch", height=320,
            column_config={
                "ext_pressure": st.column_config.NumberColumn("Давление", format="%.0f"),
                "window": st.column_config.ProgressColumn(
                    "Окно", min_value=0,
                    max_value=float(brief["niches"]["window"].max() or 1)),
            },
        )

    if not brief["ips"].empty:
        st.subheader("IP-волна: аниме → Roblox")
        st.caption("Аниме, чьё имя уже есть в названиях игр. Премьера впереди — "
                   "волна клонов ещё не накрыла платформу.")
        st.dataframe(brief["ips"], width="stretch", height=260)

    tr = [t for t in origin_trends(7) if t["delta"] > 0][:15]
    st.subheader("Что растёт")
    if tr:
        st.dataframe(pd.DataFrame(tr), width="stretch", height=300)
    else:
        st.info("Динамика появится со второго дня сбора.")

    if not brief["top"].empty:
        st.subheader("Топ внешних сигналов")
        for src in ("steam", "anime", "devforum"):
            sub = brief["top"][brief["top"]["source"] == src]
            if sub.empty:
                continue
            st.markdown(f"**{src}**")
            st.dataframe(sub[["title", "score", "tag"]].head(8),
                         width="stretch", height=200)

with tabs[4]:
    g = f.groupby("niche").agg(
        games=("universe_id", "count"), ccu=("ccu", "sum"),
        med_age=("age_days", "median"), med_score=("score", "median"),
    ).sort_values("ccu", ascending=False).reset_index()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(g.head(15), x="niche", y="ccu", title="CCU по нишам"),
                        width="stretch")
    with right:
        st.plotly_chart(px.bar(g.head(15), x="niche", y="games", title="Число игр"),
                        width="stretch")
    st.dataframe(g, width="stretch")
