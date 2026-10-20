"""Clone targets: which popular game to copy, and whether the copy can win.

    python src/clone_targets.py

The strategy this serves is fast-follow: take a formula that is already
earning, rebuild it with a different skin, and take share from the incumbent.
That changes the question completely. opportunity.py asks "where is the market
empty" - useless here, because an empty niche proves nobody wants the product.
This module asks:

    is the demand real, is it repeated, and is the leader weak enough to steal
    players from?

Four gates, all about the incumbent rather than about the gap:

  PRIZE          the target is big enough that a slice of it pays for the work
  PROOF          other games on the same formula are also earning - one hit may
                 be luck, five hits means the formula itself works
  VULNERABILITY  the field is not a single monopoly; the leader can be dented
  STEALABILITY   the audience is not locked in by age, IP or session depth

The last one matters most and is what a naive ranking misses: a 4-year-old game
with 400k CCU is usually a worse target than a 6-month-old one with 40k,
because the old one has a moat that copying cannot cross.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, flag_suspect  # noqa: E402
from link import _load_signals  # noqa: E402
from templates import tokens  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "reports")

# Below this, a slice of the audience cannot fund development at all.
MIN_TARGET_CCU = 3_000
# A peer game counts as proof of the formula only above this.
PROOF_CCU = 1_500

# Mechanics that make a game cheap to rebuild. If the target leans on these,
# a solo dev can realistically ship a version in weeks, not years.
CHEAP_MECHANICS = {
    "obby", "obby", "clicker", "simulator", "tycoon", "idle", "rebirth",
    "steal", "escape", "horror", "puzzle", "parkour", "collect", "trade",
    "farming", "mining", "grind", "progression", "pets", "battle_pass",
}

# Mechanics that mean the incumbent's advantage is production budget, which a
# copy cannot match by definition.
EXPENSIVE_MECHANICS = {
    "pvp", "competitive", "esports", "open_world", "rpg", "story", "cinematic",
    "physics", "ragdoll", "gunplay", "vehicles", "customization",
}

# Names attached to a licensed property. Copying the *formula* is ordinary
# competition; copying the *characters* is not, and a takedown kills the game
# after you have paid for it. Flagged, not forbidden - the call is yours.
IP_MARKERS = {
    "blue lock", "jujutsu", "naruto", "dragon ball", "one piece", "demon slayer",
    "solo leveling", "attack on titan", "my hero", "chainsaw", "spy family",
    "pokemon", "sonic", "mario", "lethal company", "fnaf", "squid game",
    "stranger things", "marvel", "dc ", "batman", "spider", "harry potter",
    "star wars", "minecraft", "among us", "poppy playtime", "garten of banban",
    "skibidi", "backrooms", "huggy", "springtrap", "godzilla", "kaiju",
}


def _fmt(n, digits: int = 0) -> str:
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    return f"{n:,.{digits}f}" if digits else f"{int(round(n)):,}"


def _tagset(raw) -> set[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    return {t.strip() for t in str(raw).split(",") if t.strip()}


def _ip_flag(name: str, ip_tokens: set[str]) -> str:
    """Detect a licensed property in the title.

    Only explicit markers count. A first version also matched any 6+ letter
    word shared with an anime title, which flagged "update", "server" and
    "valley" as IP risk - 39 of 208 games, almost all wrong. A false IP warning
    is worse than none: it hides the real ones.
    """
    low = (name or "").lower()
    for marker in IP_MARKERS:
        if marker in low:
            return marker.strip()
    return ""


def _ip_tokens_from_signals() -> set[str]:
    """Distinctive words from anime/Steam titles, used as IP fingerprints."""
    try:
        sig = _load_signals()
    except Exception:
        return set()
    if sig is None or sig.empty:
        return set()
    out: set[str] = set()
    for title in sig["title"].dropna().astype(str):
        for t in tokens(title):
            if len(t) >= 6:
                out.add(t.lower())
    return out


def build_targets(f: pd.DataFrame) -> pd.DataFrame:
    """Score every game as something worth copying."""
    if f.empty:
        return pd.DataFrame()

    ip_tokens = _ip_tokens_from_signals()
    rows = []

    for r in f.itertuples():
        if r.ccu < MIN_TARGET_CCU:
            continue

        tags = _tagset(getattr(r, "tags", None))
        niche = r.niche if isinstance(r.niche, str) else "other"

        # --- peers: same niche, sharing at least one mechanic
        same = f[f["niche"] == niche]
        # Match on shared mechanics when both sides have them. With no tags -
        # 187 of 452 games - fall back to the niche, otherwise a game with an
        # empty tag string compares against nobody and scores on its own.
        if tags:
            def shares(raw) -> bool:
                other = _tagset(raw)
                return bool(other & tags) if other else True
            same = same[same["tags"].map(shares)]
        peers = same[(same["universe_id"] != r.universe_id)
                     & (same["ccu"] >= PROOF_CCU)]

        proof = int(len(peers))
        peer_ccu = float(peers["ccu"].sum()) if len(peers) else 0.0
        field_ccu = float(r.ccu) + peer_ccu

        # --- vulnerability: is the field one monopoly or a real market?
        field_vals = [float(r.ccu)]
        if len(peers):
            field_vals.extend(float(x) for x in peers["ccu"].tolist())
        field_max = max(field_vals)
        field_med = float(np.median(field_vals))
        concentration = field_max / max(field_med, 1)

        # --- stealability
        age = r.age_days if r.age_days is not None else -1
        mech = set(tags) | {niche}
        cheap = len(mech & CHEAP_MECHANICS)
        heavy = len(mech & EXPENSIVE_MECHANICS)
        ip = _ip_flag(r.name, ip_tokens)

        # A falling leader is easier to attack than a rising one; a rising one
        # is easier to copy profitably but harder to overtake.
        g1 = getattr(r, "g_d1", None)
        trend = float(g1) if g1 is not None and not pd.isna(g1) else np.nan

        rows.append({
            "name": r.name,
            "universe_id": r.universe_id,
            "niche": niche,
            "ccu": float(r.ccu),
            "age_days": age,
            "score": getattr(r, "score", np.nan),
            "proof": proof,
            "field_ccu": field_ccu,
            "concentration": concentration,
            "cheap": cheap,
            "heavy": heavy,
            "ip": ip,
            "trend": trend,
            "tags": ", ".join(sorted(tags)[:5]),
            "yt_videos": int(getattr(r, "yt_videos", 0) or 0)
            if pd.notna(getattr(r, "yt_videos", None)) else 0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    def clone_score(x) -> tuple[float, list[str]]:
        notes: list[str] = []

        # PRIZE - 30. log-scaled: 400k is not 100x better than 4k to clone,
        # because you capture a fraction either way.
        prize = np.clip((np.log10(max(x.ccu, 1)) - 3.4) / 2.2, 0, 1) * 30

        # PROOF - 25. Each additional earning peer raises confidence that the
        # formula, not the specific execution, is what works.
        proof = np.clip(x.proof / 5.0, 0, 1) * 25
        if x.proof == 0:
            notes.append("формула не подтверждена копиями")

        # VULNERABILITY - 25. 1x = perfectly even field, 30x+ = monopoly.
        vuln = np.clip(1 - np.log10(max(x.concentration, 1)) / np.log10(30), 0, 1) * 25
        if x.concentration > 20:
            notes.append(f"лидер держит рынок ({x.concentration:.0f}× медианы)")

        # STEALABILITY - 20. Cheap to rebuild, young enough to still be fluid.
        ease = np.clip(x.cheap / 3.0, 0, 1) * 10
        if x.heavy >= 2:
            ease *= 0.4
            notes.append("дорогая механика: копия не догонит по бюджету")
        if x.cheap == 0:
            notes.append("механики простые не опознаны")

        if x.age_days is None or x.age_days < 0:
            youth = 0.5
        elif x.age_days <= 120:
            youth = 1.0
        elif x.age_days <= 400:
            youth = 0.75
        elif x.age_days <= 900:
            youth = 0.45
        else:
            youth = 0.2
            notes.append(f"старожил ({x.age_days} дн) — аудитория лояльна")
        ease += youth * 10

        if x.ip:
            notes.append(f"IP-риск: {x.ip}")

        return round(prize + proof + vuln + ease, 1), notes

    scored = df.apply(clone_score, axis=1)
    df["clone_score"] = [s for s, _ in scored]
    df["notes"] = [n for _, n in scored]
    return df.sort_values("clone_score", ascending=False).reset_index(drop=True)


def render(f: pd.DataFrame, t: pd.DataFrame) -> str:
    L: list[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    hist_h = float(f["history_hours"].max()) if len(f) else 0.0

    L.append(f"# Кого копировать — {stamp}\n")
    if hist_h < 30:
        L.append(f"> ⚠️ Истории {hist_h:.0f} ч. Ранжирование ниже опирается на размер "
                 f"аудитории, повторяемость формулы и возраст лидера — эти три "
                 f"величины от глубины истории не зависят. Динамика (падает/растёт) "
                 f"появится через сутки.\n")

    L.append("## Цели по порядку\n")
    if t.empty:
        L.append("_Нет игр выше порога._\n")
        return "\n".join(L)

    for i, r in enumerate(t.head(12).itertuples(), 1):
        L.append(f"### {i}. {r.name} — {r.clone_score:.0f}/100\n")
        L.append(f"**Онлайн {_fmt(r.ccu)}** · возраст {_fmt(r.age_days)} дн · "
                 f"ниша `{r.niche}`\n")
        L.append(f"- Та же формула работает ещё у **{r.proof}** игр, "
                 f"суммарно по полю **{_fmt(r.field_ccu)}** CCU")
        L.append(f"- Концентрация: лидер в {r.concentration:.0f}× от медианы поля")
        if r.ip:
            L.append(f"- ⚠️ **IP-риск: {r.ip}** — копировать механику можно, "
                     f"персонажей и названия нельзя")
        if r.notes:
            L.append(f"- Замечания: {'; '.join(r.notes)}")
        L.append("")

    L.append("## Таблица\n")
    L.append("| Игра | CCU | Возраст | Копий | Поле CCU | Конц. | IP | Скор |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in t.head(25).itertuples():
        L.append(f"| {r.name[:40]} | {_fmt(r.ccu)} | {_fmt(r.age_days)} | "
                 f"{r.proof} | {_fmt(r.field_ccu)} | {r.concentration:.0f}× | "
                 f"{r.ip or '—'} | {r.clone_score:.0f} |")
    L.append("")

    L.append(_methodology())
    return "\n".join(L)


def _methodology() -> str:
    return (
        "## Как считается\n\n"
        "Цель проходит по четырём условиям, и все они про конкурента, а не про нишу:\n\n"
        "1. **Приз (30)** — онлайн не ниже 3 000. Ниже доля с копии не окупит работу.\n"
        "2. **Подтверждение (25)** — сколько других игр на той же формуле тоже "
        "зарабатывают (от 1 500 CCU). Один хит — везение; пять хитов — работает "
        "сама формула.\n"
        "3. **Уязвимость (25)** — лидер не должен держать рынок один. Если он "
        "в 20 раз больше медианы поля, копия не пробьётся.\n"
        "4. **Скопируемость (20)** — простые механики и молодой возраст. Четырёхлетняя "
        "игра с лояльной аудиторией — плохая цель, даже при огромном онлайне.\n\n"
        "**Про IP.** Копировать механику, темп, структуру прогресса — обычная "
        "конкуренция, так работает вся индустрия. Копировать персонажей, названия "
        "и ассеты из чужой игры или лицензионного тайтла — нарушение, и жалоба "
        "правообладателя снимает игру после того, как ты за неё заплатил. "
        "Столбец IP отмечает такие цели, решение остаётся за тобой.\n"
    )


def main() -> None:
    f = build_features()
    if f.empty:
        print("нет данных")
        return
    f = flag_suspect(f)
    f = f[~f["suspect_bot"]]

    t = build_targets(f)
    md = render(f, t)

    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = os.path.join(OUT, f"{stamp}_clones.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
