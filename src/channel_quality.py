"""Score channels by how much usable signal they actually produce.

    python src/channel_quality.py            # report only
    python src/channel_quality.py --apply    # deactivate the dead weight

The top-150 list is not a Roblox list. It is a "big channels that sometimes
touch Roblox" list, and the difference matters: measured on the live database,
119 of 177 channels had never produced a single matched video. Some of them are
music channels, some are Minecraft animators, some post titles like
"DYNAMITE666" and "video0035" with no game name in them at all.

Polling them costs a slot in every rotation and returns nothing.

WHY NOT JUST "ZERO HITS = DELETE"

A real Roblox channel can legitimately show zero hits for weeks: its titles are
vague ("ESTE JUEGO DE ROBLOX ES GOD"), the game is not in our tracked set yet,
or it is new and we only have eight videos from it. Deleting on hits alone
throws away exactly the channels that will produce later.

So a channel is judged on two independent signals:

  SIGNAL     does it show any evidence of being about Roblox at all -
             the word in a title, or a single matched game
  VOLUME     do we have enough videos to judge it (10 minimum)

Only a channel with volume but no signal is pruned. That is a stricter test
than hits alone and it survives the vague-title case, because those channels
still say "roblox" somewhere.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(__file__))
from db import Session, YtChannel, utcnow  # noqa: E402

# Below this many collected videos there is not enough evidence either way.
MIN_VIDEOS = 10
# A channel that produced a match at any point in this window is kept outright.
GRACE_DAYS = 60

# "Roblox" as actually written across markets. The first version of this check
# looked for the Latin string only and pruned @SvarkaShow - a Russian channel
# with 168 videos titled "Я СЫГРАЛ В РОБЛОКС". Cyrillic, Japanese and Korean
# spellings are all common, and a title can also say "robux" without the brand.
ROBLOX_MARKERS = [
    "roblox", "ролокс", "роблокс", "рблокс", "роблкс",
    "ロブロックス", "로블록스", "ロブレックス",
    "robux", "робукс", "روبوكس", "روبلوكس",
    "bloxburg", "brookhaven",
]


def score_channels() -> list[dict]:
    """One row per active channel with its verdict and the reason for it."""
    now = utcnow()
    grace_cut = now - timedelta(days=GRACE_DAYS)

    # strpos, not LIKE. A literal '%' inside text() is interpreted by psycopg2's
    # pyformat layer and silently produced zero matches - the Russian channels
    # stayed in the prune list despite 168 videos saying "РОБЛОКС".
    # Parentheses are mandatory. Without them SQL precedence turns
    #   channel_id = ch.channel_id AND a OR b OR c
    # into (channel_id AND a) OR b OR c, so b and c match videos from every
    # channel - it reported 360 videos for a channel that has 18.
    marker_sql = "(" + " or ".join(
        f"strpos(lower(v.title), '{m}') > 0" for m in ROBLOX_MARKERS) + ")"

    with Session() as s:
        rows = s.execute(text(f"""
            select
              ch.channel_id,
              ch.handle,
              ch.lang,
              ch.subscribers,
              (select count(distinct v.video_id) from yt_videos v
                 where v.channel_id = ch.channel_id)              as videos,
              (select count(distinct m.video_id) from mentions m
                 where m.channel_id = ch.channel_id)              as hits,
              (select count(distinct v.video_id) from yt_videos v
                 where v.channel_id = ch.channel_id
                   and {marker_sql})                              as roblox_titles,
              (select max(m.observed) from mentions m
                 where m.channel_id = ch.channel_id)              as last_hit
            from yt_channels ch
            where ch.active = 1
        """)).fetchall()

    out = []
    for r in rows:
        videos = int(r.videos or 0)
        hits = int(r.hits or 0)
        roblox_titles = int(r.roblox_titles or 0)
        last_hit = r.last_hit
        handle = (r.handle or "").lower()

        # The handle is evidence too. @traakszroblox is a real Roblox channel
        # whose Portuguese titles never name the brand ("COMPREI UM OVO DE
        # HALLOWEEN" is Steal an Egg). Judging it on titles alone pruned it.
        handle_signal = 1 if any(k in handle for k in
                                 ("roblox", "rbx", "blox", "robux")) else 0

        signal = hits + roblox_titles + handle_signal

        if hits > 0 and last_hit and last_hit >= grace_cut:
            verdict, reason = "keep", f"{hits} совпадений, последнее {last_hit:%d.%m}"
        elif videos < MIN_VIDEOS:
            verdict, reason = "watch", f"мало данных ({videos} видео)"
        elif signal == 0:
            verdict, reason = "prune", f"{videos} видео, 0 сигналов Roblox"
        elif hits == 0 and roblox_titles >= 3:
            # Says "roblox" often but never names a tracked game. Keep: it may
            # cover games we have not added yet.
            verdict, reason = "keep", f"{roblox_titles} видео про Roblox, игр не названо"
        elif handle_signal and roblox_titles == 0:
            verdict, reason = "keep", "Roblox в названии канала"
        else:
            verdict, reason = "keep", f"{signal} сигналов"

        out.append({
            "channel_id": r.channel_id,
            "handle": r.handle or r.channel_id,
            "lang": r.lang or "?",
            "subscribers": int(r.subscribers or 0),
            "videos": videos,
            "hits": hits,
            "roblox_titles": roblox_titles,
            "signal": signal,
            "verdict": verdict,
            "reason": reason,
        })

    out.sort(key=lambda x: (x["verdict"] != "prune", -x["videos"]))
    return out


def apply_prune(rows: list[dict]) -> int:
    """Deactivate the channels marked prune. Returns how many changed."""
    ids = [r["channel_id"] for r in rows if r["verdict"] == "prune"]
    if not ids:
        return 0
    with Session() as s:
        for cid in ids:
            ch = s.get(YtChannel, cid)
            if ch is not None:
                ch.active = 0
        s.commit()
    return len(ids)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="deactivate channels marked prune")
    p.add_argument("--show", type=int, default=20)
    a = p.parse_args()

    rows = score_channels()
    if not rows:
        print("нет каналов")
        return

    buckets = {"keep": [], "watch": [], "prune": []}
    for r in rows:
        buckets[r["verdict"]].append(r)

    print(f"активных каналов: {len(rows)}\n")
    for name, label in [("keep", "оставить"), ("watch", "наблюдать"),
                        ("prune", "отсеять")]:
        n = len(buckets[name])
        print(f"  {label:10s} {n:>4}")

    print(f"\n=== ОТСЕЯТЬ (топ-{a.show} по числу видео) ===")
    for r in buckets["prune"][:a.show]:
        print(f"  {r['handle'][:26]:28s} {r['lang']:3s} "
              f"{r['subscribers']:>10,}  {r['videos']:>3} видео  {r['reason']}")

    print(f"\n=== ОСТАВИТЬ (топ-{min(a.show, 10)}) ===")
    for r in buckets["keep"][:10]:
        print(f"  {r['handle'][:26]:28s} {r['lang']:3s} {r['reason']}")

    if a.apply:
        n = apply_prune(rows)
        print(f"\nдеактивировано: {n}")
    else:
        print(f"\nсухой прогон. Применить: --apply")


if __name__ == "__main__":
    main()
