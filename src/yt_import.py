"""Load the curated top-150 Roblox channel list into the database.

    python src/yt_import.py --csv roblox_top150.csv
    python src/yt_import.py --csv roblox_top150.csv --resolve   # use API key

Resolution is mandatory, not optional. `channel_id` is the primary key of
yt_channels, so a row without one cannot be stored at all - the first version of
this script tried a "store handles only" mode and failed on the NOT NULL
constraint.

With YOUTUBE_API_KEY set, 150 handles cost 150 quota units (~3 s at 8 workers).
Without a key it falls back to HTML scraping: same result, ~45 s, and it is the
fragile path that gets throttled. The key is worth having just for this.

The CSV is trusted for metadata (subscribers, language, country) because it was
curated deliberately. Only the channel id has to be looked up.

    python src/yt_import.py --csv roblox_top150.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import yt_api
import youtube as yt
from db import Session, YtChannel, init_db, utcnow

# The list uses "?" for unknown language and mixed tags like "en/tl". Store the
# primary tag only, otherwise every query that filters by lang misses them.
LANG_ALIASES = {
    "ru/uk": "ru", "en/tl": "en", "en/in": "en", "pt/br": "pt",
    "zh/tw": "zh", "zh/cn": "zh", "es/es": "es", "es/latam": "es",
}


def _lang(raw: str) -> str:
    v = (raw or "").strip().lower()
    if not v or v == "?":
        return "en"
    return LANG_ALIASES.get(v, v[:2])


def _subs(raw) -> int:
    s = str(raw or "0").strip().upper()
    mult = 1
    if s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("K"):
        mult, s = 1_000, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return 0


def load_csv(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            handle = (r.get("handle") or "").strip()
            if not handle:
                continue
            rows.append({
                "handle": handle if handle.startswith("@") else "@" + handle,
                "name": (r.get("channel") or "").strip()[:128],
                "subs": _subs(r.get("subs")),
                "lang": _lang(r.get("lang")),
                "country": (r.get("country") or "").strip()[:4],
                "rank": int(r.get("rank") or 0),
            })
    return rows


def import_channels(rows: list[dict]) -> dict:
    init_db()
    stats = {"total": len(rows), "resolved": 0, "failed": 0,
             "new": 0, "updated": 0, "api_units": 0}

    # Resolve first, outside the DB session: 150 network calls must not hold a
    # transaction open, and a partial failure should still commit what worked.
    handles = [r["handle"] for r in rows]
    ids: dict[str, str] = {}
    if yt_api.available():
        print(f"resolving {len(handles)} handles via API...")
        ids = yt_api.resolve_handles(handles)
    else:
        print(f"no API key - resolving {len(handles)} handles via HTML (~45 s)")
        print("set YOUTUBE_API_KEY to make this 3 s instead")
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:
            for h, cid in zip(handles, ex.map(yt.resolve_channel, handles)):
                if cid:
                    ids[h] = cid
    stats["api_units"] = yt_api.spent()
    stats["resolved"] = len(ids)

    with Session() as s:
        for r in rows:
            cid = ids.get(r["handle"])
            if not cid:
                stats["failed"] += 1
                continue

            row = s.get(YtChannel, cid)
            if row is None:
                s.add(YtChannel(
                    channel_id=cid, handle=r["handle"], name=r["name"],
                    lang=r["lang"], subscribers=r["subs"], active=1,
                    added=utcnow(),
                ))
                stats["new"] += 1
            else:
                row.handle = row.handle or r["handle"]
                row.name = row.name or r["name"]
                row.lang = r["lang"]
                row.subscribers = r["subs"] or row.subscribers
                row.active = 1
                stats["updated"] += 1
        s.commit()

    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="path to roblox_top150.csv")
    a = p.parse_args()

    if not Path(a.csv).exists():
        print(f"not found: {a.csv}")
        return

    rows = load_csv(a.csv)
    print(f"loaded {len(rows)} channels from {a.csv}")
    langs: dict[str, int] = {}
    for r in rows:
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
    print("languages:", dict(sorted(langs.items(), key=lambda x: -x[1])))

    st = import_channels(rows)
    print()
    for k, v in st.items():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
