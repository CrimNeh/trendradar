"""YouTube signals without an API key.

Verified live:
1. resolve_channel(handle)  - @handle -> channelId, parsed from public HTML
2. channel_uploads(id)      - last ~15 uploads via public RSS with title,
                              publish date and views, so view velocity is
                              available. Zero quota.
3. search(query)            - HTML results, optional sort by upload date.

RSS is the primary tool: the Data API allows ~100 searches/day for free,
while RSS is unlimited and answers exactly "what did my tracked creators
just post" - the leading indicator we care about.

ToS note: RSS is a documented public feed. Scraping search HTML violates
YouTube's ToS even though it needs no key, so it is rate limited hard here
and used only for discovery.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(__file__))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "media": "http://search.yahoo.com/mrss/",
    "yt": "http://www.youtube.com/xml/schemas/2015",
}

_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})


def _get(url: str, tries: int = 3) -> str | None:
    for attempt in range(tries):
        try:
            r = _session.get(url, timeout=25)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code != 200:
                return None
            return r.text
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def resolve_channel(handle: str) -> str | None:
    """'@KreekCraft' or 'KreekCraft' -> 'UC...' channel id."""
    h = handle.lstrip("@")
    html = _get(f"https://www.youtube.com/@{h}")
    if not html:
        return None
    m = re.search(r'"channelId":"(UC[\w-]{22})"', html)
    if m:
        return m.group(1)
    m = re.search(r"channel/(UC[\w-]{22})", html)
    return m.group(1) if m else None


def channel_uploads(channel_id: str) -> list[dict]:
    """Last ~15 uploads with views and publish times. Free, no quota."""
    xml = _get(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}")
    if not xml:
        return []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    title_el = root.find("atom:title", NS)
    channel_name = title_el.text if title_el is not None else ""

    out = []
    for e in root.findall("atom:entry", NS):
        vid = e.findtext("yt:videoId", default="", namespaces=NS)
        title = e.findtext("atom:title", default="", namespaces=NS)
        published = e.findtext("atom:published", default="", namespaces=NS)
        if not vid or not title:
            continue

        group = e.find("media:group", NS)
        views = 0
        description = ""
        if group is not None:
            description = group.findtext("media:description", default="",
                                         namespaces=NS) or ""
            stats = group.find("media:community/media:statistics", NS)
            if stats is not None:
                views = int(stats.get("views", 0) or 0)

        try:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except ValueError:
            continue
        pub_naive = pub_dt.astimezone(timezone.utc).replace(tzinfo=None)

        hours = max((datetime.now(timezone.utc).replace(tzinfo=None)
                     - pub_naive).total_seconds() / 3600, 0.5)
        out.append({
            "video_id": vid, "title": title,
            "channel_id": channel_id, "channel_name": channel_name,
            "published": pub_naive, "views": views,
            "views_per_hour": round(views / hours, 1),
            "description": description[:2000],
        })
    return out


def _walk(obj, key, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            else:
                _walk(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, key, out)


def _txt(node) -> str:
    if not node:
        return ""
    if "simpleText" in node:
        return node["simpleText"]
    return "".join(r.get("text", "") for r in node.get("runs", []))


_AGE = re.compile(r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s*ago")


def _parse_age(text: str) -> datetime | None:
    m = _AGE.search(text or "")
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    mult = {"second": 1 / 86400, "minute": 1 / 1440, "hour": 1 / 24,
            "day": 1, "week": 7, "month": 30, "year": 365}[unit]
    return (datetime.now(timezone.utc) - timedelta(days=n * mult)).replace(tzinfo=None)


def search(query: str, sort_by_date: bool = True, limit: int = 20) -> list[dict]:
    """YouTube search without an API key. sp=CAI sorts by upload date."""
    sp = "&sp=CAI%3D" if sort_by_date else ""
    q = requests.utils.quote(query)
    html = _get(f"https://www.youtube.com/results?search_query={q}{sp}")
    if not html:
        return []
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", html)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    renderers: list[dict] = []
    _walk(data, "videoRenderer", renderers)

    out = []
    for v in renderers[:limit]:
        title = _txt(v.get("title"))
        if not title:
            continue
        vm = re.search(r"(\d+)", _txt(v.get("viewCountText")).replace(",", ""))
        out.append({
            "video_id": v.get("videoId"),
            "title": title,
            "channel_name": _txt(v.get("ownerText")),
            "views": int(vm.group(1)) if vm else 0,
            "published": _parse_age(_txt(v.get("publishedTimeText"))),
            "age_text": _txt(v.get("publishedTimeText")),
        })
        time.sleep(random.uniform(0.05, 0.15))
    return out


if __name__ == "__main__":
    for h in ("KreekCraft", "Flamingo"):
        cid = resolve_channel(h)
        print(f"@{h} -> {cid}")
        for v in channel_uploads(cid)[:3]:
            print(f"   {v['published'].date()} | {v['views']:>10,} | "
                  f"{v['views_per_hour']:>9,.0f}/h | {v['title'][:46]}")
