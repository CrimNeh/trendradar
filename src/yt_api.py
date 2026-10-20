"""YouTube Data API v3 client with hard quota accounting.

    from yt_api import api
    api.search("roblox new game")     # 100 units
    api.videos(["id1", "id2"])        # 1 unit per 50 ids

WHY THIS EXISTS AND WHY IT IS NOT THE MAIN COLLECTOR

The free quota is 10,000 units per day. Costs that matter:

    search.list          100 units per call
    videos.list            1 unit per 50 videos
    channels.list          1 unit per 50 channels
    playlistItems.list     1 unit per call

Polling 150 channels with search.list would cost 15,000 units - more than the
entire daily quota for a single pass. That is why the RSS collector stays the
primary transport: it is free, and 150 feeds in 16 threads take about 4 seconds.

The API is used where RSS cannot reach:

  DISCOVERY  search for new games by keyword. RSS only shows channels we already
             know, so without search we never find new creators. Two searches a
             day cost 200 units - 2% of quota.
  STATS      exact view counts for a batch of known video ids, 50 at a time.
             RSS gives views but rounds nothing; API is authoritative.

Every call is charged against a row in api_quota before the request is made, so
a runaway loop stops at the limit instead of being banned for the rest of the
day. The limit is read from YT_DAILY_QUOTA and defaults to 10,000.
"""
from __future__ import annotations

import os
import sys
from datetime import date

import requests

sys.path.insert(0, os.path.dirname(__file__))
from db import ApiQuota, Session  # noqa: E402

BASE = "https://www.googleapis.com/youtube/v3"

# Official costs. Kept as constants so call sites cannot drift.
COST = {
    "search": 100,
    "videos": 1,
    "channels": 1,
    "playlistItems": 1,
    "captions": 50,
}

DAILY_LIMIT = int(os.environ.get("YT_DAILY_QUOTA", "10000"))
# Leave headroom: another tool on the same key may share the quota.
SAFETY = int(os.environ.get("YT_QUOTA_SAFETY", "900"))


class QuotaExceeded(RuntimeError):
    pass


class YouTubeAPI:
    def __init__(self, key: str | None = None):
        # Accept both names: the import script and .env.example were written
        # against YOUTUBE_API_KEY, the workflow wires YT_API_KEY. Silently
        # reading the wrong one leaves the API "unavailable" with a key present.
        self.key = (key
                    or os.environ.get("YT_API_KEY", "").strip()
                    or os.environ.get("YOUTUBE_API_KEY", "").strip())

    @property
    def available(self) -> bool:
        return bool(self.key)

    # ---------------------------------------------------------------- quota
    def _spend(self, units: int) -> None:
        """Reserve quota before the call. Raises rather than overspending."""
        if not self.available:
            raise RuntimeError("YT_API_KEY is not set")
        today = date.today()
        with Session() as s:
            row = s.get(ApiQuota, today)
            if row is None:
                row = ApiQuota(day=today, used=0)
                s.add(row)
                s.flush()
            if row.used + units > DAILY_LIMIT - SAFETY:
                raise QuotaExceeded(
                    f"quota: {row.used} used + {units} needed > "
                    f"{DAILY_LIMIT - SAFETY} allowed")
            row.used += units
            s.commit()

    def quota_used(self) -> int:
        with Session() as s:
            row = s.get(ApiQuota, date.today())
            return row.used if row else 0

    def quota_left(self) -> int:
        return max(0, DAILY_LIMIT - SAFETY - self.quota_used())

    # ---------------------------------------------------------------- calls
    def _get(self, path: str, params: dict) -> dict:
        params = {**params, "key": self.key}
        r = requests.get(f"{BASE}/{path}", params=params, timeout=25)
        if r.status_code == 403:
            body = r.text[:200]
            raise QuotaExceeded(f"403 from YouTube: {body}")
        r.raise_for_status()
        return r.json()

    def search(self, query: str, max_results: int = 25,
               order: str = "date", published_after: str | None = None) -> list[dict]:
        """Keyword search. 100 units - the single most expensive call here.

        `order=date` surfaces brand-new uploads, which is what a clone hunter
        wants: a video about a game that does not exist yet on Roblox is the
        earliest possible signal.
        """
        self._spend(COST["search"])
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(max_results, 50),
            "order": order,
            "relevanceLanguage": "en",
        }
        if published_after:
            params["publishedAfter"] = published_after
        d = self._get("search", params)

        out = []
        for it in d.get("items", []):
            sn = it.get("snippet", {})
            out.append({
                "video_id": it.get("id", {}).get("videoId"),
                "title": sn.get("title", ""),
                "channel_id": sn.get("channelId"),
                "channel_name": sn.get("channelTitle", ""),
                "published": sn.get("publishedAt", ""),
                "description": (sn.get("description") or "")[:2000],
            })
        return out

    def videos(self, video_ids: list[str]) -> list[dict]:
        """Exact statistics for up to 50 ids per unit. Batched automatically."""
        if not video_ids:
            return []
        out = []
        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i:i + 50]
            self._spend(COST["videos"])
            d = self._get("videos", {
                "part": "statistics,snippet",
                "id": ",".join(chunk),
                "maxResults": 50,
            })
            for it in d.get("items", []):
                st = it.get("statistics", {})
                sn = it.get("snippet", {})
                out.append({
                    "video_id": it.get("id"),
                    "title": sn.get("title", ""),
                    "channel_id": sn.get("channelId"),
                    "published": sn.get("publishedAt", ""),
                    "views": int(st.get("viewCount", 0) or 0),
                    "likes": int(st.get("likeCount", 0) or 0),
                    "comments": int(st.get("commentCount", 0) or 0),
                    "duration": it.get("contentDetails", {}).get("duration", ""),
                })
        return out

    def channels(self, channel_ids: list[str]) -> list[dict]:
        """Subscriber counts and uploads playlists, 50 per unit."""
        if not channel_ids:
            return []
        out = []
        for i in range(0, len(channel_ids), 50):
            chunk = channel_ids[i:i + 50]
            self._spend(COST["channels"])
            d = self._get("channels", {
                "part": "statistics,snippet,contentDetails",
                "id": ",".join(chunk),
                "maxResults": 50,
            })
            for it in d.get("items", []):
                st = it.get("statistics", {})
                out.append({
                    "channel_id": it.get("id"),
                    "title": it.get("snippet", {}).get("title", ""),
                    "subscribers": int(st.get("subscriberCount", 0) or 0),
                    "videos": int(st.get("videoCount", 0) or 0),
                    "views": int(st.get("viewCount", 0) or 0),
                    "uploads": it.get("contentDetails", {})
                                .get("relatedPlaylists", {}).get("uploads", ""),
                })
        return out

    def playlist_items(self, playlist_id: str, max_results: int = 20) -> list[str]:
        """Video ids from an uploads playlist. 1 unit - cheap alternative to search."""
        self._spend(COST["playlistItems"])
        d = self._get("playlistItems", {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": min(max_results, 50),
        })
        return [it["contentDetails"]["videoId"] for it in d.get("items", [])]

    # ------------------------------------------------- existing importer API
    def resolve_handles(self, handles: list[str]) -> dict[str, str]:
        """handle -> channel_id via channels.list?forHandle. 1 unit per 50.

        This is the cheap way to turn @handles into ids: one unit covers fifty
        channels, against 100 units for a single search.list. yt_import.py was
        written against this method, so it lives here rather than in a second
        importer module.
        """
        out: dict[str, str] = {}
        clean = [h.lstrip("@") for h in handles]
        for i in range(0, len(clean), 50):
            chunk = clean[i:i + 50]
            self._spend(COST["channels"])
            try:
                d = self._get("channels", {
                    "part": "id",
                    "forHandle": ",".join(chunk),
                    "maxResults": 50,
                })
            except Exception:
                # forHandle takes one value, not a list - fall back to single
                # lookups, which still beats scraping HTML.
                for h in chunk:
                    self._spend(COST["channels"])
                    try:
                        one = self._get("channels", {"part": "id", "forHandle": h})
                        for it in one.get("items", []):
                            out["@" + h] = it["id"]
                    except Exception:
                        continue
                continue
            for it in d.get("items", []):
                h = (it.get("snippet", {}).get("customUrl") or "").lstrip("@")
                if h:
                    out["@" + h] = it["id"]
        return out

    def spent(self) -> int:
        """Units used today, for the importer's summary line."""
        return self.quota_used()

api = YouTubeAPI()

if __name__ == "__main__":
    if not api.available:
        print("YT_API_KEY не задан — API недоступен, RSS продолжает работать")
        sys.exit(0)
    print(f"квота использована : {api.quota_used()}")
    print(f"осталось           : {api.quota_left()}")
