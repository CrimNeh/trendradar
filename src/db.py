"""Storage. SQLite locally, Postgres in the cloud (same code).

Tables
------
games        : one row per universe, slowly changing attributes
snapshots    : time series of CCU / visits / favorites / chart membership
channel_hits : which carousels a game appears in, per snapshot
yt_channels  : tracked YouTube creators (multilingual)
yt_videos    : videos seen, with view counts at observation time
mentions     : video -> game edges produced by the matcher
origin_signal: external signals that precede Roblox trends
               (steam, anime calendar, platform features)
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone

from sqlalchemy import (
    BigInteger, Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint,
    create_engine, Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite only auto-increments an INTEGER PRIMARY KEY, so use a portable type.
PKType = BigInteger().with_variant(Integer, "sqlite")

def _normalise_url(url: str) -> str:
    """Pin the Postgres driver.

    SQLAlchemy 2.1 resolves a bare `postgresql://` to psycopg (v3), which is
    not in requirements. We ship psycopg2, so say so explicitly - otherwise a
    Neon URL from the dashboard fails with "No module named 'psycopg'".
    """
    if url.startswith("postgres://"):          # Heroku-style alias
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]
    return url


DB_URL = _normalise_url(os.environ.get(
    "TRENDRADAR_DB",
    "sqlite:///" + os.path.join(os.path.dirname(__file__), "..", "data", "trendradar.db"),
))

_engine_kwargs: dict = {"future": True}
if DB_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Neon (and other serverless Postgres) close idle connections. Without
    # pre_ping the first query of each run hits a dead socket; without recycle
    # a long run can hold a connection past the server's idle timeout.
    _engine_kwargs.update(pool_pre_ping=True, pool_recycle=300, pool_size=5,
                          max_overflow=5)

engine = create_engine(DB_URL, **_engine_kwargs)
Session = sessionmaker(bind=engine, future=True)
Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Game(Base):
    __tablename__ = "games"

    universe_id = Column(BigInteger, primary_key=True)
    root_place_id = Column(BigInteger)
    name = Column(String(256))
    description = Column(Text)
    creator_id = Column(BigInteger)
    creator_name = Column(String(128))
    creator_type = Column(String(16))
    max_players = Column(Integer)
    created = Column(DateTime)
    updated = Column(DateTime)
    icon_url = Column(String(512))
    niche = Column(String(64), index=True)
    tags = Column(String(256))
    first_seen = Column(DateTime, default=utcnow)
    last_seen = Column(DateTime, default=utcnow)


class Snapshot(Base):
    __tablename__ = "snapshots"

    id = Column(PKType, primary_key=True, autoincrement=True)
    universe_id = Column(BigInteger, index=True, nullable=False)
    ts = Column(DateTime, default=utcnow, index=True, nullable=False)
    playing = Column(Integer)
    visits = Column(BigInteger)
    favorites = Column(BigInteger)
    up_votes = Column(Integer)
    down_votes = Column(Integer)
    max_players = Column(Integer)
    servers = Column(Integer)
    avg_fill = Column(Float)
    chart_count = Column(Integer, default=0)
    best_chart_rank = Column(Integer)
    is_sponsored = Column(Integer, default=0)

    __table_args__ = (Index("ix_snap_game_ts", "universe_id", "ts"),)


class ChannelHit(Base):
    __tablename__ = "channel_hits"

    id = Column(PKType, primary_key=True, autoincrement=True)
    universe_id = Column(BigInteger, index=True, nullable=False)
    ts = Column(DateTime, default=utcnow, index=True, nullable=False)
    sort_id = Column(String(48), index=True)
    rank = Column(Integer)

    __table_args__ = (
        UniqueConstraint("universe_id", "ts", "sort_id", name="uq_channel_hit"),
    )


class YtChannel(Base):
    __tablename__ = "yt_channels"

    channel_id = Column(String(32), primary_key=True)
    handle = Column(String(64))
    name = Column(String(128))
    lang = Column(String(8), index=True)     # en, es, pt, id, ar, ru, ...
    subscribers = Column(BigInteger)
    active = Column(Integer, default=1)
    # Consecutive polls that returned an empty feed. Used to retire dead channels.
    empty_polls = Column(Integer, default=0)
    last_checked = Column(DateTime)
    added = Column(DateTime, default=utcnow)


class YtVideo(Base):
    """One row per video. Views only grow, so the row is refreshed in place."""
    __tablename__ = "yt_videos"

    id = Column(PKType, primary_key=True, autoincrement=True)
    video_id = Column(String(16), index=True, unique=True, nullable=False)
    channel_id = Column(String(32), index=True)
    title = Column(String(512))
    published = Column(DateTime, index=True)
    observed = Column(DateTime, default=utcnow, index=True)
    views = Column(BigInteger)
    views_per_hour = Column(Float)


class Mention(Base):
    """A video that references a tracked game."""
    __tablename__ = "mentions"

    id = Column(PKType, primary_key=True, autoincrement=True)
    video_id = Column(String(16), index=True, nullable=False)
    universe_id = Column(BigInteger, index=True, nullable=False)
    observed = Column(DateTime, default=utcnow, index=True)
    containment = Column(Float)
    matched = Column(String(128))
    channel_id = Column(String(32))
    lang = Column(String(8))
    method = Column(String(12))      # link | token | embed
    score = Column(Float)

    __table_args__ = (
        UniqueConstraint("video_id", "universe_id", name="uq_mention"),
    )


class OriginSignal(Base):
    """External leading indicators. See origins.py for producers.

    source    : 'steam' | 'anime' | 'devforum' | 'platform'
    key       : stable id within the source (appid, anime id, topic id)
    title     : human-readale label
    score     : normalised 0-100 interest
    meta      : free-form JSON string
    """
    __tablename__ = "origin_signal"

    id = Column(PKType, primary_key=True, autoincrement=True)
    source = Column(String(24), index=True, nullable=False)
    key = Column(String(64), index=True, nullable=False)
    title = Column(String(256))
    score = Column(Float)
    meta = Column(Text)
    tag = Column(String(64), index=True)   # genre / mechanic tag for matching
    observed = Column(DateTime, default=utcnow, index=True)

    __table_args__ = (
        UniqueConstraint("source", "key", name="uq_origin_key"),
    )


# Columns added after the first release. create_all() never alters existing
# tables, so a database created by an older version needs them added in place.
_MIGRATIONS = {
    "mentions": [("method", "VARCHAR(12)"), ("score", "FLOAT")],
    "yt_channels": [
        ("lang", "VARCHAR(8)"),
        # Consecutive empty polls, so a channel that never posts is retired
        # instead of being fetched every run forever.
        ("empty_polls", "INTEGER DEFAULT 0"),
    ],
}


class OriginHistory(Base):
    """Append-only time series of external signals.

    `origin_signal` holds only the latest state, so it can say "this Steam
    game is hot today" but never "it has climbed five days in a row". This
    table keeps one point per (source, key, day) so trends are computable.
    """
    __tablename__ = "origin_history"

    id = Column(PKType, primary_key=True, autoincrement=True)
    source = Column(String(24), index=True, nullable=False)
    key = Column(String(64), index=True, nullable=False)
    day = Column(String(10), index=True, nullable=False)      # YYYY-MM-DD
    score = Column(Float)
    rank = Column(Integer)
    extra = Column(Float)        # steam: peak players, anime: popularity

    __table_args__ = (
        UniqueConstraint("source", "key", "day", name="uq_origin_hist"),
    )


def _migrate() -> None:
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    with engine.begin() as con:
        for table, cols in _MIGRATIONS.items():
            if not insp.has_table(table):
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols:
                if name not in have:
                    con.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def _ensure_dir() -> None:
    """Create the SQLite directory before connecting.

    A fresh clone has no data/ (it is gitignored), and SQLite refuses to
    create a database in a directory that does not exist. Without this the
    very first run on a new machine dies with "unable to open database file".
    """
    if DB_URL.startswith("sqlite"):
        path = DB_URL.replace("sqlite:///", "")
        if path and path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


class ApiQuota(Base):
    """Daily YouTube Data API spend, so a loop cannot silently exceed the limit.

    The free quota is 10,000 units/day and is shared by every tool using the
    same key. Charging before the request - rather than after - means a crash
    mid-call still counts, which errs toward under-spending. That is the right
    direction: an over-spent key returns 403 for the rest of the day and there
    is no way to get the time back.
    """
    __tablename__ = "api_quota"

    day = Column(Date, primary_key=True)
    used = Column(Integer, default=0, nullable=False)


def init_db() -> None:
    _ensure_dir()
    Base.metadata.create_all(engine)
    _migrate()


class Prediction(Base):
    """One forecast, frozen at the moment it was made.

    APPEND-ONLY. There is no update path and no delete path anywhere in the
    codebase, by design. The value of this table comes entirely from the fact
    that a bad forecast cannot be quietly tidied up afterwards - the moment
    that becomes possible, hit rate stops measuring the signal and starts
    measuring how carefully the record was massaged.

    `outcome` is the only column that ever changes, and only from NULL to a
    value, once. `outcome_basis` records how the verdict was reached so a later
    reader can disagree without rerunning anything.
    """
    __tablename__ = "predictions"

    id = Column(PKType, primary_key=True, autoincrement=True)

    kind = Column(String(24), index=True, nullable=False)   # move|clone|anime
    universe_id = Column(BigInteger, index=True)
    subject = Column(String(256), nullable=False)           # name as written then
    claim = Column(String(512), nullable=False)             # plain-language forecast

    # Judging is fixed at creation. If the criterion could be chosen after the
    # result is known, the target would drift toward whatever happened.
    metric = Column(String(32), nullable=False)             # ccu|favorites
    baseline = Column(Float, nullable=False)                # value when predicted
    # Market median at forecast time. Needed to grade a relative claim: "grew
    # faster than the market" cannot be checked without knowing what the market
    # did over the same window, and that history may be pruned later.
    benchmark = Column(Float)
    criterion = Column(String(24), nullable=False)          # up_20pct|beat_market_p75
    # The concrete bar in percentage points, resolved once at creation.
    #
    # A criterion naming a percentile ("beat_market_p75") is not self-contained:
    # the pp value it corresponds to depends on the data at the time. Resolving
    # it at judgement time would let the bar move as the market changes, which
    # is the one thing a fixed criterion exists to prevent. So the number is
    # frozen here, next to the forecast it grades.
    threshold_pp = Column(Float)
    horizon_days = Column(Integer, nullable=False)

    # The score that produced the forecast, so accuracy can be tied back to the
    # ranking that recommended it.
    score = Column(Float)
    confidence = Column(Float)
    cohort_rank = Column(Float)

    made_at = Column(DateTime, default=utcnow, index=True, nullable=False)
    due_at = Column(DateTime, index=True, nullable=False)

    outcome = Column(String(16), index=True)                # hit|miss|partial|void
    outcome_value = Column(Float)
    outcome_basis = Column(String(256))
    checked_at = Column(DateTime)

    __table_args__ = (
        Index("ix_pred_due_outcome", "due_at", "outcome"),
    )

if __name__ == "__main__":
    path = DB_URL.replace("sqlite:///", "")
    if path and not DB_URL.startswith("postgres"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    init_db()
    print("initialised", DB_URL)
