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
    # How this game entered the database.
    #
    #   carousel : seen in a Roblox discovery carousel. The original, and still
    #              the only, source of the market sample that the digest, the
    #              market median and the base rate are built from.
    #   orphan   : discovered another way (currently YouTube mentions) and never
    #              observed in a carousel. Before 2026-10-28 these were added to
    #              `games` and never snapshotted at all - 496 of 1200 rows (41%)
    #              had no CCU history, including titles with 500M+ visits. They
    #              are now snapshotted by collect.collect_orphans(), but stay
    #              OUT of the market median and base rate until the week-2
    #              comparison is done (digest_html.INCLUDE_ORPHANS), because
    #              adding them changes the sample the comparison is made on.
    #
    # A game that later appears in a carousel is promoted back to 'carousel' by
    # collect(), and its full history becomes eligible immediately.
    origin = Column(String(16), default="carousel", index=True)
    # Last time the orphan sweep asked Roblox for this game's live CCU. Without
    # it every run re-queries the same ~340 permanently-tiny games.
    orphan_checked = Column(DateTime)


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
    """Per-carousel rank of a game at one snapshot tick.

    NOT WRITTEN ANY MORE. The table grew to 71 517 rows at ~3 200/day and had
    zero readers: `features.py` derives everything it needs from
    `snapshots.chart_count` and `snapshots.best_chart_rank`, which are written
    in the same loop and carry the same information in aggregate. Nothing in
    src/, dashboard/, queries.sql or the workflow ever selected from this table.

    The class is kept so `init_db()` does not drop the existing rows on the
    next schema pass - historical data stays queryable if the per-carousel
    detail is ever wanted. Re-enable writing by restoring the loop in
    collect.py.
    """
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
    # When the semantic pass last LOOKED at this video - set whether or not it
    # matched anything.
    #
    # The queue used to be "videos with no mention", which is not a queue: a
    # video that matches nothing keeps having no mention, so it was re-embedded
    # on every run for as long as it stayed in the age window (about 7 wasted
    # passes each). Measured 2026-10-28: 19842 videos with no mention, queue
    # growth 1.06x throughput at a 7-day window - it could never drain, because
    # the same rows kept coming back.
    #
    # With this column the queue is "not yet looked at", so every video is
    # embedded once. Reset to NULL by _ingest_new_games() for recent videos
    # when a NEW game enters the catalogue, so a video processed before its
    # game existed still gets one more chance against it.
    embed_checked = Column(DateTime, index=True)


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
    # r2 (fit quality of the growth line) was computed every run and thrown
    # away - every retroactive check of "does r2 predict the outcome" had to
    # re-fetch raw snapshots and refit the regression by hand. Store it next
    # to the forecast it was computed for, so the question is answerable with
    # one query instead of a script.
    "predictions": [("r2", "FLOAT"), ("ruleset_version", "INTEGER DEFAULT 2")],
    # origin defaults to 'carousel' so every pre-existing row keeps meaning
    # exactly what it meant before: "came from a discovery carousel".
    # orphan_checked tracks the last sweep, so a permanently-tiny game is asked
    # about once a day instead of on all 8 runs.
    "games": [("origin", "VARCHAR(16) DEFAULT 'carousel'"), ("orphan_checked", "DATETIME")],
    "yt_videos": [("embed_checked", "DATETIME")],
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
    is_pg = engine.dialect.name == "postgresql"
    added: set[str] = set()
    with engine.begin() as con:
        for table, cols in _MIGRATIONS.items():
            if not insp.has_table(table):
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols:
                if name not in have:
                    # The DDL is written once per column and has to be valid on
                    # both engines this project supports: SQLite spells the
                    # timestamp type DATETIME, Postgres spells it TIMESTAMP.
                    # Passing the SQLite spelling to Postgres fails with
                    # `type "datetime" does not exist`.
                    col_type = ddl.replace("DATETIME", "TIMESTAMP") if is_pg else ddl
                    con.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}"))
                    added.add(f"{table}.{name}")

        if insp.has_table("yt_videos") and "embed_checked" in {
                c["name"] for c in insp.get_columns("yt_videos")}:
            # Already-matched videos are NOT queue work: pass 1/2 already found
            # their game, so a semantic pass over them cannot add anything. They
            # stayed NULL only because the stamp is written by pass 3, which
            # never sees them. Without this update roughly half the reported
            # backlog (7320 of 15800 on 2026-10-28) is work that cannot produce
            # a result, which makes the queue look twice as deep as it is and
            # starves the videos that do need matching.
            #
            # Idempotent: only touches rows that are both unstamped and already
            # matched. Matches the NOT EXISTS guard in the queue query, so the
            # stamp and the query can never disagree about what "pending" means.
            con.execute(text(
                "UPDATE yt_videos SET embed_checked = CURRENT_TIMESTAMP "
                "WHERE embed_checked IS NULL AND EXISTS "
                "  (SELECT 1 FROM mentions m WHERE m.video_id = yt_videos.video_id)"))
            # Backfill the already-stale rows as "checked", so the queue means
            # exactly "still pending" and not "pending forever".
            #
            # 17162 videos were older than the 7-day semantic window when this
            # column appeared. They can never be selected (the queue query
            # filters on published), so leaving them NULL would make every
            # count of the backlog permanently inflated by rows that no run will
            # ever consume - and a backlog that cannot be measured cannot be
            # shown to be draining.
            #
            # Only the genuinely stale ones: recent videos stay NULL so they
            # get their first real pass.
            # Idempotent by construction: only ever stamps rows that are both
            # NULL and already outside the window, so a second run finds
            # nothing to do. It runs on every migration rather than only on
            # first column creation, because the first attempt can miss (the
            # column may have been added by an earlier run before this
            # backfill existed - which is exactly what happened on 2026-10-28).
            con.execute(text(
                "UPDATE yt_videos SET embed_checked = CURRENT_TIMESTAMP "
                "WHERE embed_checked IS NULL AND published IS NOT NULL "
                "  AND published <= CURRENT_TIMESTAMP - INTERVAL '7 days'"))

        if "games.origin" in added:
            # Backfill, not just a default. The column default labels every
            # pre-existing row 'carousel', but ~496 of them are not: they were
            # discovered via YouTube and never once appeared in a carousel.
            # Left as 'carousel' they would slip into the market sample the
            # week-2 comparison depends on being unchanged.
            #
            # "Has no snapshots" identifies them exactly - collect() writes a
            # snapshot in the same transaction that creates a game from a
            # carousel, so a carousel game cannot lack history. Guarded on
            # first-add so a second migration pass can never reclassify a game
            # that is merely waiting for its first snapshot.
            con.execute(text(
                "UPDATE games SET origin = 'orphan' "
                "WHERE NOT EXISTS (SELECT 1 FROM snapshots s "
                "                  WHERE s.universe_id = games.universe_id)"))
            con.execute(text("UPDATE games SET origin = 'carousel' WHERE origin IS NULL"))

        if insp.has_table("games") and is_pg:
            con.execute(text(
                "ALTER TABLE games ALTER COLUMN origin SET DEFAULT 'carousel'"))
        if insp.has_table("predictions") and is_pg:
            _install_settled_guard(con)


def _install_settled_guard(con) -> None:
    """DB-level enforcement that a settled row's judging fields never change.

    The docstrings in this file promised "append-only, no update path, no
    delete path" since the table was introduced. That promise lived only in
    application code - nothing stopped a direct UPDATE run from outside the
    app. An external audit on 2026-10-28 found exactly that: 100 of the first
    124 rows were set to `outcome='void'` with a hand-written reason in
    `outcome_basis`, and at least one of them (id=11) has a `checked_at` only
    ~1h45m after `made_at` on a 7-day-horizon row - impossible through
    `resolve()`, which only ever judges rows whose `due_at` has passed.

    This trigger makes the guarantee real: once `outcome` is non-NULL, no
    UPDATE may change `outcome`, `outcome_value`, `outcome_basis`, `baseline`,
    `benchmark`, `criterion`, or `threshold_pp` on that row, for any writer,
    including the table owner (Postgres triggers fire regardless of who runs
    the statement - only disabling or dropping the trigger bypasses this,
    which requires DDL rights and leaves its own trace). `resolve()` never
    needs to touch a settled row twice (it already skips rows where
    `outcome is not None`), so this cannot break the normal path.
    """
    from sqlalchemy import text
    con.execute(text("""
        CREATE OR REPLACE FUNCTION predictions_block_settled_edit()
        RETURNS trigger AS $$
        BEGIN
            IF OLD.outcome IS NOT NULL AND (
                NEW.outcome IS DISTINCT FROM OLD.outcome OR
                NEW.outcome_value IS DISTINCT FROM OLD.outcome_value OR
                NEW.outcome_basis IS DISTINCT FROM OLD.outcome_basis OR
                NEW.baseline IS DISTINCT FROM OLD.baseline OR
                NEW.benchmark IS DISTINCT FROM OLD.benchmark OR
                NEW.criterion IS DISTINCT FROM OLD.criterion OR
                NEW.threshold_pp IS DISTINCT FROM OLD.threshold_pp
            ) THEN
                RAISE EXCEPTION
                    'predictions: row % is settled (outcome=%); judging fields are immutable once set',
                    OLD.id, OLD.outcome;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """))
    con.execute(text("""
        DROP TRIGGER IF EXISTS trg_predictions_block_settled_edit ON predictions;
    """))
    con.execute(text("""
        CREATE TRIGGER trg_predictions_block_settled_edit
        BEFORE UPDATE ON predictions
        FOR EACH ROW EXECUTE FUNCTION predictions_block_settled_edit();
    """))


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
    # Fit quality of the growth line the forecast was based on (0..1). Stored
    # separately from `score` (the slope itself) because the two measure
    # different things and the week-1 settlement showed they behave
    # differently: slope correlated NEGATIVELY with hitting (-0.36 Spearman,
    # n=24), r2 correlated weakly POSITIVELY (+0.17). Without this column that
    # finding required re-fetching raw snapshots and refitting the regression
    # for every past row by hand.
    r2 = Column(Float)
    # Which version of the selection/judging rules produced this row. Bumped
    # whenever PLANS, _judge, or the candidate ranking changes meaningfully.
    #
    # Before this existed, a rule change meant either mixing old and new rows
    # under one hit rate (wrong - they answer different questions) or voiding
    # the old ones by hand with a prose explanation in outcome_basis. The
    # second path is how 100 of the first 124 rows ended up void: real edits,
    # individually justified, but applied directly to "settled" rows outside
    # resolve() - the opposite of the append-only guarantee this table exists
    # to provide. An external audit on 2026-10-28 traced one of them (id=11)
    # to a checked_at only ~1h45m after made_at on a 7-day horizon, which
    # resolve() cannot produce - proof the edit bypassed the application.
    #
    # Going forward: bump this constant instead of writing void+prose. Old
    # rows stay queryable by version instead of being silently dropped from
    # the sample.
    ruleset_version = Column(Integer, default=2)

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
