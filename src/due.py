"""Run-when-due gate, replacing the `if UTC hour == 00 or 12` checks.

Why this exists
---------------
The workflow used to decide "should this step run?" by reading the UTC clock:
`if H == 00 or H == 12`. That only works if the trigger fires on those exact
hours. The external pinger (cron-job.org) fires every 3 hours in the owner's
local zone - for a UTC+5 user that is 19, 22, 01, 04 ... UTC - so the hours
00 and 12 never occur, and origins, YouTube discovery and the weekly audit
silently never ran (last origins run: 20 Oct 08:06 UTC).

A clock-hour check couples every step to whichever hours a trigger happens to
hit. This gate asks a different question: "how long since this job last
succeeded?" That answer does not depend on the timezone, on the trigger's
hours, or on how many triggers fired.

Usage (from a workflow step):
    python src/due.py origins 11      # run if >= 11 h since last success
    echo $?                           # 0 = due, 1 = not yet

On success the caller runs `python src/due.py mark origins` afterwards, so a
failed run does not reset the clock and gets retried on the next tick.
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import Column, DateTime, String, text  # noqa: E402

from db import Base, Session, engine, init_db, utcnow  # noqa: E402


class JobRun(Base):
    """Last successful completion per job. One row per job name."""
    __tablename__ = "job_runs"
    name = Column(String(64), primary_key=True)
    last_ok = Column(DateTime, nullable=False)


def hours_since(name: str) -> float | None:
    """Hours since the job last succeeded, or None if it never has."""
    init_db()
    with Session() as s:
        row = s.get(JobRun, name)
        if row is None:
            return None
        return (utcnow() - row.last_ok).total_seconds() / 3600


def is_due(name: str, every_hours: float) -> bool:
    age = hours_since(name)
    # Never run before -> due. Otherwise due once the interval has elapsed.
    return age is None or age >= every_hours


def mark(name: str) -> None:
    """Record a successful completion. Call only after the job succeeded."""
    init_db()
    with Session() as s:
        row = s.get(JobRun, name)
        now = utcnow()
        if row is None:
            s.add(JobRun(name=name, last_ok=now))
        else:
            row.last_ok = now
        s.commit()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "mark":
        mark(sys.argv[2])
        print(f"marked {sys.argv[2]}")
        sys.exit(0)

    if len(sys.argv) != 3:
        print("usage: due.py <job> <every_hours> | due.py mark <job>", file=sys.stderr)
        sys.exit(2)

    name, every = sys.argv[1], float(sys.argv[2])
    age = hours_since(name)
    if is_due(name, every):
        print(f"{name}: due (last ok {'never' if age is None else f'{age:.1f} h ago'})")
        sys.exit(0)
    print(f"{name}: not due yet ({age:.1f} h since last ok, need {every})")
    sys.exit(1)
