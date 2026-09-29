"""SQLite state store for the publisher.

The publisher needs to remember what it has already pushed, so that re-running
the same command (or uploading "to every profile") never publishes the same clip
to the same profile/platform twice. A tiny SQLite database — one file in the
project root — holds that history plus the remote ids/urls of everything that
was published.

The database is intentionally boring: a single ``uploads`` table keyed by
``(profile, platform, video_sha1)``. The SHA-1 of the clip's *contents* is used
as the identity of a short, so the same file renamed in between still counts as
the same upload.

Nothing here launches a browser or touches the network, so this module is safe to
import and unit-test anywhere.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .log import log
from .model import Short, file_sha1

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    profile      TEXT    NOT NULL,
    platform     TEXT    NOT NULL,
    video_file   TEXT    NOT NULL,
    video_sha1   TEXT    NOT NULL,
    title        TEXT    DEFAULT '',
    remote_id    TEXT    DEFAULT '',
    remote_url   TEXT    DEFAULT '',
    visibility   TEXT    DEFAULT '',
    status       TEXT    NOT NULL,
    error        TEXT    DEFAULT '',
    created_at   TEXT    NOT NULL,
    published_at TEXT    DEFAULT '',
    UNIQUE(profile, platform, video_sha1)
);

CREATE INDEX IF NOT EXISTS idx_uploads_target
    ON uploads (profile, platform);
CREATE INDEX IF NOT EXISTS idx_uploads_sha1
    ON uploads (video_sha1);
"""


class UploadStatus:
    """The values stored in the ``status`` column."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


def _now() -> str:
    """Current local time as an ISO-8601 string (second resolution)."""
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class UploadRecord:
    """One row of the ``uploads`` table."""

    profile: str
    platform: str
    video_file: str
    video_sha1: str
    title: str = ""
    remote_id: str = ""
    remote_url: str = ""
    visibility: str = ""
    status: str = UploadStatus.OK
    error: str = ""
    created_at: str = ""
    published_at: str = ""
    id: int = 0

    @property
    def ok(self) -> bool:
        return self.status == UploadStatus.OK

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "UploadRecord":
        return cls(
            id=row["id"],
            profile=row["profile"],
            platform=row["platform"],
            video_file=row["video_file"],
            video_sha1=row["video_sha1"],
            title=row["title"] or "",
            remote_id=row["remote_id"] or "",
            remote_url=row["remote_url"] or "",
            visibility=row["visibility"] or "",
            status=row["status"],
            error=row["error"] or "",
            created_at=row["created_at"] or "",
            published_at=row["published_at"] or "",
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class StateDB:
    """A thin wrapper around the SQLite uploads database.

    Use it as a context manager so the connection is always closed::

        with StateDB("publish_state.sqlite3") as db:
            if db.should_skip("profile_1", "youtube", short):
                continue
            ...
            db.record(profile="profile_1", platform="youtube", short=short)
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        # WAL keeps the file readable while a long upload is running and makes
        # the write pattern (frequent small inserts) cheap.
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - closing is best-effort
            pass

    def __enter__(self) -> "StateDB":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- queries -----------------------------------------------------------

    def get(
        self, profile: str, platform: str, video_sha1: str
    ) -> Optional[UploadRecord]:
        """Return the stored record for one target, or ``None``."""
        row = self._conn.execute(
            "SELECT * FROM uploads WHERE profile=? AND platform=? AND video_sha1=?",
            (profile, platform, video_sha1),
        ).fetchone()
        return UploadRecord.from_row(row) if row else None

    def should_skip(
        self,
        profile: str,
        platform: str,
        video_sha1: str,
        *,
        rerun: bool = False,
        rerun_failed: bool = False,
    ) -> Optional[str]:
        """Decide whether an upload can be skipped.

        Returns a human-readable reason when the upload should be skipped, or
        ``None`` when it should proceed.

        * ``rerun`` — ignore the whole history (re-upload everything);
        * ``rerun_failed`` — ignore previous *failures* (retry them) but still
          skip clips that were already published successfully.
        """
        if rerun:
            return None
        record = self.get(profile, platform, video_sha1)
        if record is None:
            return None
        if record.status == UploadStatus.OK:
            return f"already uploaded ({record.published_at or record.created_at})"
        if record.status == UploadStatus.FAILED and not rerun_failed:
            return f"previous attempt failed ({record.error or 'unknown error'})"
        if record.status == UploadStatus.SKIPPED and not rerun_failed:
            return "previously skipped"
        return None

    # -- mutation ----------------------------------------------------------

    def record(
        self,
        *,
        profile: str,
        platform: str,
        short: Short,
        status: str = UploadStatus.OK,
        remote_id: str = "",
        remote_url: str = "",
        visibility: str = "",
        error: str = "",
        video_sha1: Optional[str] = None,
    ) -> UploadRecord:
        """Insert or update the row for ``(profile, platform, short)``.

        Re-running the same clip to the same target overwrites the previous row
        (thanks to the unique constraint), so the table always reflects the
        latest attempt instead of accumulating duplicates.
        """
        sha1 = video_sha1 or file_sha1(short.file)
        created = _now()
        published = created if status == UploadStatus.OK else ""
        self._conn.execute(
            """
            INSERT INTO uploads (
                profile, platform, video_file, video_sha1, title,
                remote_id, remote_url, visibility, status, error,
                created_at, published_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile, platform, video_sha1) DO UPDATE SET
                video_file   = excluded.video_file,
                title        = excluded.title,
                remote_id    = excluded.remote_id,
                remote_url   = excluded.remote_url,
                visibility   = excluded.visibility,
                status       = excluded.status,
                error        = excluded.error,
                created_at   = excluded.created_at,
                published_at = excluded.published_at
            """,
            (
                profile,
                platform,
                short.file,
                sha1,
                short.title,
                remote_id,
                remote_url,
                visibility,
                status,
                error,
                created,
                published,
            ),
        )
        self._conn.commit()
        stored = self.get(profile, platform, sha1)
        assert stored is not None  # just written
        return stored

    # -- reporting ---------------------------------------------------------

    def counts(self) -> dict:
        """Return ``{(profile, platform): {status: count}}`` for a summary."""
        rows = self._conn.execute(
            "SELECT profile, platform, status, COUNT(*) AS n "
            "FROM uploads GROUP BY profile, platform, status"
        ).fetchall()
        summary: dict = {}
        for row in rows:
            key = (row["profile"], row["platform"])
            summary.setdefault(key, {})[row["status"]] = row["n"]
        return summary

    def history(self, limit: int = 50) -> List[UploadRecord]:
        """Most recent uploads first, capped at ``limit`` rows."""
        rows = self._conn.execute(
            "SELECT * FROM uploads ORDER BY id DESC LIMIT ?",
            (max(0, limit),),
        ).fetchall()
        return [UploadRecord.from_row(row) for row in rows]

    def total(self) -> int:
        """Total number of rows in the uploads table."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM uploads").fetchone()
        return int(row["n"]) if row else 0


__all__ = ["StateDB", "UploadRecord", "UploadStatus"]
