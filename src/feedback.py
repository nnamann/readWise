"""Local SQLite persistence for reader feedback (Like, Not interested, Already read).

Feedback is kept strictly separate from the original Book-Crossing dataset and
persists locally in a small SQLite database on the reader's machine.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from .config import FEEDBACK_DB_PATH

VALID_FEEDBACK_TYPES = {"like", "dislike", "read"}

FEEDBACK_LABELS = {
    "like": "Like this book",
    "dislike": "Not interested",
    "read": "Mark as already read",
}


def _resolve_db_path(db_path: Path | str | None = None) -> str:
    if db_path is None:
        db_path = FEEDBACK_DB_PATH
    if str(db_path) != ":memory:":
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return ":memory:"


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return an open SQLite connection with thread safety and WAL enabled."""
    resolved = _resolve_db_path(db_path)
    conn = sqlite3.connect(resolved, timeout=10.0)
    conn.row_factory = sqlite3.Row
    if resolved != ":memory:":
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            pass
    return conn


def init_db(db_path: Path | str | None = None) -> str:
    """Ensure the feedback table and indices exist."""
    resolved = _resolve_db_path(db_path)
    with get_connection(resolved) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS book_feedback (
                book_id TEXT PRIMARY KEY,
                feedback_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_book_feedback_type
            ON book_feedback (feedback_type);
            """
        )
        conn.commit()
    return resolved


def set_feedback(book_id: str, feedback_type: str, db_path: Path | str | None = None) -> None:
    """Save or update feedback for a stable book ID."""
    clean_id = str(book_id).strip()
    if not clean_id:
        raise ValueError("book_id cannot be empty")
    clean_type = str(feedback_type).strip().lower()
    if clean_type not in VALID_FEEDBACK_TYPES:
        raise ValueError(
            f"Invalid feedback type: {feedback_type!r}. Must be one of {sorted(VALID_FEEDBACK_TYPES)}"
        )
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO book_feedback (book_id, feedback_type, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(book_id) DO UPDATE SET
                feedback_type = excluded.feedback_type,
                updated_at = excluded.updated_at;
            """,
            (clean_id, clean_type, now, now),
        )
        conn.commit()


def remove_feedback(book_id: str, db_path: Path | str | None = None) -> bool:
    """Remove feedback for a book ID. Returns True if an entry was removed."""
    clean_id = str(book_id).strip()
    if not clean_id:
        return False
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM book_feedback WHERE book_id = ?", (clean_id,))
        conn.commit()
        return cur.rowcount > 0


def get_feedback(book_id: str, db_path: Path | str | None = None) -> str | None:
    """Get the current feedback type for a book ID, or None if none saved."""
    clean_id = str(book_id).strip()
    if not clean_id:
        return None
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT feedback_type FROM book_feedback WHERE book_id = ?", (clean_id,)
        )
        row = cur.fetchone()
        return str(row["feedback_type"]) if row else None


def get_all_feedback(db_path: Path | str | None = None) -> dict[str, str]:
    """Return a mapping of book_id -> feedback_type for all saved feedback."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT book_id, feedback_type FROM book_feedback")
        return {str(row["book_id"]): str(row["feedback_type"]) for row in cur.fetchall()}


def get_feedback_by_type(feedback_type: str, db_path: Path | str | None = None) -> set[str]:
    """Return the set of book IDs associated with a specific feedback type."""
    clean_type = str(feedback_type).strip().lower()
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT book_id FROM book_feedback WHERE feedback_type = ?", (clean_type,)
        )
        return {str(row["book_id"]) for row in cur.fetchall()}


def get_feedback_records(db_path: Path | str | None = None) -> list[dict]:
    """Return all feedback records ordered from newest to oldest."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT book_id, feedback_type, created_at, updated_at
            FROM book_feedback
            ORDER BY updated_at DESC
            """
        )
        return [
            {
                "book_id": str(row["book_id"]),
                "feedback_type": str(row["feedback_type"]),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
            }
            for row in cur.fetchall()
        ]


def clear_all_feedback(db_path: Path | str | None = None) -> int:
    """Clear all saved reader feedback from the database. Returns count of deleted entries."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM book_feedback")
        conn.commit()
        return cur.rowcount


def get_feedback_summary(db_path: Path | str | None = None) -> dict[str, int]:
    """Return counts of each feedback type and total count."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT feedback_type, COUNT(*) as cnt
            FROM book_feedback
            GROUP BY feedback_type
            """
        )
        counts = {str(row["feedback_type"]): int(row["cnt"]) for row in cur.fetchall()}
    total = sum(counts.values())
    return {
        "like": counts.get("like", 0),
        "dislike": counts.get("dislike", 0),
        "read": counts.get("read", 0),
        "total": total,
    }
