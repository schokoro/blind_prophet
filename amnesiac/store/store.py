import logging
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    import sqlite_vec as _sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False


def upsert_channel(conn: sqlite3.Connection, username: str, title: str | None) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO channels (username, title) VALUES (?, ?)",
        (username, title),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM channels WHERE username = ?", (username,)).fetchone()
    return row[0]


def upsert_account(conn: sqlite3.Connection, phone: str, session_file: str) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO accounts (phone, session_file) VALUES (?, ?)",
        (phone, session_file),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM accounts WHERE phone = ?", (phone,)).fetchone()
    return row[0]


def insert_messages(conn: sqlite3.Connection, rows: list[dict]) -> None:
    if not rows:
        return

    conn.executemany(
        """
        INSERT OR IGNORE INTO messages
            (channel_id, tg_message_id, text, date, views, forwards,
             reply_to_msg_id, media_type, raw_json)
        VALUES
            (:channel_id, :tg_message_id, :text, :date, :views, :forwards,
             :reply_to_msg_id, :media_type, :raw_json)
        """,
        rows,
    )

    # group by channel_id and update last_scraped_msg_id per channel
    from collections import defaultdict

    by_channel: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        by_channel[row["channel_id"]].append(row["tg_message_id"])

    now = datetime.now(timezone.utc).isoformat()
    for channel_id, msg_ids in by_channel.items():
        max_id = max(msg_ids)
        conn.execute(
            """
            UPDATE channels
            SET last_scraped_msg_id = MAX(COALESCE(last_scraped_msg_id, 0), ?),
                scraped_at = ?
            WHERE id = ?
            """,
            (max_id, now, channel_id),
        )

    conn.commit()


def get_last_msg_id(conn: sqlite3.Connection, channel_id: int) -> int | None:
    row = conn.execute(
        "SELECT last_scraped_msg_id FROM channels WHERE id = ?", (channel_id,)
    ).fetchone()
    if row is None:
        return None
    return row[0]


def insert_embeddings(conn: sqlite3.Connection, pairs: list[tuple[int, list[float]]]) -> None:
    from amnesiac.config import settings

    if not pairs:
        return

    if not settings.sqlite_vec_enabled or not _SQLITE_VEC_AVAILABLE:
        logger.warning(
            "insert_embeddings: skipped (sqlite_vec_enabled=%s, available=%s)",
            settings.sqlite_vec_enabled,
            _SQLITE_VEC_AVAILABLE,
        )
        return

    conn.executemany(
        "INSERT OR REPLACE INTO vec_messages (message_id, embedding) VALUES (?, ?)",
        [(mid, _sqlite_vec.serialize_float32(vec)) for mid, vec in pairs],
    )
    conn.commit()
