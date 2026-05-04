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


def insert_processed_message(
    conn: sqlite3.Connection,
    message_id: int,
    processed_text: str,
    is_valid: bool = True,
) -> int:
    conn.execute(
        """
        INSERT OR IGNORE INTO processed_messages (message_id, processed_text, is_valid)
        VALUES (?, ?, ?)
        """,
        (message_id, processed_text, int(is_valid)),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM processed_messages WHERE message_id = ?", (message_id,)
    ).fetchone()
    return row[0]


def get_unprocessed_messages(conn: sqlite3.Connection, batch_size: int = 500) -> list[dict]:
    rows = conn.execute(
        """
        SELECT m.id, m.channel_id, m.text, m.date
        FROM messages m
        WHERE NOT EXISTS (
            SELECT 1 FROM processed_messages pm WHERE pm.message_id = m.id
        )
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()
    return [
        {"id": row[0], "channel_id": row[1], "text": row[2], "date": row[3]}
        for row in rows
    ]


def load_period(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    exclude_channels: list[str] | None = None,
) -> list[dict]:
    """Load messages with embeddings for a given date range.

    Returns a list of dicts with keys:
        message_id, channel, date, processed_text, embedding_blob

    embedding_blob is raw bytes (1536 float32, little-endian).
    Deserialization to numpy is the caller's responsibility.

    JOIN chain:
        vec_messages.message_id -> processed_messages.id
        processed_messages.message_id -> messages.id
        messages.channel_id -> channels.id
    """
    sql = """
        SELECT
            pm.id           AS message_id,
            ch.username     AS channel,
            m.date          AS date,
            pm.processed_text,
            vm.embedding    AS embedding_blob
        FROM vec_messages vm
        JOIN processed_messages pm ON vm.message_id = pm.id
        JOIN messages m ON pm.message_id = m.id
        JOIN channels ch ON m.channel_id = ch.id
        WHERE m.date BETWEEN ? AND ?
          AND pm.is_valid = 1
    """
    params: list = [date_from, date_to]

    if exclude_channels:
        placeholders = ",".join("?" * len(exclude_channels))
        sql += f" AND ch.username NOT IN ({placeholders})"
        params.extend(exclude_channels)

    sql += " ORDER BY m.date"

    rows = conn.execute(sql, params).fetchall()
    return [
        {
            "message_id": row[0],
            "channel": row[1],
            "date": row[2],
            "processed_text": row[3],
            "embedding_blob": row[4],
        }
        for row in rows
    ]


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


def insert_summary(conn, run_date: str, summary: str, doc_count: int, model: str) -> None:
    """Insert or replace summary for run_date."""
    conn.execute(
        """
        INSERT OR REPLACE INTO summaries (run_date, summary, doc_count, model)
        VALUES (?, ?, ?, ?)
        """,
        (run_date, summary, doc_count, model),
    )
    conn.commit()
