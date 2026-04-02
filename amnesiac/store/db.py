import logging
import sqlite3
from pathlib import Path

import sqlite_vec

logger = logging.getLogger(__name__)

MIGRATIONS: list[tuple[str, str]] = [
    (
        "001_create_accounts",
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY,
            phone TEXT UNIQUE NOT NULL,
            session_file TEXT UNIQUE NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            daily_requests INTEGER DEFAULT 0,
            last_used_at TIMESTAMP
        )
        """,
    ),
    (
        "002_create_channels",
        """
        CREATE TABLE IF NOT EXISTS channels (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            title TEXT,
            last_scraped_msg_id INTEGER,
            scraped_at TIMESTAMP
        )
        """,
    ),
    (
        "003_create_messages",
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,
            channel_id INTEGER REFERENCES channels(id),
            tg_message_id INTEGER NOT NULL,
            text TEXT,
            date TIMESTAMP NOT NULL,
            views INTEGER,
            forwards INTEGER,
            reply_to_msg_id INTEGER,
            media_type TEXT,
            raw_json TEXT,
            UNIQUE(channel_id, tg_message_id)
        )
        """,
    ),
    (
        "004_create_vec_messages",
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
            message_id INTEGER PRIMARY KEY,
            embedding FLOAT[1536]
        )
        """,
    ),
]


def get_connection(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            name TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()

    applied = {row[0] for row in conn.execute("SELECT name FROM migrations")}

    for name, sql in MIGRATIONS:
        if name in applied:
            continue
        conn.execute(sql)
        conn.execute("INSERT INTO migrations (name) VALUES (?)", (name,))
        conn.commit()
        logger.info("Applied migration: %s", name)
