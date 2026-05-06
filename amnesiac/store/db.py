import logging
import sqlite3
from pathlib import Path

try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    _SQLITE_VEC_AVAILABLE = False

logger = logging.getLogger(__name__)

_BASE_MIGRATIONS: list[tuple[str, str | list[str]]] = [
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
        "005_create_processed_messages",
        """
        CREATE TABLE IF NOT EXISTS processed_messages (
            id INTEGER PRIMARY KEY,
            message_id INTEGER UNIQUE NOT NULL REFERENCES messages(id),
            processed_text TEXT NOT NULL,
            is_valid BOOLEAN NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ),
    (
        "006_recreate_vec_messages",
        [
            "DROP TABLE IF EXISTS vec_messages",
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
                message_id INTEGER PRIMARY KEY,
                embedding FLOAT[1536]
            )
            """,
        ],
    ),
    (
        "007_create_summaries",
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id          INTEGER PRIMARY KEY,
            run_date    TEXT NOT NULL,
            summary     TEXT NOT NULL,
            doc_count   INTEGER,
            model       TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_date)
        )
        """,
    ),
    (
        "008_create_infom",
        """
        CREATE TABLE IF NOT EXISTS infom_expectations (
            id          INTEGER PRIMARY KEY,
            survey_date TEXT UNIQUE NOT NULL,
            median_12m  REAL NOT NULL,
            source_url  TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ),
    (
        "009_create_neutered_summaries",
        """
        CREATE TABLE IF NOT EXISTS neutered_summaries (
            run_date                  TEXT PRIMARY KEY,
            summary                   TEXT NOT NULL,
            neutering_status          TEXT NOT NULL,
            final_iteration           INTEGER,
            q3_preservation           REAL,
            raw_period_id_score       REAL,
            neutered_period_id_score  REAL,
            period_delta_vs_raw       REAL,
            judge_blind               INTEGER,
            model_n                   TEXT,
            model_j1                  TEXT,
            model_j2                  TEXT,
            created_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ),
    (
        "010_create_forecasts",
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            run_date      TEXT NOT NULL,
            condition     TEXT NOT NULL,
            persona       TEXT NOT NULL,
            sample_index  INTEGER NOT NULL,
            value         REAL,
            model         TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (run_date, condition, persona, sample_index)
        )
        """,
    ),
]


def _get_migrations() -> list[tuple[str, str | list[str]]]:
    from amnesiac.config import settings

    if settings.sqlite_vec_enabled:
        return _BASE_MIGRATIONS
    # exclude only 006 (vec-dependent) when sqlite_vec is disabled
    return _BASE_MIGRATIONS[:4] + _BASE_MIGRATIONS[5:]


def get_connection(path: str | Path) -> sqlite3.Connection:
    from amnesiac.config import settings

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))

    if settings.sqlite_vec_enabled:
        if _SQLITE_VEC_AVAILABLE:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        else:
            logger.warning(
                "sqlite_vec_enabled=True but sqlite_vec is not installed; "
                "vector storage will be unavailable"
            )

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

    for name, sql in _get_migrations():
        if name in applied:
            continue
        if isinstance(sql, list):
            for statement in sql:
                conn.execute(statement)
        else:
            conn.execute(sql)
        conn.execute("INSERT INTO migrations (name) VALUES (?)", (name,))
        conn.commit()
        logger.info("Applied migration: %s", name)
