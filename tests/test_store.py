import sqlite3
from pathlib import Path

import pytest

from amnesiac.store import (
    apply_migrations,
    get_connection,
    get_last_msg_id,
    insert_embeddings,
    insert_messages,
    upsert_channel,
)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "test.db"
    c = get_connection(db)
    apply_migrations(c)
    return c


def test_apply_migrations_idempotent(conn: sqlite3.Connection) -> None:
    apply_migrations(conn)  # second call — must not raise


def test_upsert_channel(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "test_channel", "Test Channel")
    assert isinstance(cid, int)
    cid2 = upsert_channel(conn, "test_channel", "Test Channel")
    assert cid == cid2


def test_insert_messages_no_duplicate_error(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "chan", None)
    row = dict(
        channel_id=cid,
        tg_message_id=1,
        text="hello",
        date="2024-01-01T00:00:00",
        views=10,
        forwards=0,
        reply_to_msg_id=None,
        media_type=None,
        raw_json="{}",
    )
    insert_messages(conn, [row])
    insert_messages(conn, [row])  # duplicate — must not raise


def test_get_last_msg_id_none(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "empty_chan", None)
    assert get_last_msg_id(conn, cid) is None


def test_get_last_msg_id_returns_max(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "chan2", None)
    rows = [
        dict(
            channel_id=cid,
            tg_message_id=5,
            text="a",
            date="2024-01-01T00:00:00",
            views=0,
            forwards=0,
            reply_to_msg_id=None,
            media_type=None,
            raw_json="{}",
        ),
        dict(
            channel_id=cid,
            tg_message_id=10,
            text="b",
            date="2024-01-02T00:00:00",
            views=0,
            forwards=0,
            reply_to_msg_id=None,
            media_type=None,
            raw_json="{}",
        ),
    ]
    insert_messages(conn, rows)
    assert get_last_msg_id(conn, cid) == 10


def test_insert_embeddings(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "chan3", None)
    row = dict(
        channel_id=cid,
        tg_message_id=1,
        text="x",
        date="2024-01-01T00:00:00",
        views=0,
        forwards=0,
        reply_to_msg_id=None,
        media_type=None,
        raw_json="{}",
    )
    insert_messages(conn, [row])
    msg_id = conn.execute("SELECT id FROM messages WHERE tg_message_id = 1").fetchone()[0]
    vector = [0.1] * 1536
    insert_embeddings(conn, [(msg_id, vector)])  # must not raise
