import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from amnesiac.store import (
    apply_migrations,
    get_connection,
    get_last_msg_id,
    get_unprocessed_messages,
    insert_embeddings,
    insert_messages,
    insert_processed_message,
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


def test_sqlite_vec_disabled(tmp_path: Path) -> None:
    import amnesiac.config as cfg
    import amnesiac.store.db as db_mod
    import amnesiac.store.store as store_mod

    with patch.object(cfg.settings, "sqlite_vec_enabled", False):
        c = get_connection(tmp_path / "disabled.db")
        apply_migrations(c)

        # vec_messages table must not exist
        tables = {
            row[0]
            for row in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "vec_messages" not in tables

        # insert_embeddings must not raise
        insert_embeddings(c, [(1, [0.1] * 1536)])


def _make_message(cid: int, tg_id: int, text: str = "x") -> dict:
    return dict(
        channel_id=cid,
        tg_message_id=tg_id,
        text=text,
        date="2024-01-01T00:00:00",
        views=0,
        forwards=0,
        reply_to_msg_id=None,
        media_type=None,
        raw_json="{}",
    )


def test_insert_processed_message(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "chan_pm", None)
    insert_messages(conn, [_make_message(cid, 1)])
    msg_id = conn.execute("SELECT id FROM messages WHERE tg_message_id = 1").fetchone()[0]

    pm_id = insert_processed_message(conn, msg_id, "processed text")

    assert isinstance(pm_id, int)
    row = conn.execute(
        "SELECT processed_text FROM processed_messages WHERE id = ?", (pm_id,)
    ).fetchone()
    assert row is not None
    assert row[0] == "processed text"


def test_insert_processed_message_idempotent(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "chan_pm2", None)
    insert_messages(conn, [_make_message(cid, 2)])
    msg_id = conn.execute("SELECT id FROM messages WHERE tg_message_id = 2").fetchone()[0]

    pm_id1 = insert_processed_message(conn, msg_id, "text")
    pm_id2 = insert_processed_message(conn, msg_id, "text")

    assert pm_id1 == pm_id2


def test_get_unprocessed_messages(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "chan_unproc", None)
    insert_messages(conn, [_make_message(cid, 10), _make_message(cid, 11), _make_message(cid, 12)])
    ids = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT tg_message_id, id FROM messages WHERE tg_message_id IN (10, 11, 12)"
        ).fetchall()
    }
    # process only message with tg_message_id=10
    insert_processed_message(conn, ids[10], "done")

    result = get_unprocessed_messages(conn)

    assert len(result) == 2
    result_ids = {r["id"] for r in result}
    assert result_ids == {ids[11], ids[12]}
    for r in result:
        assert set(r.keys()) == {"id", "channel_id", "text", "date"}


def test_get_unprocessed_messages_batch_size(conn: sqlite3.Connection) -> None:
    cid = upsert_channel(conn, "chan_batch", None)
    insert_messages(conn, [_make_message(cid, tg_id) for tg_id in range(20, 25)])

    result = get_unprocessed_messages(conn, batch_size=2)

    assert len(result) == 2


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
