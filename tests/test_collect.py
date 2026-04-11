import asyncio
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amnesiac.store import apply_migrations, get_connection


def _make_msg(
    id: int,
    text: str | None = "some text",
    fwd_from=None,
    views: int = 0,
    forwards: int = 0,
    reply_to_msg_id: int | None = None,
    media=None,
) -> MagicMock:
    msg = MagicMock()
    msg.id = id
    msg.text = text
    msg.date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    msg.views = views
    msg.forwards = forwards
    msg.reply_to_msg_id = reply_to_msg_id
    msg.media = media
    msg.fwd_from = fwd_from
    msg.to_json.return_value = "{}"
    return msg


def _make_conn() -> sqlite3.Connection:
    conn = get_connection(":memory:")
    apply_migrations(conn)
    return conn


async def _aiter(items):
    for item in items:
        yield item


async def _aiter_raise(exc):
    raise exc
    # make it an async generator
    yield  # noqa: unreachable


def _make_client(messages) -> MagicMock:
    client = MagicMock()
    client.get_entity = AsyncMock(return_value=MagicMock(title="Test Channel"))
    client.iter_messages = MagicMock(return_value=_aiter(messages))
    return client


@pytest.mark.asyncio
async def test_scrape_channel_empty():
    from amnesiac.collect import scrape_channel

    conn = _make_conn()
    client = _make_client([])
    await scrape_channel(client, "testchan", conn)

    count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 0


@pytest.mark.asyncio
async def test_scrape_channel_inserts_messages():
    from amnesiac.collect import scrape_channel

    conn = _make_conn()
    msgs = [_make_msg(i, text=f"message {i}") for i in range(1, 4)]
    client = _make_client(msgs)
    await scrape_channel(client, "testchan", conn)

    count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 3


@pytest.mark.asyncio
async def test_scrape_channel_skips_reposts():
    from amnesiac.collect import scrape_channel

    conn = _make_conn()
    msgs = [
        _make_msg(1, text="normal"),
        _make_msg(2, text="repost", fwd_from=MagicMock()),  # should be skipped
        _make_msg(3, text="also normal"),
    ]
    client = _make_client(msgs)
    await scrape_channel(client, "testchan", conn)

    count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 2


@pytest.mark.asyncio
async def test_scrape_channel_skips_no_text():
    from amnesiac.collect import scrape_channel

    conn = _make_conn()
    msgs = [
        _make_msg(1, text="has text"),
        _make_msg(2, text=None),  # should be skipped
        _make_msg(3, text="also has text"),
    ]
    client = _make_client(msgs)
    await scrape_channel(client, "testchan", conn)

    count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 2


@pytest.mark.asyncio
async def test_scrape_channel_resumes_from_cursor():
    from amnesiac.collect import scrape_channel

    conn = _make_conn()
    # pre-seed channel with last_scraped_msg_id = 5
    conn.execute("INSERT INTO channels (username, title, last_scraped_msg_id) VALUES (?, ?, ?)", ("testchan", "T", 5))
    conn.commit()

    client = _make_client([])
    await scrape_channel(client, "testchan", conn)

    client.iter_messages.assert_called_once()
    call_kwargs = client.iter_messages.call_args
    assert call_kwargs.kwargs.get("min_id") == 5, (
        f"Expected min_id=5 in iter_messages call, got: {call_kwargs}"
    )


@pytest.mark.asyncio
async def test_scrape_channel_flood_wait():
    from telethon.errors import FloodWaitError

    from amnesiac.collect import scrape_channel

    conn = _make_conn()

    # Build a FloodWaitError — Telethon uses capture= for the wait seconds
    error = FloodWaitError(request=None, capture=1)

    async def _raising_iter(*args, **kwargs):
        raise error
        yield  # make it an async generator

    async def _empty_iter(*args, **kwargs):
        return
        yield  # make it an async generator

    client = MagicMock()
    client.get_entity = AsyncMock(return_value=MagicMock(title="Test Channel"))
    # First call raises FloodWaitError, second call (after sleep) returns empty
    client.iter_messages = MagicMock(side_effect=[_raising_iter(), _empty_iter()])

    with patch("amnesiac.collect.scraper.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        # should not raise
        await scrape_channel(client, "testchan", conn)
        mock_sleep.assert_awaited_once_with(1)
        assert client.iter_messages.call_count == 2, (
            f"Expected iter_messages called twice, got {client.iter_messages.call_count}"
        )


@pytest.mark.asyncio
async def test_run_all_distributes_channels():
    """Test that run_all distributes channels across accounts via round-robin."""
    accounts_yaml = {
        "accounts": [
            {"phone": "+70000000001", "session_file": "sess_a", "api_id": 111, "api_hash": "hash_a"},
            {"phone": "+70000000002", "session_file": "sess_b", "api_id": 222, "api_hash": "hash_b"},
        ]
    }
    channels_yaml = {
        "channels": ["chan1", "chan2", "chan3", "chan4"],
    }

    scraped: dict[str, list[str]] = defaultdict(list)
    client_calls: list[tuple] = []

    async def fake_scrape(client, channel, conn, batch_size=200, inter_batch_sleep=1.0):
        scraped[client._session_file].append(channel)

    def make_mock_client(session_file, api_id, api_hash):
        client_calls.append((session_file, api_id, api_hash))
        c = MagicMock()
        c._session_file = session_file
        c.connect = AsyncMock()
        c.is_user_authorized = AsyncMock(return_value=True)
        c.disconnect = AsyncMock()
        return c

    with (
        patch("amnesiac.collect.runner.yaml.safe_load", side_effect=[accounts_yaml, channels_yaml]),
        patch("amnesiac.collect.runner.Path.read_text", return_value=""),
        patch("amnesiac.collect.runner.TelegramClient", side_effect=make_mock_client),
        patch("amnesiac.collect.runner.scrape_channel", side_effect=fake_scrape),
        patch("amnesiac.collect.runner.get_connection", return_value=MagicMock()),
        patch("amnesiac.collect.runner.apply_migrations"),
        patch("amnesiac.collect.runner.settings") as mock_settings,
    ):
        mock_settings.batch_size = 200
        mock_settings.inter_batch_sleep = 1.0

        from amnesiac.collect.runner import run_all
        await run_all(":memory:")

    total_calls = sum(len(v) for v in scraped.values())
    assert total_calls == 4, f"Expected 4 scrape_channel calls, got {total_calls}"
    assert len(scraped["sess_a"]) == 2, f"Expected sess_a to handle 2 channels, got {len(scraped['sess_a'])}"
    assert len(scraped["sess_b"]) == 2, f"Expected sess_b to handle 2 channels, got {len(scraped['sess_b'])}"

    # TelegramClient must be called with per-account credentials, not settings-level
    assert ("sess_a", 111, "hash_a") in client_calls, f"Expected sess_a with account creds, got {client_calls}"
    assert ("sess_b", 222, "hash_b") in client_calls, f"Expected sess_b with account creds, got {client_calls}"
