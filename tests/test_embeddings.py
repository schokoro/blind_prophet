"""
Tests for amnesiac/store/embeddings.py.

Required dependencies not yet in pyproject.toml:
  - httpx          (runtime, used by embeddings module)
  - pytest-asyncio (dev, for async test support)

Add to pyproject.toml:
  dependencies:      httpx
  [project.optional-dependencies] dev: pytest-asyncio
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_DIM = 1536


def _make_response(vectors: list[list[float]]) -> MagicMock:
    """Build a mock httpx.Response for the given embedding vectors."""
    body = {
        "data": [
            {"embedding": vec, "index": i}
            for i, vec in enumerate(vectors)
        ]
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value=body)
    return mock_resp


@pytest.mark.asyncio
async def test_embed_documents_adds_prefix():
    from amnesiac.store.embeddings import embed_documents

    fake_vector = [0.0] * _DIM
    mock_resp = _make_response([fake_vector])

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("amnesiac.store.embeddings.httpx.AsyncClient", return_value=mock_client):
        await embed_documents(["привет"])

    call_kwargs = mock_client.post.call_args
    sent_body = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert "search_document: привет" in sent_body["input"]


@pytest.mark.asyncio
async def test_embed_query_adds_prefix():
    from amnesiac.store.embeddings import embed_query

    fake_vector = [0.0] * _DIM
    mock_resp = _make_response([fake_vector])

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("amnesiac.store.embeddings.httpx.AsyncClient", return_value=mock_client):
        await embed_query("привет")

    call_kwargs = mock_client.post.call_args
    sent_body = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
    assert "search_query: привет" in sent_body["input"]


@pytest.mark.asyncio
async def test_embed_documents_returns_vectors():
    from amnesiac.store.embeddings import embed_documents

    fake_vectors = [[float(i)] * _DIM for i in range(3)]
    mock_resp = _make_response(fake_vectors)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("amnesiac.store.embeddings.httpx.AsyncClient", return_value=mock_client):
        result = await embed_documents(["a", "b", "c"])

    assert isinstance(result, list)
    assert len(result) == 3
    for vec in result:
        assert isinstance(vec, list)
        assert len(vec) == _DIM
        assert all(isinstance(v, float) for v in vec)


@pytest.mark.asyncio
async def test_embed_query_returns_vector():
    from amnesiac.store.embeddings import embed_query

    fake_vector = [0.5] * _DIM
    mock_resp = _make_response([fake_vector])

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("amnesiac.store.embeddings.httpx.AsyncClient", return_value=mock_client):
        result = await embed_query("тест")

    assert isinstance(result, list)
    assert len(result) == _DIM
    assert all(isinstance(v, float) for v in result)
