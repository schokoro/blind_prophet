import logging

import httpx

from amnesiac.config import settings

logger = logging.getLogger(__name__)

_MODEL = "ai-forever/FRIDA"
_DIM = 1536


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents. Prepends 'search_document: ' prefix to each text."""
    prefixed = [f"search_document: {t}" for t in texts]
    vectors = await _call_api(prefixed)
    assert all(len(v) == _DIM for v in vectors), (
        f"Expected vectors of dimension {_DIM}, got {[len(v) for v in vectors]}"
    )
    logger.debug("embed_documents: %d texts -> %d vectors", len(texts), len(vectors))
    return vectors


async def embed_query(text: str) -> list[float]:
    """Embed a single query. Prepends 'search_query: ' prefix to the text."""
    prefixed = f"search_query: {text}"
    vectors = await _call_api([prefixed])
    vector = vectors[0]
    assert len(vector) == _DIM, f"Expected vector of dimension {_DIM}, got {len(vector)}"
    logger.debug("embed_query: 1 text -> vector of dim %d", len(vector))
    return vector


async def _call_api(inputs: list[str]) -> list[list[float]]:
    url = f"{settings.embeddings_url}/embeddings"
    payload = {"model": _MODEL, "input": inputs}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
    data = response.json()
    # OpenAI-compatible response: {"data": [{"embedding": [...], "index": 0}, ...]}
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]
