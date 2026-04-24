import asyncio
from datetime import date, timedelta

import numpy as np

from amnesiac.config import settings
from amnesiac.store.embeddings import embed_query
from amnesiac.store.store import load_period


def _rag_get(name: str):
    rag = settings.rag
    if isinstance(rag, dict):
        return rag[name]
    return getattr(rag, name)


def _dedup(
    subset_indices: list[int],
    X_full: np.ndarray,
    texts: list[str],
    threshold: float,
) -> list[int]:
    if not subset_indices:
        return []

    X_sub = X_full[subset_indices]
    sim = X_sub @ X_sub.T
    np.fill_diagonal(sim, 0.0)

    dropped: set[int] = set()
    for i in range(len(subset_indices)):
        if i in dropped:
            continue
        for j in range(i + 1, len(subset_indices)):
            if j in dropped:
                continue
            if sim[i, j] < threshold:
                continue

            idx_i = subset_indices[i]
            idx_j = subset_indices[j]
            len_i = len(texts[idx_i])
            len_j = len(texts[idx_j])

            if len_i <= len_j:
                dropped.add(j)
            else:
                dropped.add(i)
                break

    return [subset_indices[i] for i in range(len(subset_indices)) if i not in dropped]


async def retrieve(conn, run_date: str) -> dict[str, list[dict]]:
    run_day = date.fromisoformat(run_date)
    horizon_days = int(_rag_get("horizon_days"))
    date_to = run_day
    date_from = run_day - timedelta(days=horizon_days)

    rows = load_period(
        conn,
        str(date_from),
        str(date_to),
        _rag_get("exclude_channels"),
    )

    axes = _rag_get("axes")
    if not rows:
        return {axis_name: [] for axis_name in axes}

    X_rows: list[np.ndarray] = []
    texts: list[str] = []
    day_numbers: list[int] = []

    for row in rows:
        vec = np.frombuffer(row["embedding_blob"], dtype="<f4").copy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        X_rows.append(vec)

        texts.append(row["processed_text"])
        row_day = date.fromisoformat(row["date"][:10])
        day_numbers.append((row_day - date_from).days + 1)

    X = np.stack(X_rows, axis=0)

    top_k = int(_rag_get("top_k_per_axis"))
    dedup_threshold = float(_rag_get("dedup_threshold"))

    result: dict[str, list[dict]] = {}
    for axis_name, axis_queries in axes.items():
        q_vectors = await asyncio.gather(*(embed_query(q) for q in axis_queries))
        q_mat = np.asarray(q_vectors, dtype=np.float32)
        q_vec = q_mat.mean(axis=0)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        sims = X @ q_vec

        k = min(top_k, len(rows))
        if k == len(rows):
            top_idx = np.argsort(-sims)
        else:
            part = np.argpartition(sims, -k)[-k:]
            top_idx = part[np.argsort(-sims[part])]

        kept_idx = _dedup(top_idx.tolist(), X, texts, dedup_threshold)
        kept_idx.sort(key=lambda idx: day_numbers[idx])

        result[axis_name] = [
            {
                "message_id": rows[idx]["message_id"],
                "channel": rows[idx]["channel"],
                "day_number": day_numbers[idx],
                "processed_text": rows[idx]["processed_text"],
            }
            for idx in kept_idx
        ]

    return result


def format_for_prompt(docs: list[dict]) -> str:
    return "\n".join(
        f"[день {doc['day_number']} | {doc['channel']}] {doc['processed_text']}"
        for doc in docs
    )
