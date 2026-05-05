"""Parallel J2 holdout sampling with period-aware scoring aggregate."""

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

from amnesiac.neuter.config import J2_N_SAMPLES, J2_SEMAPHORE_LIMIT
from amnesiac.neuter.judges import call_j2
from amnesiac.neuter.metrics import score_identification

logger = logging.getLogger(__name__)


async def run_j2_holdout(
    client: AsyncOpenAI,
    summary_text: str,
    true_year: int,
    true_month: int,
    *,
    n_samples: int = J2_N_SAMPLES,
    label: str = "holdout",
) -> dict[str, Any]:
    """
    Run n_samples parallel J2 calls under a semaphore and return an aggregate score.

    Failed samples are tolerated only while at least 3 successful samples remain.
    """
    semaphore = asyncio.Semaphore(J2_SEMAPHORE_LIMIT)

    async def one_sample(sample_index: int) -> dict | None:
        async with semaphore:
            try:
                parsed = await call_j2(client, summary_text)
                scoring = score_identification(parsed, true_year, true_month)
                logger.debug(
                    "j2 holdout %s sample %d/%d: period_success=%s weight=%.3f",
                    label,
                    sample_index,
                    n_samples,
                    scoring["period_success_level"],
                    scoring["period_weight"],
                )
                return scoring
            except Exception as exc:
                logger.warning("j2 holdout %s sample %d failed: %r", label, sample_index, exc)
                return None

    raw_results = await asyncio.gather(
        *(one_sample(i) for i in range(1, n_samples + 1)),
        return_exceptions=False,
    )
    samples = [result for result in raw_results if result is not None]
    n_failed = n_samples - len(samples)

    if not samples:
        raise RuntimeError(
            f"j2 holdout '{label}' produced zero successful samples "
            f"out of {n_samples}; cannot aggregate"
        )
    if n_failed > 2 or len(samples) < 3:
        raise RuntimeError(
            f"j2 holdout '{label}' had {n_failed}/{n_samples} failed samples; "
            "refusing to aggregate (threshold: tolerate up to 2 failed samples "
            "and require at least 3 successful samples)"
        )

    period_id_score = sum(sample["period_weighted_success"] for sample in samples) / len(samples)
    logger.info(
        "j2 holdout %s: period_id_score=%.3f over %d/%d successful samples",
        label,
        period_id_score,
        len(samples),
        n_samples,
    )
    return {
        "period_id_score": period_id_score,
        "n_samples": len(samples),
        "n_failed": n_failed,
        "samples": samples,
    }
