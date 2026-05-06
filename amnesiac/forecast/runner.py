import asyncio
import logging
import math
import os
import random
import re
from pathlib import Path

from amnesiac.forecast.personas import PERSONA_BODIES, TRAILING_PROMPT
from amnesiac.forecast.store import (
    delete_forecasts,
    forecast_exists,
    insert_samples,
    load_summary_for_forecast,
)
from amnesiac.store import apply_migrations, get_connection

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_P = "deepseek/deepseek-v4-flash"
CONDITIONS = {"raw", "neutered"}


def _validate_inputs(persona: str, condition: str, n_samples: int) -> None:
    if persona not in PERSONA_BODIES:
        raise ValueError(f"Unknown persona: {persona}")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")


def _parse_float(text: str | None) -> float:
    if not text:
        return float("nan")

    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text.strip())
    if not match:
        return float("nan")

    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return float("nan")


def _build_messages(persona: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": PERSONA_BODIES[persona]},
        {
            "role": "user",
            "content": f"Новостная сводка:\n\n{summary}\n\n{TRAILING_PROMPT}",
        },
    ]


async def _run_one_sample(client, persona: str, summary: str) -> float:
    response = await client.chat.completions.create(
        model=MODEL_P,
        messages=_build_messages(persona, summary),
        temperature=1.0,
    )
    return _parse_float(response.choices[0].message.content)


async def run(
    persona: str,
    summary: str,
    condition: str,
    n_samples: int,
    mock: bool = False,
) -> dict:
    """
    Returns:
        {
            "persona": persona,
            "condition": condition,
            "samples": list[float],
        }
    """
    _validate_inputs(persona, condition, n_samples)

    if mock:
        samples = [float(random.uniform(5.0, 20.0)) for _ in range(n_samples)]
        return {"persona": persona, "condition": condition, "samples": samples}

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=600.0,
    )
    samples = await asyncio.gather(*(_run_one_sample(client, persona, summary) for _ in range(n_samples)))

    nan_count = sum(1 for sample in samples if math.isnan(sample))
    if nan_count:
        logger.warning("Forecast runner produced %.1f%% NaN samples", nan_count / n_samples * 100.0)

    return {"persona": persona, "condition": condition, "samples": samples}


async def run_forecast_pipeline(
    db_path: Path,
    run_date: str,
    *,
    condition: str,
    force: bool = False,
    n_samples: int = 3,
) -> None:
    """
    Run forecast pipeline for a given run_date and condition spec.
    Persists samples in `forecasts` table.
    """
    valid_conditions = {"raw", "neutered", "both"}
    if condition not in valid_conditions:
        raise ValueError(f"Unknown condition: {condition}")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    conn = get_connection(db_path)
    try:
        apply_migrations(conn)

        conditions = ["raw", "neutered"] if condition == "both" else [condition]
        persona_names = list(PERSONA_BODIES)

        for cond in conditions:
            existing = forecast_exists(conn, run_date, cond)
            if existing and not force:
                logger.info(
                    "forecasts already has rows for %s/%s; pass --force to overwrite",
                    run_date,
                    cond,
                )
                continue

            if existing and force:
                deleted = delete_forecasts(conn, run_date, cond)
                logger.info("deleted %d forecast rows for %s/%s", deleted, run_date, cond)

            summary = load_summary_for_forecast(conn, run_date, cond)

            for persona in persona_names:
                result = await run(persona, summary, cond, n_samples)
                insert_samples(conn, run_date, cond, persona, result["samples"], MODEL_P)

            logger.info(
                "forecast run_date=%s condition=%s personas=%d n_samples=%d total_rows=%d",
                run_date,
                cond,
                len(PERSONA_BODIES),
                n_samples,
                len(PERSONA_BODIES) * n_samples,
            )
    finally:
        conn.close()
