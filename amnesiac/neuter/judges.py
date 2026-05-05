"""Async judge call stubs and retry helper for summary neutering."""

import asyncio
import json
import logging

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from amnesiac.neuter.config import MODEL_J1, MODEL_J2, MODEL_N, J2_TEMPERATURE, TEMPERATURE_J1, TEMPERATURE_N
from amnesiac.neuter.metrics import extract_json_response, validate_j2_parsed
from amnesiac.neuter.prompts import (
    J2_IDENTIFIABILITY_SYSTEM,
    N_REWRITER_SYSTEM,
    Q1_EVIDENCE_SYSTEM,
    Q3_SIGNALS_SYSTEM,
    make_j1_user_prompt,
    make_j2_user_prompt,
    make_n_rewriter_user_prompt,
)

logger = logging.getLogger(__name__)


async def _call_with_retry(coro_factory, axis_name: str, max_attempts: int = 3):
    """
    Run an OpenRouter request with retries for transient provider failures.

    coro_factory must return a fresh coroutine on every call.
    """
    retry_delays = [5, 15, 45]
    retryable_errors = (
        json.JSONDecodeError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        APIConnectionError,
        APITimeoutError,
    )
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("OpenRouter request attempt %s/%s for axis %s", attempt, max_attempts, axis_name)
            return await coro_factory()
        except retryable_errors as exc:
            last_error = exc
            if attempt >= max_attempts:
                logger.exception(
                    "OpenRouter request failed after %s attempts for axis %s",
                    max_attempts,
                    axis_name,
                )
                raise

            delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
            logger.warning(
                "OpenRouter request failed for axis %s on attempt %s/%s: %r; retrying in %ss",
                axis_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    raise last_error


async def call_j1_q1(client: AsyncOpenAI, summary_text: str) -> dict:
    """Run J1 Q1-evidence judge on a summary; return the parsed JSON dict."""
    user_prompt = make_j1_user_prompt(summary_text)
    response = await _call_with_retry(
        lambda: client.chat.completions.create(
            model=MODEL_J1,
            temperature=TEMPERATURE_J1,
            messages=[
                {"role": "system", "content": Q1_EVIDENCE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        ),
        axis_name="j1_q1",
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Model {MODEL_J1} returned empty content for j1_q1")
    return extract_json_response(content)


async def call_j1_q3(client: AsyncOpenAI, summary_text: str) -> dict:
    """Run J1 Q3-signals judge on a summary; return the parsed JSON dict."""
    user_prompt = make_j1_user_prompt(summary_text)
    response = await _call_with_retry(
        lambda: client.chat.completions.create(
            model=MODEL_J1,
            temperature=TEMPERATURE_J1,
            messages=[
                {"role": "system", "content": Q3_SIGNALS_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        ),
        axis_name="j1_q3",
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Model {MODEL_J1} returned empty content for j1_q3")
    return extract_json_response(content)


async def call_n_rewriter(
    client: AsyncOpenAI,
    prev_summary: str,
    q1_identifiers: list[dict],
    q3_signals: list[dict],
) -> str:
    """Run the N rewriter; return the candidate summary as plain text."""
    user_prompt = make_n_rewriter_user_prompt(prev_summary, q1_identifiers, q3_signals)
    response = await _call_with_retry(
        lambda: client.chat.completions.create(
            model=MODEL_N,
            temperature=TEMPERATURE_N,
            messages=[
                {"role": "system", "content": N_REWRITER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        ),
        axis_name="n_rewriter",
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Model {MODEL_N} returned empty content for n_rewriter")
    stripped = content.strip()
    if not stripped:
        raise RuntimeError(f"Model {MODEL_N} returned empty content for n_rewriter")
    return stripped


async def call_j2(client: AsyncOpenAI, summary_text: str) -> dict:
    """Run the J2 holdout judge on a summary; return the parsed and validated JSON dict."""
    user_prompt = make_j2_user_prompt(summary_text)
    response = await _call_with_retry(
        lambda: client.chat.completions.create(
            model=MODEL_J2,
            temperature=J2_TEMPERATURE,
            messages=[
                {"role": "system", "content": J2_IDENTIFIABILITY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        ),
        axis_name="j2",
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Model {MODEL_J2} returned empty content for j2")
    parsed = extract_json_response(content)
    validate_j2_parsed(parsed)
    return parsed
