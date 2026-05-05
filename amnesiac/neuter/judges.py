"""Async judge call stubs and retry helper for summary neutering."""

import asyncio
import json
import logging

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

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
    raise NotImplementedError("p02")


async def call_j1_q3(client: AsyncOpenAI, summary_text: str) -> dict:
    raise NotImplementedError("p02")


async def call_n_rewriter(
    client: AsyncOpenAI,
    prev_summary: str,
    q1_identifiers: list[dict],
    q3_signals: list[dict],
) -> str:
    raise NotImplementedError("p02")


async def call_j2(client: AsyncOpenAI, summary_text: str) -> dict:
    raise NotImplementedError("p03")

