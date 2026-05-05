"""Iterative neutering cycle entry point."""

from openai import AsyncOpenAI


async def run_cycle(client: AsyncOpenAI, raw_summary: str) -> dict:
    """Run up to N_MAX iterations of N → J1(Q1+Q3) → residual check.
    Returns dict with keys: summary, status, final_iteration, q3_preservation."""
    raise NotImplementedError("p02")

