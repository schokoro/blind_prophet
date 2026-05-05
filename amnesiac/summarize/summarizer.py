import asyncio
import json
import logging
import time

import httpx
from openai import APIConnectionError, APITimeoutError

from amnesiac.summarize.retriever import format_for_prompt

logger = logging.getLogger(__name__)

AXIS_SYSTEM = """Ты — экономический аналитик. Твоя задача — составить структурированный таймлайн событий по одной конкретной тематической оси из потока новостных сообщений.

Правила:

1. **Относительные даты везде.** Используй «день 1», «день 5-7», «вторая неделя». Это касается:
   - дат публикации новостей (маркер `[день N | канал]` на входе)
   - **дат событий ВНУТРИ самих новостей** — если в тексте написано «с 18 октября АвтоВАЗ повысил цены» или «инфляция за неделю 21-27 сентября составила 7.26%», ты должен переписать это в относительных терминах («в конце периода производитель повысил цены», «по данным за неделю в конце предыдущего месяца инфляция составила 7.26%»), сохранив смысл и числовые показатели.
   - Если дату невозможно перефразировать без потери смысла, замени фрагмент с датой на токен `[дата]` с пробелами вокруг.
   - Не удаляй даты inline: не вырезай слово вместе с пробелами и не склеивай соседние токены.
   - **никаких упоминаний месяцев и лет** (сентябрь, октябрь, 2021) — ни в каком виде.
   - **никаких числовых дат** (21-27, 18.10 и т.п.).

2. **Релевантность оси — функциональная, а не тематическая.** Сохраняй только те сообщения, которые содержательно влияют на прогноз потребительских цен и инфляционных ожиданий. Для оси «{axis}» это означает:
   - ось про макроэкономический показатель, денежную политику, цены или доходы — сохраняй
   - ось про организационную культуру, HR-исследования, конфликты на работе, карьерные истории, бытовые частности — пропускай, даже если тема формально совпадает

3. Сохраняй числовые показатели (проценты, значения курса, ставки, объёмы) и имена собственные — они важны для последующего анализа.
4. Сохраняй направления движений: подорожал/подешевел, вырос/упал, повысил/понизил.
5. Фиксируй причинно-следственные связки там, где они явно прослеживаются.
6. Формат: структурированный список событий по смысловым узлам, а не по каждому сообщению в отдельности. Одинаковые/близкие события схлопывай в один пункт.
7. В конце — краткий блок «Ключевые показатели»: числа и значения, встречающиеся в текстах (без дат).
8. Не добавляй общих оценок, прогнозов или мета-комментариев. Только факты из текстов.

Ось: {axis}
Период: {horizon_days} дней
"""

AXIS_USER = """Новостные сообщения (отсортированы по времени, формат [день N | канал] текст):

{docs}

Составь таймлайн по оси «{axis}»."""

META_SYSTEM = """Ты — экономический аналитик. На вход ты получаешь 9 тематических таймлайнов одного и того же 14-дневного периода, составленных по разным осям (ДКП, инфляция, курс и т.д.).

Твоя задача — собрать из них единый нарративный таймлайн по смысловым событийным узлам.

Правила:

1. **Относительные даты везде.** Используй «день 1», «вторая неделя», «в конце периода». 
   - **Если в исходных axis-саммари просочились абсолютные даты** (месяцы, годы, числа месяца — «сентябрь 2021», «18 октября», «21-27 числа») — переписывай их в относительных терминах или заменяй фрагмент с датой на токен `[дата]` с пробелами вокруг.
   - Не удаляй даты inline: не вырезай слово вместе с пробелами и не склеивай соседние токены.
   - **Никаких упоминаний конкретных месяцев, годов или числовых дат** в твоём выходе. Это жёсткое требование.

2. **Пустые оси — пропускай молча.** Если axis-саммари содержит только «нет релевантного контента» или аналогичный шлюз-ответ — не упоминай эту ось в выходе вообще, продолжай по остальным.

3. Группируй события по смысловым узлам, а не по осям. Если в один день/дни пересекаются события из нескольких осей (например, повышение ставки + ослабление рубля + паника потребителей) — это один узел.

4. В каждом узле фиксируй:
   - что произошло (факты, действующие лица, числа)
   - какие темы он затрагивает
   - причинно-следственные связи с другими узлами, если они очевидны

5. Сохраняй все числовые показатели и имена собственные из исходных axis-саммари. Ничего не обобщай и не обезличивай (кроме дат, см. правило 1).

6. **Формат:** связный текст, организованный по узлам (каждый узел — секция с кратким заголовком и относительным временем). Объём: 800–1500 слов для спокойных периодов, до 2000 слов для периодов с множественными одновременными шоками.

7. В конце — блок «Общий контекст периода»: 3-5 предложений о том, каким был этот период в экономическом смысле, без прогнозов и оценок.

8. Не добавляй собственных интерпретаций и прогнозов за пределами того, что есть в исходных axis-саммари.
"""

META_USER = """Тематические таймлайны:

{axis_blocks}

Собери единый нарративный таймлайн по смысловым узлам."""


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


async def summarize_axis(client, axis: str, docs: list[dict], model: str, horizon_days: int) -> str:
    """Run axis summarization. Returns summary text."""
    t = time.time()
    formatted = format_for_prompt(docs)
    response = await _call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": AXIS_SYSTEM.format(axis=axis, horizon_days=horizon_days)},
                {"role": "user", "content": AXIS_USER.format(axis=axis, docs=formatted)},
            ],
            temperature=0.3,
        ),
        axis,
    )
    print(f"{axis}: {time.time() - t:.1f}s")
    return response.choices[0].message.content


async def summarize_meta(client, axis_summaries: dict[str, str], model: str) -> str:
    """Run meta summarization over axis summaries. Returns final summary text."""
    axis_blocks = "\n\n".join(
        f"=== Ось: {axis} ===\n\n{summary}"
        for axis, summary in axis_summaries.items()
    )
    t = time.time()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": META_SYSTEM},
            {"role": "user", "content": META_USER.format(axis_blocks=axis_blocks)},
        ],
        temperature=0.3,
    )
    print(f"\tmeta: {time.time() - t:.1f}s")
    return response.choices[0].message.content


async def run_summarize(
    client,
    retrieved: dict[str, list[dict]],
    model: str,
    horizon_days: int,
) -> str:
    """Full two-stage summarization pipeline. Returns meta summary text."""
    semaphore = asyncio.Semaphore(5)

    async def bounded_axis(axis, docs):
        async with semaphore:
            return await summarize_axis(client, axis, docs, model, horizon_days)

    axes = list(retrieved.keys())
    results = await asyncio.gather(
        *(bounded_axis(axis, retrieved[axis]) for axis in axes),
        return_exceptions=True,
    )

    failed = [(axes[i], r) for i, r in enumerate(results) if isinstance(r, Exception)]
    succeeded = {axes[i]: r for i, r in enumerate(results) if not isinstance(r, Exception)}

    if len(failed) >= 3:
        failed_names = ", ".join(name for name, _ in failed)
        raise RuntimeError(f"Too many axis failures ({len(failed)}/9): {failed_names}")
    if failed:
        failed_names = ", ".join(name for name, _ in failed)
        logger.warning("Axis failed but continuing: %s", failed_names)
        for name, _ in failed:
            succeeded[name] = "(нет данных по оси из-за ошибки провайдера)"

    axis_summaries = {axis: succeeded[axis] for axis in axes}
    return await summarize_meta(client, axis_summaries, model)
