from .anxious_consumer import PROMPT as ANXIOUS_CONSUMER_PROMPT
from .mid_manager import PROMPT as MID_MANAGER_PROMPT
from .pensioner import PROMPT as PENSIONER_PROMPT
from .small_business_owner import PROMPT as SMALL_BUSINESS_OWNER_PROMPT
from .young_urban import PROMPT as YOUNG_URBAN_PROMPT

TRAILING_PROMPT = """Прочитай новостную сводку за прошедшие две недели. На сколько процентов, по твоему мнению, вырастут цены в России в следующие 12 месяцев?

Ответь одним числом — процентом годового роста цен. Не объясняй ход рассуждений, не перечисляй категории. Только число с одной цифрой после запятой, например: 12.5"""

PERSONA_BODIES: dict[str, str] = {
    "mid_manager": MID_MANAGER_PROMPT,
    "young_urban": YOUNG_URBAN_PROMPT,
    "anxious_consumer": ANXIOUS_CONSUMER_PROMPT,
    "pensioner": PENSIONER_PROMPT,
    "small_business_owner": SMALL_BUSINESS_OWNER_PROMPT,
}

PERSONAS: dict[str, str] = {
    name: f"{body}\n\n{TRAILING_PROMPT}" for name, body in PERSONA_BODIES.items()
}


def get_persona(name: str) -> str:
    return PERSONAS[name]


__all__ = ["PERSONAS", "PERSONA_BODIES", "TRAILING_PROMPT", "get_persona"]
