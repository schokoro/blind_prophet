import re

import emoji

# Маркеры списков (эмодзи и символьные)
_LIST_MARKER_RE = re.compile(
    r"^(?:👉|🔹|🔸|▪|➡|•|-|\*|\d+[.)]\s)"
)

# URL-паттерны
_BARE_URL_RE = re.compile(r"https?://\S+")
_MD_LINK_RE = re.compile(r"\[(.+?)\]\(.+?\)")
_MD_OR_BARE_URL_RE = re.compile(r"https?://\S+|\[.+?\]\(.+?\)")

# @упоминания
_MENTION_RE = re.compile(r"@\w+")
# Строка, состоящая только из @mention (с опциональными пробелами/пунктуацией)
_MENTION_ONLY_LINE_RE = re.compile(r"^\s*[@\w\s,;]+$")


def is_digest(text: str) -> bool:
    """Определяет, является ли пост дайджестом.

    Работает на RAW-тексте (до нормализации), потому что эмодзи и markdown
    являются структурными сигналами.
    """
    # По хэштегу
    if re.search(r"#digest|#дайджест", text, re.IGNORECASE):
        return True

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return False

    # По доле строк с маркерами списка
    list_lines = sum(1 for l in non_empty if _LIST_MARKER_RE.match(l.strip()))
    if list_lines / len(non_empty) > 0.5:
        return True

    # По плотности URL
    url_lines = sum(1 for l in non_empty if _MD_OR_BARE_URL_RE.search(l))
    if url_lines / len(non_empty) > 0.4:
        return True

    return False


def normalize(text: str) -> str:
    """Нормализует текст сообщения.

    Шаги (по порядку):
    a) Markdown-ссылки → anchor text
    b) Голые URL → удалить
    c) Trailing @mentions → удалить
    d) #hashtags → удалить
    e) Соединить строки через '. '
    f) Удалить эмодзи
    g) Схлопнуть пробелы
    """
    # a) [anchor](url) → anchor
    text = _MD_LINK_RE.sub(r"\1", text)

    # b) Голые URL
    text = _BARE_URL_RE.sub("", text)

    # c) Trailing @mentions
    lines = text.splitlines()
    # Убираем строки, состоящие исключительно из @mention-токенов
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Строка является "mention-only", если все токены — это @mention или пунктуация/пробел
        if stripped and re.fullmatch(r"[\s,;.]*(@\w+[\s,;.]*)+", stripped):
            continue
        cleaned_lines.append(line)
    lines = cleaned_lines

    # Убираем trailing @mentions из последней непустой строки
    if lines:
        last_idx = len(lines) - 1
        while last_idx >= 0 and not lines[last_idx].strip():
            last_idx -= 1
        if last_idx >= 0:
            lines[last_idx] = re.sub(r"(\s+@\w+)+\s*$", "", lines[last_idx])

    text = "\n".join(lines)

    # d) #hashtags
    text = re.sub(r"#\w+", "", text)

    # e) Соединить строки
    parts = [l.strip() for l in text.splitlines() if l.strip()]
    text = ". ".join(parts)

    # f) Эмодзи
    text = emoji.replace_emoji(text, replace="")

    # g) Схлопнуть пробелы
    text = re.sub(r"[ \t]+", " ", text).strip()

    return text


def process_message(text: str, min_length: int = 50) -> tuple[str, bool]:
    """Оркестрирует препроцессинг одного сообщения.

    Returns:
        (processed_text, is_valid)
        is_valid=False если текст — дайджест или слишком короткий после нормализации.
    """
    if is_digest(text):
        return normalize(text), False

    cleaned = normalize(text)

    if len(cleaned) < min_length:
        return cleaned, False

    return cleaned, True
