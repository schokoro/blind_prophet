import pytest

from amnesiac.preprocess import is_digest, normalize, process_message


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def test_normalize_strips_markdown_links():
    result = normalize("[текст](https://example.com)")
    assert result == "текст"
    assert "https://example.com" not in result


def test_normalize_strips_bare_urls():
    result = normalize("Ставка повышена. https://cbr.ru/press/ Подробности позже.")
    assert "https://cbr.ru/press/" not in result
    assert "Ставка повышена" in result


def test_normalize_strips_trailing_mention():
    # Trailing mention на отдельной строке должен быть удалён
    text = "ЦБ повысил ставку до 21%.\n@frank_media"
    result = normalize(text)
    assert "@frank_media" not in result
    assert "ЦБ повысил ставку" in result

    # Инлайн-упоминание должно быть сохранено
    inline = "Об этом сообщает @rbc_news в своём материале."
    result_inline = normalize(inline)
    assert "@rbc_news" in result_inline


def test_normalize_strips_hashtags():
    result = normalize("Новости #БанковскийСектор сегодня.")
    assert "#БанковскийСектор" not in result
    assert "Новости" in result


def test_normalize_joins_newlines():
    text = "Первая строка\nВторая строка\nТретья строка"
    result = normalize(text)
    assert result == "Первая строка. Вторая строка. Третья строка"


def test_normalize_strips_emoji():
    result = normalize("Рынок растёт 📈 сегодня")
    assert "📈" not in result
    assert "Рынок растёт" in result


def test_normalize_collapses_whitespace():
    result = normalize("Много   пробелов\tи\tтабов")
    assert "  " not in result
    assert "\t" not in result


# ---------------------------------------------------------------------------
# is_digest
# ---------------------------------------------------------------------------

def test_is_digest_by_hashtag():
    assert is_digest("Подборка материалов за неделю #digest")
    assert is_digest("Подборка #DIGEST за неделю")
    assert is_digest("Лучшее за неделю #дайджест")


def test_is_digest_by_list_markers():
    text = (
        "Главные новости:\n"
        "👉 ЦБ повысил ставку\n"
        "👉 Инфляция ускорилась\n"
        "👉 Рубль ослаб\n"
        "👉 Нефть подорожала\n"
    )
    assert is_digest(text)


def test_is_digest_by_url_density():
    text = (
        "Читайте подробнее:\n"
        "[Статья 1](https://example.com/1)\n"
        "[Статья 2](https://example.com/2)\n"
        "[Статья 3](https://example.com/3)\n"
        "Конец подборки\n"
    )
    assert is_digest(text)


def test_is_digest_normal_text():
    text = (
        "Банк России на заседании совета директоров принял решение повысить "
        "ключевую ставку до 21% годовых. Регулятор объяснил это ускорением "
        "инфляции и необходимостью вернуть её к целевому уровню."
    )
    assert not is_digest(text)


# ---------------------------------------------------------------------------
# process_message
# ---------------------------------------------------------------------------

def test_process_message_valid():
    text = (
        "Банк России повысил ключевую ставку до 21% годовых. "
        "Решение принято на фоне ускорения инфляции. "
        "Регулятор планирует удерживать жёсткую денежно-кредитную политику."
    )
    processed, is_valid = process_message(text)
    assert is_valid is True
    assert len(processed) >= 50


def test_process_message_digest():
    text = (
        "Дайджест недели #digest\n"
        "👉 ЦБ поднял ставку\n"
        "👉 Инфляция растёт\n"
        "👉 Рубль слабеет\n"
    )
    processed, is_valid = process_message(text)
    assert is_valid is False
    # нормализация всё равно применяется
    assert isinstance(processed, str)


def test_process_message_too_short():
    text = "Коротко."
    processed, is_valid = process_message(text, min_length=50)
    assert is_valid is False
    assert processed == "Коротко."
