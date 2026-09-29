# -*- coding: utf-8 -*-
"""
Извлечение подлинного текста статей по уже собранному списку URL.
ВКР Спринчан И.А. — замена ручных пересказов на автоматически
извлечённый текст (исправление методологической проблемы, описанной
в §2.1: раньше текст записывался как пересказ, а не как подлинный
извлечённый текст).

ЗАПУСК: локально, на компьютере с интернетом.
    pip install trafilatura tqdm

Логика:
1. Читаем urls_to_scrape.json — список из id/source/genre/author/date/
   title/url, собранный вручную по ходу работы (без текста статьи).
2. Для каждого URL trafilatura автоматически извлекает чистый текст
   статьи из HTML — без ручного разбора вёрстки под каждый сайт.
3. Результат — corpus.json с теми же метаданными, что и раньше,
   но с полем text = подлинный, автоматически извлечённый текст
   (не пересказ).
4. Записи, которые не удалось извлечь (сайт недоступен, изменилась
   вёрстка, статья удалена), помечаются отдельно и печатаются в конце —
   их нужно будет проверить вручную.
"""

import json
import time
import trafilatura
from tqdm import tqdm


INPUT_FILE = "urls_to_scrape.json"
OUTPUT_FILE = "corpus.json"
FAILED_FILE = "failed_urls.json"

MIN_TEXT_LENGTH = 200  # символов — короче обычно значит "не то извлеклось"


def fetch_clean_text(url: str):
    """Скачивает страницу и извлекает из неё чистый текст статьи."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None, "не удалось скачать страницу"

    result = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        output_format="json",
        with_metadata=True,
    )
    if not result:
        return None, "trafilatura не смогла извлечь текст"

    parsed = json.loads(result)
    text = parsed.get("text", "")
    if len(text) < MIN_TEXT_LENGTH:
        return None, f"извлечённый текст слишком короткий ({len(text)} симв.)"

    return text, None


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        items = json.load(f)

    print(f"Всего URL для обработки: {len(items)}")

    corpus = []
    failed = []

    for item in tqdm(items, desc="Извлечение текста"):
        url = item["url"]
        text, error = fetch_clean_text(url)

        if text is None:
            failed.append({**item, "error": error})
            print(f"  [!] id={item['id']} ({item['source']}): {error}")
            time.sleep(0.5)
            continue

        entry = dict(item)  # копируем id/source/genre/author/date/title/url
        entry["text"] = text
        corpus.append(entry)

        time.sleep(0.5)  # вежливая пауза между запросами

    corpus.sort(key=lambda d: d["id"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    with open(FAILED_FILE, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    print(f"\n=== ИТОГ ===")
    print(f"Успешно извлечено: {len(corpus)} из {len(items)}")
    print(f"Не удалось извлечь: {len(failed)} — см. {FAILED_FILE}")
    print(f"Результат сохранён в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
