# -*- coding: utf-8 -*-
"""
Предобработка корпуса цифрового модного дискурса
ВКР Спринчан И.А., §2.2

ЗАПУСК: Google Colab.
Перед первым запуском выполнить в отдельной ячейке:
    !pip install pymorphy3 pymorphy3-dicts-ru razdel nltk

Загрузите файл corpus.json в Colab (слева на панели Files -> Upload),
либо через:
    from google.colab import files
    uploaded = files.upload()
"""

import json
import re
from collections import Counter

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

import pymorphy3
from razdel import tokenize as razdel_tokenize

morph = pymorphy3.MorphAnalyzer()

# ---------------------------------------------------------------------------
# 1. ЗАГРУЗКА КОРПУСА
# ---------------------------------------------------------------------------
with open('corpus.json', encoding='utf-8') as f:
    corpus = json.load(f)

print(f"Загружено текстов: {len(corpus)}")
print(f"Источники: {sorted(set(d['source'] for d in corpus))}")
print(f"Жанры: {sorted(set(d['genre'] for d in corpus))}")

# ---------------------------------------------------------------------------
# 2. СТОП-СЛОВА
# ---------------------------------------------------------------------------
# Стандартный список nltk для русского языка (предлоги, союзы, частицы и т.д.)
russian_stopwords = set(stopwords.words('russian'))

# Дополнительный список: высокочастотные лексемы, не несущие содержательной
# нагрузки именно в контексте модного дискурса (местоимения, служебные глаголы,
# вводные слова, которые NLTK не считает стоп-словами по умолчанию)
custom_stopwords = {
    'это', 'весь', 'свой', 'который', 'мочь', 'также', 'ещё', 'уже',
    'самый', 'стать', 'один', 'два', 'быть', 'говорить', 'год', 'новый',
}

all_stopwords = russian_stopwords | custom_stopwords

# ---------------------------------------------------------------------------
# 3. ФУНКЦИИ ПРЕДОБРАБОТКИ
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Базовая очистка текста перед токенизацией:
    - приведение к нижнему регистру
    - удаление пунктуации (кроме внутрисловного дефиса: важно для таких
      единиц модного дискурса, как «сумка-мешочек», «total-white»)
    - удаление цифр (частотны в исходных текстах: годы коллекций,
      цены, проценты — не являются предметом лексического анализа)
    - схлопывание повторных пробелов
    """
    text = text.lower()
    text = re.sub(r'[^\w\sёЁ-]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)  # дефис только между буквами
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """
    Токенизация с помощью razdel — токенизатора, учитывающего специфику
    русской пунктуации и морфологии лучше, чем базовый nltk.word_tokenize.
    """
    return [tok.text for tok in razdel_tokenize(text)]


def lemmatize(tokens: list[str]) -> list[str]:
    """
    Лемматизация через pymorphy3. Дополнительно отсекаются:
    - токены короче 3 символов (в основном шум после очистки)
    - токены из списка стоп-слов
    - леммы, попавшие в список стоп-слов после нормализации
      (некоторые словоформы стоп-слов pymorphy3 не всегда узнаёт исходно)
    """
    lemmas = []
    for token in tokens:
        if len(token) < 3 or token in all_stopwords:
            continue
        lemma = morph.parse(token)[0].normal_form
        if lemma in all_stopwords:
            continue
        lemmas.append(lemma)
    return lemmas


# ---------------------------------------------------------------------------
# 4. ОБРАБОТКА КОРПУСА
# ---------------------------------------------------------------------------
processed_corpus = []

for doc in corpus:
    raw_text = doc['text']
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    lemmas = lemmatize(tokens)

    processed_corpus.append({
        'id': doc['id'],
        'source': doc['source'],
        'genre': doc['genre'],
        'author': doc.get('author', ''),
        'date': doc.get('date', ''),
        'raw_text': raw_text,
        'tokens': tokens,
        'lemmas': lemmas,
        'token_count': len(tokens),
        'lemma_count': len(lemmas),
    })

# ---------------------------------------------------------------------------
# 5. СОХРАНЕНИЕ РЕЗУЛЬТАТА
# ---------------------------------------------------------------------------
with open('corpus_processed.json', 'w', encoding='utf-8') as f:
    json.dump(processed_corpus, f, ensure_ascii=False, indent=2)

print("\nСохранено: corpus_processed.json")

# Для скачивания результата на компьютер в Colab раскомментируйте:
# from google.colab import files
# files.download('corpus_processed.json')

# ---------------------------------------------------------------------------
# 6. ДИАГНОСТИКА (для проверки качества предобработки и текста §2.2)
# ---------------------------------------------------------------------------
total_tokens = sum(d['token_count'] for d in processed_corpus)
total_lemmas = sum(d['lemma_count'] for d in processed_corpus)
unique_lemmas = len(set(l for d in processed_corpus for l in d['lemmas']))

print(f"\nВсего токенов в корпусе (до удаления стоп-слов): {total_tokens}")
print(f"Всего лемм (после очистки и удаления стоп-слов): {total_lemmas}")
print(f"Уникальных лемм (словарь корпуса): {unique_lemmas}")
print(f"Лексическое разнообразие (type-token ratio): {unique_lemmas / total_lemmas:.4f}")

all_lemmas = [lemma for d in processed_corpus for lemma in d['lemmas']]
freq = Counter(all_lemmas)

print("\nТоп-30 самых частотных лемм корпуса:")
for lemma, count in freq.most_common(30):
    print(f"  {lemma}: {count}")

# Частотность по источникам — полезно для проверки, не «перекошен» ли
# итоговый частотный список в сторону одного издания
print("\nСредняя длина текста (в токенах) по источникам:")
by_source = {}
for d in processed_corpus:
    by_source.setdefault(d['source'], []).append(d['token_count'])
for source, counts in by_source.items():
    print(f"  {source}: {sum(counts) / len(counts):.1f} токенов/текст (n={len(counts)})")
