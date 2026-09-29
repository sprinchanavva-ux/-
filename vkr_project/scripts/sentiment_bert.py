# -*- coding: utf-8 -*-
"""
Анализ тональности корпуса — через предобученную модель HuggingFace.
ВКР Спринчан И.А., §2.3

Почему этот вариант вместо dostoevsky: dostoevsky зависит от библиотеки
fasttext, которая требует компиляции C++-расширения — и эта сборка
ломается в текущей версии Python в Colab (несовместимость версий,
а не ошибка в коде). transformers распространяется в виде уже собранных
пакетов, компиляция не нужна — надёжнее для воспроизводимости.

Модель: cointegrated/rubert-tiny-sentiment-balanced — компактная (Tiny),
быстро работает даже на CPU, обучена именно на трёхклассовой русской
тональности (negative / neutral / positive), общедоступна на HuggingFace Hub.

ЗАПУСК: Google Colab.
Перед запуском выполните в отдельной ячейке:
    !pip install transformers sentencepiece

Загрузите corpus_processed.json (файл на 202 текста, полученный после
предобработки — там уже есть нужное поле raw_text) перед запуском.
"""

import json
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# 1. ЗАГРУЗКА МОДЕЛИ
# ---------------------------------------------------------------------------
MODEL_NAME = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

id2label = model.config.id2label
print(f"Модель загружена. Классы: {id2label}")

# ---------------------------------------------------------------------------
# 2. ЗАГРУЗКА КОРПУСА
# ---------------------------------------------------------------------------
with open('corpus_processed.json', encoding='utf-8') as f:
    corpus = json.load(f)

print(f"Загружено текстов: {len(corpus)}")

# ---------------------------------------------------------------------------
# 3. КЛАССИФИКАЦИЯ ТОНАЛЬНОСТИ
# ---------------------------------------------------------------------------
# Модель BERT-типа имеет ограничение на длину входа (обычно 512 токенов) —
# для текстов длиннее этого используем усечение (truncation=True).
# Это метод-ограничение, которое стоит явно оговорить в тексте параграфа:
# оценивается фактически начало текста, а не весь текст целиком, если
# он превышает лимит модели.

def predict_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    return {id2label[i]: round(probs[i], 4) for i in range(len(probs))}


for d in corpus:
    scores = predict_sentiment(d['raw_text'])
    d['sentiment_scores'] = scores
    d['sentiment_label'] = max(scores, key=scores.get)

# ---------------------------------------------------------------------------
# 4. ОБЩЕЕ РАСПРЕДЕЛЕНИЕ
# ---------------------------------------------------------------------------
label_counts = Counter(d['sentiment_label'] for d in corpus)
print("\n=== Распределение тональности по корпусу ===")
for label, count in label_counts.most_common():
    print(f"  {label}: {count} текстов ({100*count/len(corpus):.1f}%)")

# ---------------------------------------------------------------------------
# 5. ПО ЖАНРАМ
# ---------------------------------------------------------------------------
print("\n=== Тональность по жанрам ===")
genre_labels = defaultdict(list)
for d in corpus:
    genre_labels[d['genre']].append(d['sentiment_label'])
for genre, labels in sorted(genre_labels.items()):
    counts = Counter(labels)
    print(f"  {genre} (n={len(labels)}): {dict(counts)}")

# ---------------------------------------------------------------------------
# 6. ПО ИСТОЧНИКАМ
# ---------------------------------------------------------------------------
print("\n=== Тональность по источникам ===")
source_labels = defaultdict(list)
for d in corpus:
    source_labels[d['source']].append(d['sentiment_label'])
for source, labels in sorted(source_labels.items()):
    counts = Counter(labels)
    print(f"  {source} (n={len(labels)}): {dict(counts)}")

# ---------------------------------------------------------------------------
# 7. САМЫЕ ВЫРАЖЕННЫЕ ПРИМЕРЫ
# ---------------------------------------------------------------------------
def top_by_label(label, n=3):
    scored = [(d, d['sentiment_scores'].get(label, 0)) for d in corpus]
    scored.sort(key=lambda x: -x[1])
    return scored[:n]

print("\n=== Топ-3 самых позитивных ===")
for d, score in top_by_label('positive'):
    print(f"  id={d['id']} [{d['genre']}] score={score:.3f} — {d['raw_text'][:70]}...")

print("\n=== Топ-3 самых негативных ===")
for d, score in top_by_label('negative'):
    print(f"  id={d['id']} [{d['genre']}] score={score:.3f} — {d['raw_text'][:70]}...")

# ---------------------------------------------------------------------------
# 6b. ПРОВЕРКА НАДЁЖНОСТИ: тексты с низкой уверенностью модели
# ---------------------------------------------------------------------------
# Если top-класс набрал меньше 0.5 вероятности — модель фактически колеблется
# между двумя классами. Такие случаи стоит проверить вручную, а не принимать
# автоматическую метку как окончательную.
low_confidence = [d for d in corpus if max(d['sentiment_scores'].values()) < 0.5]
print(f"\n=== Текстов с неуверенным предсказанием (max prob < 0.5): {len(low_confidence)} из {len(corpus)} ===")
for d in low_confidence:
    print(f"  id={d['id']} [{d['genre']}] label={d['sentiment_label']} scores={d['sentiment_scores']}")

# Средняя уверенность отдельно по жанрам с малым n — предупреждение, если
# жанр представлен < 3 текстами, статистику по нему нельзя считать надёжной
print("\n=== Внимание: жанры с n < 3 (статистика по ним ненадёжна) ===")
genre_counts = Counter(d['genre'] for d in corpus)
for g, c in genre_counts.items():
    if c < 3:
        print(f"  {g}: n={c} — не использовать для содержательных выводов")

# ---------------------------------------------------------------------------
# 8. СОХРАНЕНИЕ
# ---------------------------------------------------------------------------
output = [
    {
        'id': d['id'], 'source': d['source'], 'genre': d['genre'],
        'sentiment_label': d['sentiment_label'],
        'sentiment_scores': d['sentiment_scores'],
    }
    for d in corpus
]
with open('sentiment_results_bert.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\nСохранено: sentiment_results_bert.json")

# from google.colab import files
# files.download('sentiment_results_bert.json')
