# -*- coding: utf-8 -*-
"""
Тематическое моделирование корпуса (§2.2)
ВКР Спринчан И.А.

ЗАПУСК: после preprocessing.py — требуется файл corpus_processed.json
в той же папке.

Установка: pip install scikit-learn (обычно уже есть в Colab)
"""

import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

INPUT_FILE = "corpus_processed.json"
N_TOPICS = 8  # см. §2.2: число тем подобрано под финальный объём корпуса (202 текста)
N_TOP_WORDS = 12

with open(INPUT_FILE, encoding="utf-8") as f:
    corpus = json.load(f)

documents = [" ".join(d["lemmas"]) for d in corpus]

# token_pattern сохраняет дефисные составные слова целиком
# (иначе "весна-лето" разбивается на "весна" + "то" — см. разбор в §2.2)
vectorizer = CountVectorizer(min_df=3, max_df=0.5, token_pattern=r"(?u)\b[\w-]+\b")
doc_term_matrix = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()
print(f"Размер словаря: {len(vocab)}")

lda = LatentDirichletAllocation(
    n_components=N_TOPICS, random_state=42, max_iter=25, learning_method="batch"
)
doc_topic_matrix = lda.fit_transform(doc_term_matrix)

topics_words = {}
print(f"\n=== {N_TOPICS} ТЕМ ===")
for topic_idx, topic in enumerate(lda.components_):
    top_indices = topic.argsort()[-N_TOP_WORDS:][::-1]
    top_words = [vocab[i] for i in top_indices]
    topics_words[topic_idx] = top_words
    print(f"Тема {topic_idx + 1}: {', '.join(top_words)}")

dominant_topics = np.argmax(doc_topic_matrix, axis=1)
for i, d in enumerate(corpus):
    d["dominant_topic"] = int(dominant_topics[i]) + 1

print("\n=== Размер тем ===")
for t, c in sorted(Counter(dominant_topics.tolist()).items()):
    print(f"  Тема {t + 1}: {c} текстов")

print("\n=== Тема по жанру (n >= 5) ===")
genre_topic = {}
for d in corpus:
    genre_topic.setdefault(d["genre"], []).append(d["dominant_topic"])
for genre, topics in sorted(genre_topic.items(), key=lambda x: -len(x[1])):
    if len(topics) < 5:
        continue
    counts = Counter(topics)
    print(f"  {genre} (n={len(topics)}): {dict(sorted(counts.items()))}")

result = {
    "n_topics": N_TOPICS,
    "topics": {f"Тема {k + 1}": v for k, v in topics_words.items()},
    "documents": [
        {"id": d["id"], "source": d["source"], "genre": d["genre"], "dominant_topic": d["dominant_topic"]}
        for d in corpus
    ],
}
with open("topic_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\nСохранено: topic_results.json")
