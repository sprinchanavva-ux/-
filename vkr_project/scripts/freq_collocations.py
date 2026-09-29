# -*- coding: utf-8 -*-
"""
Частотный и коллокационный анализ корпуса (§2.2)
ВКР Спринчан И.А.

ЗАПУСК: после preprocessing.py — требуется файл corpus_processed.json
в той же папке (результат работы preprocessing.py).

Не требует установки библиотек сверх стандартной библиотеки Python.
"""

import json
import math
from collections import Counter

INPUT_FILE = "corpus_processed.json"
EXTRA_STOPWORDS = {"именно"}  # добавлено по итогам ручной проверки топ-30

with open(INPUT_FILE, encoding="utf-8") as f:
    corpus = json.load(f)

for d in corpus:
    d["lemmas"] = [l for l in d["lemmas"] if l not in EXTRA_STOPWORDS]
    d["lemma_count"] = len(d["lemmas"])

# ---------------------------------------------------------------------------
# 1. ЧАСТОТНЫЙ АНАЛИЗ
# ---------------------------------------------------------------------------
all_lemmas = [l for d in corpus for l in d["lemmas"]]
total_lemmas = len(all_lemmas)
unique_lemmas = len(set(all_lemmas))
hapax = sum(1 for l, c in Counter(all_lemmas).items() if c == 1)

print("=== ОБЩАЯ СТАТИСТИКА ===")
print(f"Текстов: {len(corpus)}")
print(f"Токенов: {sum(d['token_count'] for d in corpus)}")
print(f"Лемм: {total_lemmas}")
print(f"Уникальных лемм: {unique_lemmas}")
print(f"TTR: {unique_lemmas/total_lemmas:.4f}")
print(f"Хапаксов: {hapax} ({100*hapax/unique_lemmas:.1f}% словаря)")

freq = Counter(all_lemmas)
print("\n=== ТОП-30 ЛЕММ ===")
for lemma, cnt in freq.most_common(30):
    print(f"  {lemma}: {cnt}")

# ---------------------------------------------------------------------------
# 2. КОЛЛОКАЦИОННЫЙ АНАЛИЗ (PMI)
# ---------------------------------------------------------------------------
MIN_FREQ = 5  # порог частоты биграммы для расчёта PMI

bigram_counts = Counter()
unigram_counts = Counter()
total_bigrams = 0
for d in corpus:
    lemmas = d["lemmas"]
    unigram_counts.update(lemmas)
    for i in range(len(lemmas) - 1):
        bigram_counts[(lemmas[i], lemmas[i + 1])] += 1
        total_bigrams += 1
total_unigrams = sum(unigram_counts.values())

pmi_scores = []
for (w1, w2), count in bigram_counts.items():
    if count < MIN_FREQ:
        continue
    p_w1w2 = count / total_bigrams
    p_w1 = unigram_counts[w1] / total_unigrams
    p_w2 = unigram_counts[w2] / total_unigrams
    pmi = math.log2(p_w1w2 / (p_w1 * p_w2))
    pmi_scores.append(((w1, w2), count, round(pmi, 2)))
pmi_scores.sort(key=lambda x: -x[2])

print(f"\n=== КОЛЛОКАЦИИ (биграммы с частотой >= {MIN_FREQ}) ===")
print(f"Всего биграмм: {total_bigrams}, уникальных: {len(bigram_counts)}")
print(f"Прошли порог частоты: {len(pmi_scores)}")

print("\n--- ТОП-25 ПО PMI ---")
for (w1, w2), c, p in pmi_scores[:25]:
    print(f"  {w1} {w2}: частота={c}, PMI={p}")

print("\n--- ТОП-15 ПО АБСОЛЮТНОЙ ЧАСТОТЕ ---")
for (w1, w2), c in bigram_counts.most_common(15):
    print(f"  {w1} {w2}: {c}")

# ---------------------------------------------------------------------------
# 3. СОХРАНЕНИЕ
# ---------------------------------------------------------------------------
result = {
    "total_tokens": sum(d["token_count"] for d in corpus),
    "total_lemmas": total_lemmas,
    "unique_lemmas": unique_lemmas,
    "ttr": round(unique_lemmas / total_lemmas, 4),
    "hapax_share": round(hapax / unique_lemmas, 4),
    "top50_lemmas": freq.most_common(50),
    "top_pmi": [{"bigram": f"{w1} {w2}", "freq": c, "pmi": p} for (w1, w2), c, p in pmi_scores[:40]],
    "top_freq_bigrams": [{"bigram": f"{w1} {w2}", "freq": c} for (w1, w2), c in bigram_counts.most_common(20)],
}
with open("freq_colloc_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\nСохранено: freq_colloc_results.json")
