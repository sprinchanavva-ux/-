# -*- coding: utf-8 -*-
"""
Стилометрический анализ по жанрам (§2.2)
ВКР Спринчан И.А.

ЗАПУСК: после preprocessing.py — требуется файл corpus_processed.json
в той же папке. Не требует установки библиотек.
"""

import json
import re
from collections import defaultdict

INPUT_FILE = "corpus_processed.json"
MIN_GENRE_N = 5  # жанры с меньшим числом текстов не считаются надёжными


def split_sentences(text: str):
    text = re.sub(r"\b([а-яё])\.\s*([а-яё])\.", r"\1_\2_", text, flags=re.IGNORECASE)
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def count_words(s: str) -> int:
    return len(re.findall(r"[а-яёa-z]+", s.lower()))


with open(INPUT_FILE, encoding="utf-8") as f:
    corpus = json.load(f)

for d in corpus:
    lens = [count_words(s) for s in split_sentences(d["raw_text"])]
    lens = [l for l in lens if l > 0]
    d["avg_sentence_length"] = round(sum(lens) / len(lens), 2) if lens else 0
    d["lexical_density"] = round(d["lemma_count"] / d["token_count"], 3) if d["token_count"] else 0

genre_groups = defaultdict(list)
for d in corpus:
    genre_groups[d["genre"]].append(d)

print(f"{'Жанр':<20} {'n':>3} {'ДлПредл':>8} {'ЛексПлотн':>10} {'TTR':>7}")
genre_stats = {}
for genre, docs in sorted(genre_groups.items(), key=lambda x: -len(x[1])):
    n = len(docs)
    if n < MIN_GENRE_N:
        continue
    avg_sent = sum(d["avg_sentence_length"] for d in docs) / n
    avg_lex = sum(d["lexical_density"] for d in docs) / n
    all_l = [l for d in docs for l in d["lemmas"]]
    ttr = round(len(set(all_l)) / len(all_l), 3) if all_l else 0
    genre_stats[genre] = {
        "n": n,
        "avg_sentence_length": round(avg_sent, 2),
        "lexical_density": round(avg_lex, 3),
        "ttr": ttr,
    }
    print(f"{genre:<20} {n:>3} {avg_sent:>8.2f} {avg_lex:>10.3f} {ttr:>7.3f}")

with open("stylometry_results.json", "w", encoding="utf-8") as f:
    json.dump(genre_stats, f, ensure_ascii=False, indent=2)

print("\nСохранено: stylometry_results.json")
