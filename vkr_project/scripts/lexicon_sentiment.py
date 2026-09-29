# -*- coding: utf-8 -*-
"""
Словарный (лексиконный) анализ тональности —
запасной метод, не требующий внешних библиотек/моделей.
"""
import re
import json
from collections import Counter, defaultdict

# --- Компактный лексикон оценочной лексики -------------------------------
POSITIVE = {
    'хороший','отличный','прекрасный','великолепный','восхитительный','красивый',
    'любимый','любить','нравиться','понравиться','эффектный','эффектно','стильный',
    'элегантный','изящный','гармоничный','удобный','комфортный','практичный',
    'универсальный','модный','актуальный','трендовый','культовый','легендарный',
    'идеальный','лучший','успех','успешный','победа','триумф','гордость','радость',
    'счастливый','вдохновлять','вдохновение','восторг','обожать','шикарный',
    'роскошный','выразительный','яркий','свежий','новый','инновационный',
    'смелый','дерзкий','уверенный','качественный','долговечный','любимец',
    'нравится','понравился','понравилась','приятный','приятно','достойный',
    'выигрышный','привлекательный','притягательный','желанный','хочется',
}

NEGATIVE = {
    'плохой','ужасный','отвратительный','скучный','провал','провальный','неудачный',
    'вульгарный','безвкусный','дешёвый','кричащий','нелепый','странный','неудобный',
    'некомфортный','непрактичный','устаревший','banальный','вторичный','подражание',
    'критика','критиковать','критиковали','осуждать','осуждали','возмущение',
    'возмутились','негатив','негативный','разочарование','разочаровать','провалиться',
    'кризис','падение','упасть','упал','упала','снижение','смерть','умирать','гибель',
    'проблема','проблемный','сложный','трудный','неприятный','отталкивающий',
    'кощунство','скандал','скандальный','эпатаж','эпатажный','вызывающий',
    'жалкий','посредственный','грустный','печальный','тревога','тревожный',
    'потерять','потеря','ошибка','ошибочный','неуместный','пошлость','пошлый',
}

NEGATORS = {'не', 'ни', 'нет', 'без'}
WINDOW = 3  # ищем отрицание в пределах 3 слов перед оценочным словом


def tokenize(text: str):
    text = text.lower()
    return re.findall(r'[а-яёa-z]+', text)


def score_document(text: str):
    tokens = tokenize(text)
    pos, neg = 0, 0
    for i, tok in enumerate(tokens):
        negated = any(t in NEGATORS for t in tokens[max(0, i - WINDOW):i])
        if tok in POSITIVE:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        elif tok in NEGATIVE:
            pos += 1 if negated else 0
            neg += 0 if negated else 1
    total = pos + neg
    MIN_COVERAGE = 3  # минимум оценочных слов, чтобы считать балл надёжным
    reliable = total >= MIN_COVERAGE
    if total == 0:
        return {'label': 'neutral', 'pos': pos, 'neg': neg, 'score': 0.0,
                'coverage': total, 'reliable': reliable}
    score = (pos - neg) / total
    if not reliable:
        label = 'neutral'  # недостаточно сигналов — не делаем категоричных выводов
    elif score > 0.2:
        label = 'positive'
    elif score < -0.2:
        label = 'negative'
    else:
        label = 'neutral'
    return {'label': label, 'pos': pos, 'neg': neg, 'score': round(score, 3),
            'coverage': total, 'reliable': reliable}


if __name__ == '__main__':
    with open('corpus_processed.json', encoding='utf-8') as f:
        corpus = json.load(f)

    for d in corpus:
        d['sentiment'] = score_document(d['raw_text'])

    labels = Counter(d['sentiment']['label'] for d in corpus)
    print("=== Общее распределение тональности ===")
    for lbl, cnt in labels.most_common():
        print(f"  {lbl}: {cnt} ({100*cnt/len(corpus):.1f}%)")

    print("\n=== По жанрам ===")
    genre_map = defaultdict(list)
    for d in corpus:
        genre_map[d['genre']].append(d['sentiment']['label'])
    for genre, labs in sorted(genre_map.items()):
        c = Counter(labs)
        print(f"  {genre} (n={len(labs)}): {dict(c)}")

    reliable_docs = [d for d in corpus if d['sentiment']['reliable']]
    print(f"\nТекстов с надёжным покрытием лексикона (>= {3} оценочных слов): "
          f"{len(reliable_docs)} из {len(corpus)} ({100*len(reliable_docs)/len(corpus):.1f}%)")

    print("\n=== Топ-3 позитивных (среди надёжных) ===")
    top_pos = sorted(reliable_docs, key=lambda d: -d['sentiment']['score'])[:3]
    for d in top_pos:
        s = d['sentiment']
        print(f"  id={d['id']} [{d['genre']}] score={s['score']} (покрытие={s['coverage']}) — {d['raw_text'][:70]}...")

    print("\n=== Топ-3 негативных (среди надёжных) ===")
    top_neg = sorted(reliable_docs, key=lambda d: d['sentiment']['score'])[:3]
    for d in top_neg:
        s = d['sentiment']
        print(f"  id={d['id']} [{d['genre']}] score={s['score']} (покрытие={s['coverage']}) — {d['raw_text'][:70]}...")
