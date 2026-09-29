# -*- coding: utf-8 -*-
"""Проверка устойчивости LDA (§2.6): 10 запусков с разными random_state, сравнение по ARI и NMI
относительно основного запуска (random_state=42) и совпадение с жанровой разметкой (purity)."""
import json, itertools
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
corpus = json.load(open("corpus_processed.json", encoding="utf-8"))
docs = [" ".join(d["lemmas"]) for d in corpus]
X = CountVectorizer(min_df=3, max_df=0.5, token_pattern=r"(?u)\b[\w-]+\b").fit_transform(docs)
genres = [d["genre"] for d in corpus]
def run(seed, k=8):
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=25, learning_method="batch")
    return np.argmax(lda.fit_transform(X), axis=1)
def purity(labels):
    tot = 0
    for t in set(labels):
        idx = [i for i, l in enumerate(labels) if l == t]
        tot += Counter(genres[i] for i in idx).most_common(1)[0][1]
    return tot / len(labels)
base = run(42)
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
runs = {s: run(s) for s in seeds}
ari = [adjusted_rand_score(base, r) for r in runs.values()]
nmi = [normalized_mutual_info_score(base, r) for r in runs.values()]
pair = [adjusted_rand_score(a, b) for a, b in itertools.combinations(runs.values(), 2)]
# случайная перестановка жанров как нулевая модель для purity
rng = np.random.default_rng(0)
null = []
for _ in range(200):
    g = rng.permutation(genres); tot = 0
    for t in set(base):
        idx = np.where(base == t)[0]; tot += Counter(g[i] for i in idx).most_common(1)[0][1]
    null.append(tot / len(base))
res = {"ari_vs_base_mean": round(float(np.mean(ari)), 3), "ari_vs_base_min": round(float(min(ari)), 3), "ari_vs_base_max": round(float(max(ari)), 3),
       "nmi_vs_base_mean": round(float(np.mean(nmi)), 3), "ari_pairwise_mean": round(float(np.mean(pair)), 3),
       "purity_base": round(purity(base), 3), "purity_runs_mean": round(float(np.mean([purity(r) for r in runs.values()])), 3),
       "purity_null_mean": round(float(np.mean(null)), 3), "purity_null_p95": round(float(np.percentile(null, 95)), 3)}
# по числу тем
res["k_scan"] = {}
for k in (4, 6, 8, 10, 12):
    lda = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=25, learning_method="batch").fit(X)
    res["k_scan"][k] = {"perplexity": round(float(lda.perplexity(X)), 1)}
json.dump(res, open("robustness_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
