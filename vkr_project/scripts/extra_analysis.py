# -*- coding: utf-8 -*-
"""
Дополнительный анализ корпуса для расширенной версии Главы 2 (§2.2, §2.5).
ЗАПУСК: после preprocessing.py; нужны corpus.json, corpus_processed.json,
freq_colloc_results.json, stylometry_results.json, topic_results.json в той же папке.
Зависимости: numpy, matplotlib, scipy. Результат: extra_results.json, fig*.png
"""
import json, math, re, statistics as st
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
GREY = "#4a4a4a"; ACC = "#2f5d8a"; LIGHT = "#9db7d1"

corpus = json.load(open("corpus_processed.json", encoding="utf-8"))
raw = {d["id"]: d for d in json.load(open("corpus.json", encoding="utf-8"))}
topics = {d["id"]: d["dominant_topic"] for d in json.load(open("topic_results.json", encoding="utf-8"))["documents"]}
STOP_EXTRA = {"именно"}
for d in corpus:
    d["lemmas"] = [l for l in d["lemmas"] if l not in STOP_EXTRA]
    d["words"] = len(d["raw_text"].split())            # длина текста в словах (по пробелам, как в §2.1)
    d["alpha"] = len(re.findall(r"[А-Яа-яЁёA-Za-z]+", d["raw_text"]))  # буквенные токены
    d["topic"] = topics[d["id"]]
    # укрупнённый жанр: основная часть до «/»
    d["genre_main"] = d["genre"]

SRC = {"theblueprint.ru": "The Blueprint", "thesymbol.ru": "SYMBOL", "mnenie-red.ru": "«Мнение редакции»",
       "posta-magazine.ru": "Posta-Magazine", "marieclaire.ru": "MarieClaire", "moskvichka.ru": "Moskvichka",
       "snob.ru": "Snob", "frwf.ru": "FRWF", "thevoicemag.ru": "The Voice"}
R = {}

def sent_lens(text):
    t = re.sub(r"\b([а-яё])\.\s*([а-яё])\.", r"\1_\2_", text, flags=re.I)
    s = [x for x in re.split(r"(?<=[.!?])\s+", t) if x.strip()]
    return [len(re.findall(r"[а-яёa-z]+", x.lower())) for x in s if re.findall(r"[а-яёa-z]+", x.lower())]

# 1. паспорт корпуса по источникам
by_src = defaultdict(list)
for d in corpus: by_src[d["source"]].append(d)
R["sources"] = []
for s, docs in sorted(by_src.items(), key=lambda x: -len(x[1])):
    lem = [l for d in docs for l in d["lemmas"]]
    sl = [x for d in docs for x in sent_lens(d["raw_text"])]
    R["sources"].append({"source": SRC[s], "n": len(docs), "share": round(100 * len(docs) / len(corpus), 1),
        "words": sum(d["words"] for d in docs), "mean_words": round(st.mean(d["words"] for d in docs)),
        "median_words": round(st.median(d["words"] for d in docs)),
        "ttr": round(len(set(lem)) / len(lem), 3), "sent_len": round(st.mean(sl), 1),
        "genres": len({d["genre_main"] for d in docs})})

# 2. жанры (все, укрупнённо)
gm = defaultdict(list)
for d in corpus: gm[d["genre_main"]].append(d)
R["genres_main"] = [{"genre": g, "n": len(v), "share": round(100 * len(v) / len(corpus), 1),
                     "mean_words": round(st.mean(x["words"] for x in v))} for g, v in sorted(gm.items(), key=lambda x: -len(x[1]))]
R["genres_full"] = Counter(d["genre"] for d in corpus).most_common()

# 3. длина текстов
w = [d["words"] for d in corpus]
R["length"] = {"total_words": sum(w), "mean": round(st.mean(w)), "median": st.median(w), "min": min(w), "max": max(w),
               "q1": float(np.percentile(w, 25)), "q3": float(np.percentile(w, 75)),
               "lt400": sum(x < 400 for x in w), "400_1000": sum(400 <= x <= 1000 for x in w), "gt1000": sum(x > 1000 for x in w)}
gl = {g: [x["words"] for x in v] for g, v in gm.items() if len(v) >= 5}
R["genre_lengths"] = {g: {"n": len(v), "mean": round(st.mean(v)), "median": st.median(v), "max": max(v)} for g, v in gl.items()}

# 4. Частотность: топ-30 с относительной частотой; POS не размечалась
lem_all = [l for d in corpus for l in d["lemmas"]]
freq = Counter(lem_all); N = len(lem_all)
R["top30"] = [(l, c, round(1000 * c / N, 2)) for l, c in freq.most_common(30)]
R["N_lemmas"] = N

# 5. Латиница в тексте
lat_tok = lambda t: re.findall(r"[A-Za-z][A-Za-z'’\-]*", t)
tot_lat = 0; tot_tok = 0; latc = Counter()
share_by_doc = {}
for d in corpus:
    lt = lat_tok(d["raw_text"]); tk = d["alpha"]
    tot_lat += len(lt); tot_tok += tk; latc.update(x.lower() for x in lt)
    share_by_doc[d["id"]] = len(lt) / tk if tk else 0
R["latin"] = {"tokens": tot_lat, "share_pct": round(100 * tot_lat / tot_tok, 2), "types": len(latc), "top": latc.most_common(30)}
R["latin_by_source"] = {SRC[s]: round(100 * sum(len(lat_tok(d["raw_text"])) for d in v) / sum(d["alpha"] for d in v), 2) for s, v in by_src.items()}
R["latin_by_genre"] = {g: round(100 * sum(len(lat_tok(d["raw_text"])) for d in v) / sum(d["alpha"] for d in v), 2) for g, v in gm.items() if len(v) >= 5}

# 6. Лексика цифровой сферы (поиск по корням в исходном тексте, регистр не учитывается)
DIG = {
 "ИИ и нейросети": r"\bИИ\b|ИИ-|нейросет|искусственн\w+ интеллект|chatgpt|\bGPT\b|генеративн",
 "цифровой": r"цифров",
 "диджитал/digital": r"диджитал|дигитал|\bdigital\b",
 "метавселенная": r"метавселенн|metaverse",
 "виртуальный": r"виртуальн",
 "соцсети и платформы": r"соцсет|социальн\w+ сет|instagram|инстаграм|tiktok|тикток|telegram|телеграм|youtube|threads",
 "блогеры и инфлюенсеры": r"блогер|инфлюенсер|\bблог\b|\bблога\b",
 "алгоритм": r"алгоритм",
 "онлайн/интернет": r"онлайн|интернет|маркетплейс",
 "контент и подписчики": r"контент|хештег|подписчик",
}
R["digital"] = {}
per_doc_hits = defaultdict(int)
for k, pat in DIG.items():
    tf = df = 0
    for d in corpus:
        m = len(re.findall(pat, d["raw_text"], flags=re.I)); tf += m; df += m > 0; per_doc_hits[d["id"]] += m
    R["digital"][k] = {"tf": tf, "df": df}
R["digital_total"] = sum(per_doc_hits.values())
def per10k(docs): return round(10000 * sum(per_doc_hits[d["id"]] for d in docs) / sum(d["alpha"] for d in docs), 1)
R["digital_by_genre"] = {g: per10k(v) for g, v in gm.items() if len(v) >= 5}
R["digital_by_source"] = {SRC[s]: per10k(v) for s, v in by_src.items()}
R["digital_overall_per10k"] = per10k(corpus)
R["docs_with_digital"] = sum(1 for d in corpus if per_doc_hits[d["id"]] > 0)

# 6б. MATTR (скользящий TTR, окно 500) по леммам — устойчив к объёму выборки
def mattr(seq, w=500):
    if len(seq) < w: return None
    win = Counter(seq[:w]); tot = len(win); n = 1
    for i in range(w, len(seq)):
        a = seq[i - w]; win[a] -= 1
        if win[a] == 0: del win[a]
        win[seq[i]] += 1; tot += len(win); n += 1
    return tot / n / w
R["mattr_genre"] = {}
for g, v in gm.items():
    seq = [l for d in sorted(v, key=lambda x: x["id"]) for l in d["lemmas"]]
    R["mattr_genre"][g] = {"lemmas": len(seq), "mattr": round(mattr(seq), 3) if mattr(seq) else None}
R["mattr_source"] = {}
for s, v in by_src.items():
    seq = [l for d in sorted(v, key=lambda x: x["id"]) for l in d["lemmas"]]
    R["mattr_source"][SRC[s]] = {"lemmas": len(seq), "mattr": round(mattr(seq), 3) if mattr(seq) else None}
R["mattr_corpus"] = round(mattr([l for d in sorted(corpus, key=lambda x: x["id"]) for l in d["lemmas"]]), 3)

# 7. Структурные приметы англицизмов (по Гальцевой): -инг, -ция(-сия), -мент, -ер; учёт лемм из словаря корпуса
def is_cyr(l): return re.fullmatch(r"[а-яё\-]+", l) is not None
marks = {"-инг": r".{3,}инг$", "-ция/-сия": r".{3,}[цс]ия$", "-мент": r".{3,}мент$", "-ер/-ор": r".{3,}[ео]р$"}
R["suffix"] = {}
for k, pat in marks.items():
    types = [l for l in freq if is_cyr(l) and re.match(pat, l)]
    tok = sum(freq[l] for l in types)
    top = sorted(types, key=lambda l: -freq[l])[:8]
    R["suffix"][k] = {"types": len(types), "tokens": tok, "share_pct": round(100 * tok / N, 2), "top": [(l, freq[l]) for l in top]}

# 8. Дефисные единицы
hy = [l for l in freq if "-" in l and is_cyr(l) and not l.startswith("-") and not l.endswith("-")]
R["hyphen_artifacts"] = {l: freq[l] for l in freq if l.endswith("-") and is_cyr(l) and freq[l] >= 20}
R["hyphen"] = {"types": len(hy), "tokens": sum(freq[l] for l in hy), "top": [(l, freq[l]) for l in sorted(hy, key=lambda l: -freq[l])[:12]]}

# 9. Ключевые слова: Dunning G2, The Blueprint vs SYMBOL
def keyness(A, B, minf=15, top=15):
    ca = Counter(l for d in A for l in d["lemmas"]); cb = Counter(l for d in B for l in d["lemmas"])
    na, nb = sum(ca.values()), sum(cb.values()); out = []
    for l in set(ca) | set(cb):
        a, b = ca[l], cb[l]
        if a + b < minf or l.endswith("-") or l.startswith("-"): continue
        ea = (a + b) * na / (na + nb); eb = (a + b) * nb / (na + nb)
        g = 2 * ((a * math.log(a / ea) if a else 0) + (b * math.log(b / eb) if b else 0))
        out.append((l, a, b, round(g, 1), "A" if a / na > b / nb else "B"))
    A_ = sorted([o for o in out if o[4] == "A"], key=lambda x: -x[3])[:top]
    B_ = sorted([o for o in out if o[4] == "B"], key=lambda x: -x[3])[:top]
    return {"A": A_, "B": B_, "nA": na, "nB": nb}
R["key_bp_sym"] = keyness(by_src["theblueprint.ru"], by_src["thesymbol.ru"])
trend = [d for d in corpus if d["genre_main"] == "тенденции"]; rest = [d for d in corpus if d["genre_main"] != "тенденции"]
R["key_trend_rest"] = keyness(trend, rest)

# 10. Тема × жанр (укрупнённый), доли
tg = defaultdict(Counter)
for d in corpus: tg[d["genre_main"]][d["topic"]] += 1
R["topic_genre"] = {g: dict(sorted(c.items())) for g, c in tg.items() if sum(c.values()) >= 5}
tsrc = defaultdict(Counter)
for d in corpus: tsrc[SRC[d["source"]]][d["topic"]] += 1
R["topic_source"] = {s: dict(sorted(c.items())) for s, c in tsrc.items()}
R["topic_sizes"] = dict(sorted(Counter(d["topic"] for d in corpus).items()))

# 11. Корреляции длины/TTR (на документ): длина текста vs средняя длина предложения
from scipy.stats import spearmanr
ttr_doc = [len(set(d["lemmas"])) / len(d["lemmas"]) if d["lemmas"] else 0 for d in corpus]
sl_doc = [st.mean(sent_lens(d["raw_text"])) for d in corpus]
r, pv = spearmanr(w, ttr_doc); R["spearman_len_ttr"] = [round(r, 3), pv]
r2, pv2 = spearmanr(w, sl_doc); R["spearman_len_sent"] = [round(r2, 3), pv2]

# 12. Динамика по месяцам публикации
R["months"] = sorted(Counter(d["date"][:7] for d in corpus).items())
mg = defaultdict(list)
for d in corpus: mg[d["date"][:7]].append(d)
R["months_trend_share"] = {m: round(100 * sum(x["genre_main"] == "тенденции" for x in v) / len(v), 1) for m, v in sorted(mg.items())}
R["authors"] = {"editorial": sum(d["author"] == "Редакция сайта" for d in corpus), "named": sum(d["author"] != "Редакция сайта" for d in corpus),
                "unique_named": len({d["author"] for d in corpus if d["author"] != "Редакция сайта"})}

json.dump(R, open("extra_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

# ---------- ФИГУРЫ ----------
# 1 источники
fig, ax = plt.subplots(figsize=(6.4, 3.4))
names = [s["source"] for s in R["sources"]][::-1]; vals = [s["n"] for s in R["sources"]][::-1]
ax.barh(names, vals, color=ACC)
for i, v in enumerate(vals): ax.text(v + 0.8, i, str(v), va="center", fontsize=10)
ax.set_xlabel("Число текстов"); ax.set_xlim(0, max(vals) * 1.12); fig.tight_layout(); fig.savefig("fig1_sources.png", dpi=200); plt.close()
# 2 топ-20 лемм
fig, ax = plt.subplots(figsize=(6.4, 4.6))
t = R["top30"][:20][::-1]
ax.barh([x[0] for x in t], [x[1] for x in t], color=ACC)
for i, x in enumerate(t): ax.text(x[1] + 6, i, str(x[1]), va="center", fontsize=9)
ax.set_xlabel("Абсолютная частота (леммы)"); ax.set_xlim(0, 680); fig.tight_layout(); fig.savefig("fig2_top_lemmas.png", dpi=200); plt.close()
# 3 стилометрия: простой TTR, MATTR и длина предложения
sty = json.load(open("stylometry_results.json", encoding="utf-8"))
gs = sorted(sty, key=lambda g: R["mattr_genre"][g]["mattr"])
fig, axs = plt.subplots(1, 3, figsize=(8.6, 3.9), sharey=True)
axs[0].barh(gs, [sty[g]["ttr"] for g in gs], color=LIGHT); axs[0].set_xlabel("TTR по леммам\n(зависит от объёма)")
axs[1].barh(gs, [R["mattr_genre"][g]["mattr"] for g in gs], color=ACC); axs[1].set_xlabel("MATTR (окно 500)"); axs[1].set_xlim(0.5, 0.85)
axs[2].barh(gs, [sty[g]["avg_sentence_length"] for g in gs], color=GREY); axs[2].set_xlabel("Средняя длина\nпредложения, слов")
for a_ in axs: a_.tick_params(axis="y", labelsize=9)
fig.tight_layout(); fig.savefig("fig3_stylometry.png", dpi=200); plt.close()
# 4 тепловая карта тема × жанр
gl_order = [g for g in R["topic_genre"]]
M = np.array([[R["topic_genre"][g].get(str(t), R["topic_genre"][g].get(t, 0)) for t in range(1, 9)] for g in gl_order], float)
Mp = 100 * M / M.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.6, 4.0))
im = ax.imshow(Mp, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(8)); ax.set_xticklabels([f"Т{t}" for t in range(1, 9)]); ax.set_yticks(range(len(gl_order)))
ax.set_yticklabels([f"{g} (n={int(M[i].sum())})" for i, g in enumerate(gl_order)], fontsize=9)
for i in range(Mp.shape[0]):
    for j in range(Mp.shape[1]):
        if Mp[i, j] >= 1: ax.text(j, i, f"{Mp[i,j]:.0f}", ha="center", va="center", fontsize=8, color="white" if Mp[i, j] > 55 else "black")
cb = fig.colorbar(im, ax=ax); cb.set_label("% текстов жанра")
ax.spines[:].set_visible(False); fig.tight_layout(); fig.savefig("fig4_topic_genre.png", dpi=200); plt.close()
# 5 длины
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.hist([min(x, 3000) for x in w], bins=range(0, 3100, 150), color=ACC, edgecolor="white")
ax.axvline(st.median(w), color=GREY, ls="--", lw=1); ax.text(st.median(w) + 40, ax.get_ylim()[1] * 0.9, f"медиана = {int(st.median(w))}", fontsize=9)
ax.set_xlabel("Длина текста, слов (значения выше 3000 объединены)"); ax.set_ylabel("Число текстов"); fig.tight_layout(); fig.savefig("fig5_lengths.png", dpi=200); plt.close()
# 6 цифровая лексика по источникам
ds = sorted(R["digital_by_source"].items(), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.barh([f"{k} (n={len(by_src[[a for a,b in SRC.items() if b==k][0]])})" for k, v in ds], [v for k, v in ds], color=ACC)
for i, (k, v) in enumerate(ds): ax.text(v + 0.6, i, f"{v:.1f}".replace(".", ","), va="center", fontsize=9)
ax.set_xlabel("Употреблений на 10 000 буквенных токенов"); ax.set_xlim(0, 58); fig.tight_layout(); fig.savefig("fig6_digital_source.png", dpi=200); plt.close()
print("ok")

# ---------- 13. Упоминания модных домов и сезонная привязка (дополнение к §2.2) ----------
BR = {
 "Chanel": r"chanel|шанель", "Dior": r"\bdior\b|диор", "Prada": r"\bprada\b|прада", "Miu Miu": r"miu\s+miu|миу\s+миу", "Gucci": r"gucci|гуччи",
 "Balenciaga": r"balenciaga|баленсиага", "Celine": r"celine|селин\b", "Saint Laurent": r"saint\s+laurent|сен-лоран|сен\s+лоран",
 "Louis Vuitton": r"louis\s+vuitton|луи\s+вюиттон|луи\s+виттон", "Bottega Veneta": r"bottega\s+veneta|боттега\s+венета",
 "Loewe": r"loewe|лоэве", "Valentino": r"valentino|валентино", "Dolce & Gabbana": r"dolce\s*(&|and)?\s*gabbana|дольче\s*(&|и)?\s*габбана",
 "Dries Van Noten": r"van\s+noten|ван\s+нотен", "Hermès": r"herm[eè]s|эрмес", "Jil Sander": r"jil\s+sander|джил\s+сандер",
 "Burberry": r"burberry|бёрберри|барберри", "Versace": r"versace|версаче", "Fendi": r"fendi|фенди", "Givenchy": r"givenchy|живанши",
}
brand = {}
for k, pat in BR.items():
    tf = df = 0
    for d in corpus:
        m = len(re.findall(pat, d["raw_text"], flags=re.I)); tf += m; df += m > 0
    brand[k] = {"tf": tf, "df": df}
R["brands"] = dict(sorted(brand.items(), key=lambda x: -x[1]["tf"]))
seas = r"весна[-–—\s]лето|осень[-–—\s]зима|\bSS\s?'?\d\d|\bFW\s?'?\d\d|\bAW\s?'?\d\d|(?:осенью|осень|зимой|зима|весной|весна|летом|лето)[-–—]\s?20\d\d|(?:осень|зима|весна|лето)\s+20\d\d"
R["season_docs"] = sum(1 for d in corpus if re.search(seas, d["raw_text"], flags=re.I))
R["brand_docs_any"] = sum(1 for d in corpus if any(re.search(p_, d["raw_text"], flags=re.I) for p_ in BR.values()))
json.dump(R, open("extra_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R["brands"], ensure_ascii=False), R["season_docs"], R["brand_docs_any"])
