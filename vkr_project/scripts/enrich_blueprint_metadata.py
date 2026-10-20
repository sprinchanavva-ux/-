# -*- coding: utf-8 -*-
"""Дополнительные метаданные статей The Blueprint: рубрика, ключевые слова, теги, авторы, дата изменения, описание.
Вход: blueprint_2026-09.json (результат collect_blueprint_month.py). Выход: blueprint_2026-09_meta.json
Запуск: python enrich_blueprint_metadata.py blueprint_2026-09.json blueprint_2026-09_meta.json"""
import json, re, sys, time, html
import requests
UA = {"User-Agent": "Mozilla/5.0 (compatible; VKR-research-bot/1.0; academic corpus collection)"}
src, out = sys.argv[1], sys.argv[2]
rows = json.load(open(src, encoding="utf-8"))
res = []
for i, x in enumerate(rows, 1):
    r = requests.get(x["url"], headers=UA, timeout=30)
    r.encoding = "utf-8"
    h = r.text
    art = None
    for m in re.finditer(r'<script type="application/ld\+json">(.*?)</script>', h, flags=re.S):
        try:
            j = json.loads(m.group(1))
        except Exception:
            continue
        for g in (j.get("@graph", []) if isinstance(j, dict) else []):
            if g.get("@type") == "Article":
                art = g
    crumbs = re.findall(r'"name"\s*:\s*"([^"]+)"', (re.search(r'BreadcrumbList.*?\]\s*\}', h, flags=re.S) or [""])[0] if re.search(r'BreadcrumbList.*?\]\s*\}', h, flags=re.S) else "")
    tag_links = sorted(set(re.findall(r'href="/tags/([a-z0-9_\-]+)"', h)))
    d = {"url": x["url"], "title": x["title"], "date": x["date"], "rubric_path": [c.strip("/ ") for c in crumbs],
         "articleSection": (art or {}).get("articleSection"), "keywords": (art or {}).get("keywords"),
         "description": html.unescape((art or {}).get("description", "") or "").strip(),
         "dateModified": (art or {}).get("dateModified"),
         "authors": [{"name": a.get("name", "").replace("  ", " ").strip(), "jobTitle": a.get("jobTitle")} for a in ((art or {}).get("author") or [])],
         "tag_pages": tag_links}
    res.append(d)
    print(i, len(rows), x["url"], d["keywords"], tag_links, flush=True)
    time.sleep(2)
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
