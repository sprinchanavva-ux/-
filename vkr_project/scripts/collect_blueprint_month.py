# -*- coding: utf-8 -*-
"""
Сбор текстов раздела «Мода» The Blueprint за заданный месяц (по умолчанию — сентябрь 2026).

Принцип отбора (протокол):
  1. Проверяется robots.txt; если раздел запрещён для автоматического сбора, скрипт останавливается.
  2. Адреса берутся из sitemap (из robots.txt или из --sitemap) и/или из файла --urls (список адресов вручную).
  3. Остаются адреса раздела /fashion/ (параметр --section).
  4. Дата берётся со страницы статьи (JSON-LD datePublished, затем meta article:published_time, затем trafilatura).
     Остаются статьи, опубликованные в выбранном месяце.
  5. Текст извлекается trafilatura; тексты короче --min-words слов отбрасываются.
  6. Дубли (одинаковый адрес или заголовок) отбрасываются.
  7. Каждое решение записывается в журнал (CSV) с причиной.

Результат: blueprint_<YYYY-MM>.json (тексты и метаданные), blueprint_<YYYY-MM>_log.csv (журнал отбора).

Запуск (на компьютере с доступом к сайту):
    pip install trafilatura requests
    python collect_blueprint_month.py --month 2026-09
    python collect_blueprint_month.py --month 2026-09 --urls my_urls.txt     # дополнительно свой список адресов

Параметры вежливости: пауза между запросами --delay (секунды) и идентифицирующий User-Agent.
"""

import argparse
import csv
import gzip
import json
import re
import sys
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

import requests
import trafilatura

UA = "Mozilla/5.0 (compatible; VKR-research-bot/1.0; academic corpus collection)"
HEADERS = {"User-Agent": UA}


def get(url, timeout=30):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


def get_html(url):
    r = get(url)
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = "utf-8"
    return r.text


def check_robots(base, section):
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urljoin(base, "/robots.txt")
    txt = get(robots_url).text
    rp.parse(txt.splitlines())
    allowed = rp.can_fetch(UA, urljoin(base, section))
    sitemaps = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", txt)
    return allowed, sitemaps, rp


def read_sitemap(url, depth=0):
    """Рекурсивно читает sitemap и sitemap-индексы; возвращает список (loc, lastmod)."""
    out = []
    try:
        r = get(url)
    except Exception as e:
        print(f"  sitemap недоступен: {url} ({e})", file=sys.stderr)
        return out
    content = r.content
    if url.endswith(".gz"):
        content = gzip.decompress(content)
    root = ET.fromstring(content)
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    if root.tag.endswith("sitemapindex") and depth < 3:
        for sm in root.findall("s:sitemap", ns):
            loc = sm.findtext("s:loc", default="", namespaces=ns).strip()
            if loc:
                out += read_sitemap(loc, depth + 1)
    else:
        for u in root.findall("s:url", ns):
            loc = u.findtext("s:loc", default="", namespaces=ns).strip()
            lm = u.findtext("s:lastmod", default="", namespaces=ns).strip()
            if loc:
                out.append((loc, lm))
    return out


def date_from_html(html):
    """Дата публикации: JSON-LD datePublished -> meta published_time -> None."""
    m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', html)
    if m:
        return m.group(1)[:10], "datePublished"
    m = re.search(r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|og:published_time|pubdate)["\'][^>]+content=["\']([^"\']+)', html, re.I)
    if m:
        return m.group(1)[:10], "meta:published_time"
    m = re.search(r'<time[^>]+datetime=["\']([^"\']+)', html, re.I)
    if m:
        return m.group(1)[:10], "time[datetime]"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://theblueprint.ru")
    ap.add_argument("--section", default="/fashion/")
    ap.add_argument("--month", default="2026-09", help="ГГГГ-ММ")
    ap.add_argument("--sitemap", action="append", default=[], help="адрес sitemap (можно несколько раз)")
    ap.add_argument("--urls", help="файл со списком адресов (по одному в строке)")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--skip-robots-check", action="store_true", help="ТОЛЬКО для локального теста")
    a = ap.parse_args()

    log = []

    def note(url, decision, reason, **extra):
        row = {"url": url, "decision": decision, "reason": reason}
        row.update(extra)
        log.append(row)

    # 1. robots.txt
    sitemaps = []
    if not a.skip_robots_check:
        allowed, sitemaps, _ = check_robots(a.base, a.section)
        print(f"robots.txt: раздел {a.section} {'разрешён' if allowed else 'ЗАПРЕЩЁН'} для автоматического сбора")
        if not allowed:
            print("Остановка: сбор в запрещённом разделе не выполняется.")
            return 2
    sitemaps = a.sitemap or sitemaps

    # 2. адреса
    cand = {}
    for sm in sitemaps:
        for loc, lm in read_sitemap(sm):
            cand[loc] = lm
    if a.urls:
        for line in open(a.urls, encoding="utf-8"):
            u = line.strip()
            if u and not u.startswith("#"):
                cand.setdefault(u, "")
    print(f"Кандидатов в списках: {len(cand)}")

    # 3. раздел
    sec = [u for u in cand if urlparse(u).path.startswith(a.section)]
    for u in cand:
        if u not in sec:
            note(u, "исключён", "вне раздела " + a.section)
    # статья — адрес глубже раздела (не сама рубрика)
    arts = [u for u in sec if len([p for p in urlparse(u).path.split("/") if p]) >= 3]
    for u in sec:
        if u not in arts:
            note(u, "исключён", "страница рубрики, не статья")
    print(f"В разделе {a.section}: {len(arts)} адресов статей")

    # 4-6. страницы
    seen_titles, out = set(), []
    for i, u in enumerate(sorted(arts), 1):
        try:
            html = get_html(u)
        except Exception as e:
            note(u, "исключён", f"не удалось скачать: {e}")
            time.sleep(a.delay)
            continue
        date, src = date_from_html(html)
        if not date:
            res = trafilatura.extract(html, output_format="json", with_metadata=True)
            if res:
                date, src = (json.loads(res).get("date") or "")[:10] or None, "trafilatura"
        if not date:
            note(u, "на ручную проверку", "дата не найдена")
            time.sleep(a.delay)
            continue
        if not date.startswith(a.month):
            note(u, "исключён", f"дата {date} вне {a.month}", date=date, date_source=src)
            time.sleep(a.delay)
            continue
        res = trafilatura.extract(html, include_comments=False, include_tables=False, output_format="json", with_metadata=True)
        if not res:
            note(u, "на ручную проверку", "текст не извлечён", date=date, date_source=src)
            time.sleep(a.delay)
            continue
        p = json.loads(res)
        text, title = p.get("text", ""), (p.get("title") or "").strip()
        words = len(text.split())
        if words < a.min_words:
            note(u, "исключён", f"короче {a.min_words} слов ({words})", date=date, date_source=src, words=words)
        elif title and title in seen_titles:
            note(u, "исключён", "дубль заголовка", date=date, date_source=src, words=words)
        else:
            seen_titles.add(title)
            sub = [x for x in urlparse(u).path.split("/") if x][1:-1]
            out.append({"id": len(out) + 1, "source": urlparse(a.base).netloc, "subsection": "/".join(sub), "genre": "", "author": p.get("author") or "",
                        "date": date, "date_source": src, "title": title, "url": u, "words": words, "text": text})
            note(u, "включён", "", date=date, date_source=src, words=words)
        print(f"[{i}/{len(arts)}] {date or '—'} {u}")
        time.sleep(a.delay)

    out_json = f"{a.out}/blueprint_{a.month}.json"
    out_log = f"{a.out}/blueprint_{a.month}_log.csv"
    json.dump(out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    keys = ["url", "decision", "reason", "date", "date_source", "words"]
    with open(out_log, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(log)
    inc = sum(1 for r in log if r["decision"] == "включён")
    exc = sum(1 for r in log if r["decision"] == "исключён")
    man = sum(1 for r in log if r["decision"] == "на ручную проверку")
    print(f"\nГотово: включено {inc}, исключено {exc}, на ручную проверку {man}. Файлы: {out_json}, {out_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
