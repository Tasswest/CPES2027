#!/usr/bin/env python3
"""Collecte les données brutes Genius d'un artiste (même source que LRFAF).

Usage : python3 collecte_genius.py <artist_id> <sortie.json>

Tous les titres rattachés à la page artiste sont conservés, sans filtre : les
critères d'inclusion de LRFAF (artiste principal, langue) sont appliqués au
moment de l'analyse, pour que les exclusions restent visibles et motivées.
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": UA})


def api(path, **params):
    r = S.get(f"https://genius.com/api{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()["response"]


def scrape_lyrics(url):
    """Bloc de paroles tel que le lit lyricsgenius, en-tête du site compris."""
    html = S.get(url, timeout=30).text
    soup = BeautifulSoup(html.replace("<br/>", "\n").replace("<br>", "\n"), "html.parser")
    divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    if not divs:
        divs = soup.find_all("div", class_=re.compile(r"Lyrics__Container"))
    return "\n".join(d.get_text() for d in divs) if divs else None


def main(artist_id: int, out: Path) -> None:
    songs, page = [], 1
    while True:
        r = api(f"/artists/{artist_id}/songs", per_page=50, page=page, sort="popularity")
        songs += r["songs"]
        if not r.get("next_page"):
            break
        page += 1
        time.sleep(1)
    print(f"{len(songs)} titres listés pour l'artiste {artist_id}", flush=True)

    rows = []
    for i, s in enumerate(songs, 1):
        d = api(f"/songs/{s['id']}")["song"]
        time.sleep(1.2)
        lyrics = scrape_lyrics(d["url"])
        time.sleep(1.2)
        rd = d.get("release_date_components") or {}
        stats = d.get("stats") or {}
        rows.append({
            "song_id": d["id"],
            "title": d["title"],
            "primary_artist": d["primary_artist"]["name"],
            "primary_artist_id": d["primary_artist"]["id"],
            "featured_artists": [a["name"] for a in d.get("featured_artists", [])],
            "url": d["url"],
            "language": d.get("language"),
            "release_date": d.get("release_date"),
            "year": rd.get("year"),
            "pageviews": stats.get("pageviews"),
            "contributors": stats.get("contributors"),
            "album": (d.get("album") or {}).get("name"),
            "lyrics_raw": lyrics,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(songs)}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"écrit : {out} ({sum(1 for r in rows if r['lyrics_raw'])} titres avec paroles)")


if __name__ == "__main__":
    main(int(sys.argv[1]), Path(sys.argv[2]))
