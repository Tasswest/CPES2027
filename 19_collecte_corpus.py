#!/usr/bin/env python3
"""Re-collecte les paroles balisées des artistes du corpus, depuis Genius.

LRFAF a supprimé les balises de section (« [Couplet 1 : X] ») avant publication.
Sans elles, impossible de savoir quelle strophe revient à l'artiste principal et
laquelle appartient à un invité : le couplet de l'invité est attribué au
propriétaire du morceau, ce qui rapproche mécaniquement les artistes qui
collaborent. Comme l'hypothèse testée par ce dépôt — « Ziak est un autre
rappeur » — porte précisément sur une proximité, cette confusion doit être levée.

L'information est récupérable : LRFAF conserve l'URL Genius de chaque titre, et
la page porte encore ses balises. Ce script les re-télécharge.

**Reprise.** La sortie est un JSONL en ajout seul, indexé par URL : relancer le
script reprend là où il s'est arrêté. C'est indispensable ici — la collecte
porte sur plus de trente mille pages et dure une dizaine d'heures.

Usage :
    python3 19_collecte_corpus.py               # tous les artistes éligibles
    python3 19_collecte_corpus.py --limit 500   # par tranches
    python3 19_collecte_corpus.py --artistes Kerchak "Beendo Z"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from stylo_features import build_cache, corpus_csv

SORTIE = Path(".cache_lex/corpus_balises.jsonl")
JOURNAL = Path(".cache_lex/corpus_balises_echecs.jsonl")
MIN_TOKENS_ARTISTE = 12_000
# Quatre fils, chacun marquant une pause d'une seconde : environ trois pages par
# seconde au total. Assez pour boucler en une nuit, assez peu pour ne pas peser
# sur Genius — et le moindre refus déclenche une attente exponentielle.
N_FILS = 4
PAUSE = 1.0          # s entre deux pages, pour chaque fil
PAUSE_ERREUR = 30.0  # s après un refus, avant de réessayer
MAX_ESSAIS = 4

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


_local = threading.local()


def session() -> requests.Session:
    """Une session par fil : `requests.Session` n'est pas conçue pour le partage."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Language": "fr,en;q=0.8"})
        _local.session = s
    return s


def paroles(url: str) -> tuple[str | None, str]:
    """Bloc de paroles balisé d'une page Genius, ou (None, motif)."""
    s = session()
    for essai in range(MAX_ESSAIS):
        try:
            r = s.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(PAUSE_ERREUR * (essai + 1))
            continue
        if r.status_code == 404:
            return None, "404"
        if r.status_code in (429, 403) or r.status_code >= 500:
            # Genius limite le débit : on attend de plus en plus longtemps.
            time.sleep(PAUSE_ERREUR * (2 ** essai) + random.uniform(0, 5))
            continue
        soup = BeautifulSoup(
            r.text.replace("<br/>", "\n").replace("<br>", "\n"), "html.parser")
        # `data-lyrics-container` est l'attribut stable, que Genius maintient
        # pour son propre front-end. Les deux replis visent les balisages plus
        # anciens, dont les noms de classes minifiés changent à chaque refonte
        # (c'est sur eux que repose lyricsgenius, et ce qui le casse).
        divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
        if not divs:
            divs = soup.find_all("div", class_=re.compile(r"Lyrics__Container"))
        if not divs:
            divs = soup.find_all("div", class_=re.compile(r"^Lyrics-\w{2}.\w+.[1]"))
        if not divs:
            return None, "pas de bloc de paroles"
        return "\n".join(d.get_text() for d in divs), "ok"
    return None, "refus répété"


def a_collecter(artistes: list[str] | None) -> pd.DataFrame:
    """Titres à récupérer : ceux des artistes retenus, non encore collectés."""
    meta = build_cache()["meta"]
    if artistes:
        retenus = set(artistes)
    else:
        totaux = meta.groupby("artist")["n_tokens"].sum()
        retenus = set(totaux[totaux >= MIN_TOKENS_ARTISTE].index)
    corpus = pd.read_csv(corpus_csv(), usecols=["artist", "title", "url", "year"])
    corpus = corpus[corpus.artist.isin(retenus) & corpus.url.notna()]
    corpus = corpus.drop_duplicates(subset="url")

    faits = set()
    if SORTIE.exists():
        with open(SORTIE, encoding="utf-8") as f:
            for ligne in f:
                try:
                    faits.add(json.loads(ligne)["url"])
                except (json.JSONDecodeError, KeyError):
                    continue
    if JOURNAL.exists():          # les 404 ne seront pas réessayés
        with open(JOURNAL, encoding="utf-8") as f:
            for ligne in f:
                try:
                    e = json.loads(ligne)
                    if e.get("motif") == "404":
                        faits.add(e["url"])
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"{len(retenus)} artistes, {len(corpus):,} titres ; "
          f"{len(faits):,} déjà traités", flush=True)
    return corpus[~corpus.url.isin(faits)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--artistes", nargs="*", default=None)
    args = ap.parse_args()

    reste = a_collecter(args.artistes)
    if args.limit:
        reste = reste.head(args.limit)
    if reste.empty:
        print("rien à collecter.")
        return
    SORTIE.parent.mkdir(parents=True, exist_ok=True)

    debut = time.time()
    compte = {"i": 0, "ok": 0, "ko": 0}
    verrou = threading.Lock()

    def traite(r) -> None:
        texte, motif = paroles(r.url)
        time.sleep(PAUSE + random.uniform(0, 0.3))
        with verrou:
            compte["i"] += 1
            if texte:
                out.write(json.dumps({"url": r.url, "artist": r.artist,
                                      "title": r.title, "year": r.year,
                                      "lyrics_raw": texte}, ensure_ascii=False) + "\n")
                compte["ok"] += 1
            else:
                log.write(json.dumps({"url": r.url, "artist": r.artist,
                                      "motif": motif}, ensure_ascii=False) + "\n")
                compte["ko"] += 1
            i = compte["i"]
            if i % 100 == 0:
                out.flush(); log.flush()
                vitesse = i / (time.time() - debut)
                print(f"  {i:,}/{len(reste):,}  ok={compte['ok']:,} "
                      f"échecs={compte['ko']:,}  {vitesse:.2f} titres/s  "
                      f"reste ~{(len(reste) - i) / vitesse / 3600:.1f} h", flush=True)

    with open(SORTIE, "a", encoding="utf-8") as out, \
         open(JOURNAL, "a", encoding="utf-8") as log:
        with ThreadPoolExecutor(max_workers=N_FILS) as pool:
            list(pool.map(traite, list(reste.itertuples())))
    print(f"\nterminé : {compte['ok']:,} titres collectés, "
          f"{compte['ko']:,} échecs -> {SORTIE}")


if __name__ == "__main__":
    main()
