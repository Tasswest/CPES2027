#!/usr/bin/env python3
"""Produit les lignes Mikeysem au format LRFAF.

Sorties :
- `mikeysem_lrfaf.csv` : colonnes, ordre et conventions du corpus LRFAF. Les
  colonnes non reproductibles y sont vides, comme le sont les valeurs
  manquantes du corpus original.
- `export/08_mikeysem_estimations.csv` : les mêmes titres avec les colonnes
  approximées ou estimées, suffixées `_est`, et un indicateur de statut par
  colonne. Rien d'estimé n'entre dans le fichier principal.
- `export/08_mikeysem_collecte.csv` : trace des données brutes Genius, pour
  distinguer ce qui est récupéré de ce qui est calculé.

Prérequis : `corpus.csv` à la racine (constantes annuelles), les lexiques dans
`.cache_lex/`, et les données brutes collectées par `collecte_genius.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import lrfaf_pipeline as P
from stylo_features import corpus_csv

ARTIST = "Mikeysem"
RAW_JSON = Path(".cache_lex/mikeysem_raw.json")
LEX_DIR = Path(".cache_lex")
RESULT_DIR = Path("export")
RESULT_DIR.mkdir(exist_ok=True)

# Colonnes que le pipeline LRFAF ne permet pas de reproduire (voir RAPPORT).
NOT_REPRODUCIBLE = [
    "topic", "topic_clean", "ranking",      # Bunka + embeddings Solon-large
    "n_onomatopee",                          # lexique non documenté
    "n_negative", "n_positive",              # LIWC-fr, ressource sous licence
    "sentiment2",                            # modèle non documenté
    "birthdate_artist", "age_artist", "born_in_france",  # Wikidata : absent
]


def load_raw() -> list[dict]:
    if not RAW_JSON.exists():
        raise SystemExit(
            f"{RAW_JSON} introuvable — lancer d'abord "
            f"collecte_genius.py 3152412 {RAW_JSON}")
    return json.load(open(RAW_JSON, encoding="utf-8"))


def apply_lrfaf_filters(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """Applique les deux critères d'inclusion du corpus LRFAF.

    1. Artiste principal : LRFAF n'a retenu que les titres dont l'artiste est le
       `primary_artist` Genius (99,1 % des URL du corpus commencent par le slug
       de l'artiste), ce qui correspond au défaut `include_features=False` de
       lyricsgenius.
    2. Langue : l'article indique avoir « filtré les textes en langue
       étrangère » à partir du champ renvoyé par Genius.
    """
    kept, dropped = [], []
    for s in raw:
        if s["primary_artist"].lower() != ARTIST.lower():
            dropped.append({**s, "motif_exclusion":
                            f"artiste principal = {s['primary_artist']}"})
        elif s.get("language") != "fr":
            dropped.append({**s, "motif_exclusion":
                            f"langue Genius = {s.get('language')}"})
        elif not s.get("lyrics_raw"):
            dropped.append({**s, "motif_exclusion": "paroles introuvables"})
        else:
            kept.append(s)
    return kept, dropped


def build_rows(songs: list[dict], consts: pd.DataFrame,
               lex: dict[str, set[str]], scores: dict[str, dict[str, float]],
               fallback_pvm: float) -> pd.DataFrame:
    rows = []
    for s in songs:
        withhdr = P.lg_clean(s["lyrics_raw"])     # texte des compteurs
        lyrics = P.strip_genius_header(withhdr)   # texte publié
        year = s.get("year")
        pv = s.get("pageviews") or 0              # LRFAF code 0 l'absence de vues
        nw = P.n_words(withhdr)
        nfr = (P.count_lexicon(withhdr, lex["french"]) if "french" in lex
               else np.nan)

        row = {
            "artist": ARTIST,
            "title": s["title"],
            "year": float(year) if year else np.nan,
            "lyrics": lyrics,
            "pageviews": int(pv),
            "contributors": int(s.get("contributors") or 0),
            "url": s["url"],
            "n_je": P.n_je(withhdr),
            "n_profanity": (P.count_lexicon(withhdr, lex["profanity"])
                            if "profanity" in lex else np.nan),
            "n_verlan": (P.count_lexicon(withhdr, lex["verlan"])
                         if "verlan" in lex else np.nan),
            "n_french_words": nfr,
            "means_word_length": P.means_word_length(withhdr),
            "n_unique_words": P.n_unique_words(withhdr),
            "n_words": nw,
            "pageviews_corrected": P.pageviews_corrected(pv, year, consts),
            "n_non_french_words": (nw - nfr) if pd.notna(nfr) else np.nan,
            "n_argot": (P.count_lexicon(withhdr, lex["argot"])
                        if "argot" in lex else np.nan),
            "pageview_mean": (float(consts.loc[year, "pageview_mean"])
                              if year and year in consts.index else fallback_pvm),
            "pageviews_2": P.pageviews_2(pv, year, consts, fallback_pvm),
            "n_sexe": (P.count_lexicon(withhdr, lex["sexe"])
                       if "sexe" in lex else np.nan),
            "hate": scores.get("hate", {}).get(s["url"], np.nan),
            "sexism": scores.get("sexism", {}).get(s["url"], np.nan),
            "n_lines": P.n_lines(withhdr),
        }
        for c in NOT_REPRODUCIBLE:
            row[c] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[P.LRFAF_COLUMNS]


def match_lrfaf_dtypes(df: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    """Aligne les types sur ceux du corpus original."""
    for c in df.columns:
        target = corpus[c].dtype
        if target == np.int64 and df[c].notna().all():
            df[c] = df[c].astype(np.int64)
        elif target == np.float64:
            df[c] = df[c].astype(np.float64)
    return df


def main() -> None:
    corpus = pd.read_csv(corpus_csv())
    consts = P.year_constants(corpus)
    lex = P.load_lexicons(LEX_DIR)
    print(f"Lexiques chargés : "
          + ", ".join(f"{k} ({len(v)})" for k, v in lex.items()))

    raw = load_raw()
    kept, dropped = apply_lrfaf_filters(raw)
    print(f"\n{len(raw)} titres Genius pour {ARTIST}")
    print(f"  retenus par les critères LRFAF : {len(kept)}")
    for d in dropped:
        print(f"  exclu : {d['title']:26} — {d['motif_exclusion']}")

    scores_path = RESULT_DIR / "08_mikeysem_scores_modeles.json"
    scores = json.load(open(scores_path)) if scores_path.exists() else {}
    if not scores:
        print("\n  (scores hate/sexism absents — lancer scores_modeles.py)")

    fallback_pvm = P.fallback_pageview_mean(corpus)
    df = build_rows(kept, consts, lex, scores, fallback_pvm)
    df = match_lrfaf_dtypes(df, corpus)
    df.to_csv("mikeysem_lrfaf.csv", index=False)
    print(f"\nmikeysem_lrfaf.csv écrit : {df.shape[0]} lignes, "
          f"{df.shape[1]} colonnes")

    # Trace de la collecte brute : ce qui vient de Genius, sans transformation.
    pd.DataFrame([{k: v for k, v in s.items() if k != "lyrics_raw"}
                  for s in raw]).to_csv(
        RESULT_DIR / "08_mikeysem_collecte.csv", index=False)

    # Colonnes approximées, tenues à l'écart du fichier principal.
    est = df[["title", "url"]].copy()
    est["n_lines_est"] = df["n_lines"]
    for c in ["n_french_words", "n_non_french_words", "n_profanity",
              "n_verlan", "n_argot", "n_sexe"]:
        est[f"{c}_est"] = df[c]
    est.to_csv(RESULT_DIR / "08_mikeysem_estimations.csv", index=False)

    manquantes = [c for c in P.LRFAF_COLUMNS if df[c].isna().all()]
    print(f"Colonnes entièrement vides ({len(manquantes)}) : "
          + ", ".join(manquantes))
    print("\nAperçu :")
    print(df[["title", "year", "n_words", "n_unique_words", "means_word_length",
              "n_je", "n_lines", "pageviews"]].to_string(index=False))


if __name__ == "__main__":
    main()
