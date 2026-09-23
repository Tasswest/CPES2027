#!/usr/bin/env python3
"""Contrôle : le pipeline reconstitué reproduit-il les valeurs de LRFAF ?

Deux contrôles indépendants, parce qu'ils ne mesurent pas la même chose.

**Contrôle A — sur archive.** Le dépôt `regicid/genius_french_rap_corpus`
contient les textes bruts scrapés en février 2024, avant nettoyage. En les
faisant passer dans le pipeline, on compare des valeurs calculées sur *le même
texte source* que celui du corpus. C'est le vrai test du pipeline.

**Contrôle B — cohérence interne.** On applique le pipeline aux paroles
publiées dans `corpus.csv` et on les compare aux compteurs de la même ligne.
Ce contrôle mesure surtout une propriété du corpus lui-même : compteurs et
paroles n'y ont pas été produits au même moment.

Le contrôle A demande l'archive (~18 Mo), téléchargée puis mise en cache.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import lrfaf_pipeline as P
from stylo_features import corpus_brut_csv, export_dir

CACHE = Path(".cache_lex")
ARCHIVE_URL = ("https://github.com/regicid/genius_french_rap_corpus/"
               "raw/master/corpus_rap_francais.tar.gz")
RESULT_DIR = export_dir()


def ensure_archive() -> Path:
    """Télécharge et décompresse l'archive des textes bruts si nécessaire."""
    raw_dir = CACHE / "raw_2024"
    if raw_dir.exists() and any(raw_dir.glob("*.txt")):
        return raw_dir
    CACHE.mkdir(exist_ok=True)
    tgz = CACHE / "corpus_rap_francais.tar.gz"
    if not tgz.exists():
        print(f"téléchargement de l'archive des textes bruts...")
        urllib.request.urlretrieve(ARCHIVE_URL, tgz)
    raw_dir.mkdir(exist_ok=True)
    with tarfile.open(tgz) as t:
        t.extractall(raw_dir)
    return raw_dir


def controle_A(corpus: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Pipeline appliqué aux textes bruts d'archive, comparé au corpus."""
    ref = {r.url: r for r in corpus.itertuples()}
    rows = []
    for f in glob.glob(str(raw_dir / "**/*.txt"), recursive=True):
        slug = os.path.basename(f)[:-4].rsplit("_", 1)[0]
        r = ref.get(f"https://genius.com/{slug}")
        if r is None or not isinstance(r.lyrics, str):
            continue
        raw = open(f, encoding="utf-8", errors="replace").read()
        withhdr = P.lg_clean(raw)
        # Les textes d'archive sont dépourvus de l'en-tête « N Contributors… » :
        # ils servent donc directement de texte de comptage.
        rows.append({
            "url": r.url,
            "lyrics_ok": withhdr == r.lyrics or r.lyrics.startswith(withhdr),
            "lyrics_exact": withhdr == r.lyrics,
            "d_n_words": P.n_words(withhdr) - r.n_words,
            "d_n_unique": P.n_unique_words(withhdr) - r.n_unique_words,
            "d_n_je": P.n_je(withhdr) - r.n_je,
            "d_n_lines": P.n_lines(withhdr) - r.n_lines,
            "d_mwl": P.means_word_length(withhdr) - r.means_word_length,
        })
    return pd.DataFrame(rows)


def controle_B(corpus: pd.DataFrame, lex: dict, n: int = 4000) -> pd.DataFrame:
    """Pipeline appliqué aux paroles publiées, comparé aux compteurs publiés."""
    s = corpus.dropna(subset=["lyrics"]).sample(n, random_state=0)
    rows = []
    for r in s.itertuples():
        t = r.lyrics
        row = {
            "url": r.url,
            "d_n_words": P.n_words(t) - r.n_words,
            "d_n_unique": P.n_unique_words(t) - r.n_unique_words,
            "d_n_je": P.n_je(t) - r.n_je,
            "d_n_lines": P.n_lines(t) - r.n_lines,
            "d_mwl": P.means_word_length(t) - r.means_word_length,
        }
        if "french" in lex:
            row["d_n_french"] = P.count_lexicon(t, lex["french"]) - r.n_french_words
        rows.append(row)
    return pd.DataFrame(rows)


def controle_arithmetique(corpus: pd.DataFrame) -> pd.DataFrame:
    """Relations internes exactes, indépendantes du texte."""
    d = corpus.copy()
    out = []

    e = (d["n_words"] - d["n_french_words"]) - d["n_non_french_words"]
    out.append({"relation": "n_non_french_words = n_words - n_french_words",
                "n": int(e.notna().sum()), "exact_%": 100 * (e == 0).mean()})

    m = d.dropna(subset=["year", "pageviews_corrected"])
    f = m["pageviews_corrected"] - np.log(m["pageviews"] + 10)
    sd = f.groupby(m["year"]).std().max()
    out.append({"relation": "pageviews_corrected = log(pageviews+10) + f(year)",
                "n": len(m), "exact_%": 100.0 if sd < 1e-9 else np.nan})

    p = d[(d["pageviews"] > 0) & d["pageviews_2"].notna()]
    e2 = np.log(p["pageviews"] / p["pageview_mean"]) - p["pageviews_2"]
    out.append({"relation": "pageviews_2 = log(pageviews / pageview_mean)",
                "n": len(p), "exact_%": 100 * (e2.abs() < 2e-3).mean()})

    z = d[d["pageviews"] == 0]
    out.append({"relation": "pageviews_2 = 0 quand pageviews = 0",
                "n": len(z), "exact_%": 100 * (z["pageviews_2"] == 0).mean()})

    a = d.dropna(subset=["age_artist", "birthdate_artist", "year"])
    out.append({"relation": "age_artist = year - birthdate_artist",
                "n": len(a),
                "exact_%": 100 * ((a["year"] - a["birthdate_artist"])
                                  == a["age_artist"]).mean()})
    return pd.DataFrame(out)


def resume(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df:
            continue
        v = df[c].dropna()
        tol = 1e-6 if c == "d_mwl" else 0
        rows.append({
            "colonne": c.replace("d_", ""),
            "n": len(v),
            "exact_%": 100 * (v.abs() <= tol).mean(),
            "ecart_median": v.median(),
            "|ecart|<=2_%": 100 * (v.abs() <= 2).mean(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    corpus = pd.read_csv(corpus_brut_csv())
    lex = P.load_lexicons(CACHE)
    cols = ["d_n_words", "d_n_unique", "d_n_je", "d_n_lines", "d_mwl",
            "d_n_french"]

    print("=== RELATIONS ARITHMÉTIQUES INTERNES ===")
    ar = controle_arithmetique(corpus)
    ar.to_csv(RESULT_DIR / "09_1_controle_arithmetique.csv", index=False)
    print(ar.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\n=== CONTRÔLE A : pipeline sur les textes bruts d'archive ===")
    raw_dir = ensure_archive()
    A = controle_A(corpus, raw_dir)
    A.to_csv(RESULT_DIR / "09_2_controle_archive.csv", index=False)
    print(f"  {len(A)} titres appariés avec l'archive de février 2024")
    print(f"  paroles reproduites (exact ou préfixe) : {100 * A.lyrics_ok.mean():.1f} %")
    print(f"  dont strictement identiques            : {100 * A.lyrics_exact.mean():.1f} %")
    # Sur les titres dont le texte est strictement identique, les compteurs
    # doivent l'être aussi : c'est là que le pipeline est réellement testable.
    sub = A[A.lyrics_exact]
    print(f"\n  Sur les {len(sub)} titres au texte strictement identique :")
    print(resume(sub, cols).to_string(index=False,
                                      float_format=lambda x: f"{x:.1f}"))

    print("\n=== CONTRÔLE B : cohérence interne du corpus publié ===")
    B = controle_B(corpus, lex)
    B.to_csv(RESULT_DIR / "09_3_controle_interne.csv", index=False)
    print(resume(B, cols).to_string(index=False,
                                    float_format=lambda x: f"{x:.1f}"))
    print("\n  Lecture : ces écarts ne mesurent pas une erreur du pipeline mais")
    print("  un décalage propre au corpus — ses compteurs n'ont pas été calculés")
    print("  sur les paroles qu'il publie (l'en-tête Genius y pèse 3 à 4 mots).")


if __name__ == "__main__":
    main()
