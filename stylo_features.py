#!/usr/bin/env python3
"""Préparation du corpus et mise en cache des compteurs stylométriques.

Ce module produit, une fois pour toutes, les matrices de comptes utilisées par
les scripts d'analyse (`02_` à `06_`) :

- `counts_mfw`  : comptes des N mots les plus fréquents, par chanson ;
- `counts_char` : comptes des 4-grammes de caractères les plus fréquents ;
- `counts_word_ext` : vocabulaire élargi, pour les seuls marqueurs lexicaux ;
- `n_tokens`    : longueur en tokens de chaque chanson.

Travailler sur des *comptes par chanson* permet ensuite de recomposer
n'importe quel « document d'artiste » de taille contrôlée par simple somme,
ce qui rend possibles les centaines de rééchantillonnages nécessaires à la
validation du protocole.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

CACHE_DIR = Path(".cache_stylo")


def corpus_csv() -> Path:
    """Chemin du corpus LRFAF. Le dépôt amont l'a renommé `corpus.csv` ;
    l'ancien nom reste accepté pour les copies locales déjà téléchargées."""
    for nom in ("corpus.csv", "RapFr.csv"):
        if Path(nom).exists():
            return Path(nom)
    raise FileNotFoundError(
        "Corpus LRFAF introuvable — le télécharger depuis "
        "huggingface.co/datasets/regicid/LRFAF (voir README).")



# Nombre de traits retenus pour chaque famille de descripteurs.
N_MFW = 500          # mots les plus fréquents (Burrows's Delta)
N_CHAR = 3000        # 4-grammes de caractères les plus fréquents
N_WORD_EXT = 6000    # vocabulaire élargi, réservé à l'analyse lexicale

# Balises de section Genius ([Couplet 1], [Refrain], ...). Absentes du corpus
# LRFAF mais on les neutralise par sécurité si une version enrichie est utilisée.
GENIUS_NOISE = re.compile(r"\[[^\]]{0,80}\]")

# On conserve les lettres accentuées (pertinentes en français) et l'apostrophe,
# qui porte l'élision et donc une part du signal grammatical.
NON_LETTERS = re.compile(r"[^a-zàâäéèêëïîôöùûüÿçœæ'\s]+")
APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})


def clean_lyrics(text: str) -> str:
    """Normalise un texte de chanson : minuscules, apostrophes, ponctuation."""
    if not isinstance(text, str):
        return ""
    text = GENIUS_NOISE.sub(" ", text)
    text = text.translate(APOSTROPHES).lower()
    text = text.replace("\n", " ")
    text = NON_LETTERS.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Découpe en mots, l'apostrophe séparant l'élision (j'ai -> j' + ai)."""
    return re.findall(r"[a-zàâäéèêëïîôöùûÿçœæ]+'?", text)


def char_ngrams(text: str, n: int = 4) -> list[str]:
    """4-grammes de caractères, espaces compris (capture les fins de mots)."""
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def load_corpus(min_tokens: int = 100) -> pd.DataFrame:
    """Charge le corpus LRFAF, nettoie les paroles et retire doublons et titres courts.

    Les doublons de paroles entre artistes (0,3 % du corpus) correspondent à des
    featurings ou à des rééditions : les conserver reviendrait à attribuer le
    même texte à deux auteurs, ce qui fausse mécaniquement les distances.
    """
    df = pd.read_csv(corpus_csv())
    df = df.dropna(subset=["artist", "lyrics", "year"]).copy()
    df["lyrics_clean"] = df["lyrics"].map(clean_lyrics)
    df["tokens"] = df["lyrics_clean"].map(tokenize)
    df["n_tokens"] = df["tokens"].map(len)
    df = df[df["n_tokens"] >= min_tokens].copy()

    # Un même texte peut apparaître sous plusieurs artistes (featurings).
    df["dup_key"] = df["lyrics_clean"].str.slice(0, 300)
    df = df.drop_duplicates(subset="dup_key", keep="first")

    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def build_count_matrix(
    docs: list[list[str]], vocab: list[str]
) -> sparse.csr_matrix:
    """Matrice creuse (documents x vocabulaire) de comptes bruts."""
    index = {w: i for i, w in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for r, doc in enumerate(docs):
        counts = Counter(doc)
        for token, c in counts.items():
            j = index.get(token)
            if j is not None:
                rows.append(r)
                cols.append(j)
                vals.append(c)
    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(docs), len(vocab)), dtype=np.float64
    )


def build_cache(force: bool = False) -> dict:
    """Construit (ou recharge) le cache des comptes par chanson."""
    CACHE_DIR.mkdir(exist_ok=True)
    meta_path = CACHE_DIR / "meta.parquet"
    mfw_path = CACHE_DIR / "counts_mfw.npz"
    char_path = CACHE_DIR / "counts_char.npz"
    vocab_path = CACHE_DIR / "vocab.npz"

    ext_path = CACHE_DIR / "counts_word_ext.npz"

    if not force and all(p.exists() for p in (meta_path, mfw_path, char_path,
                                              vocab_path, ext_path)):
        vocabs = np.load(vocab_path, allow_pickle=True)
        return {
            "meta": pd.read_parquet(meta_path),
            "counts_mfw": sparse.load_npz(mfw_path),
            "counts_char": sparse.load_npz(char_path),
            "counts_word_ext": sparse.load_npz(ext_path),
            "vocab_mfw": list(vocabs["mfw"]),
            "vocab_char": list(vocabs["char"]),
            "vocab_word_ext": list(vocabs["word_ext"]),
        }

    print("Construction du cache stylométrique (quelques minutes)...")
    df = load_corpus()
    print(f"  {len(df):,} chansons retenues, {df['artist'].nunique():,} artistes")

    # Vocabulaire des mots les plus fréquents, calculé sur tout le corpus.
    word_freq: Counter = Counter()
    for toks in df["tokens"]:
        word_freq.update(toks)
    vocab_mfw = [w for w, _ in word_freq.most_common(N_MFW)]
    # Vocabulaire élargi : sert aux marqueurs lexicaux, pas à l'attribution.
    vocab_ext = [w for w, _ in word_freq.most_common(N_WORD_EXT)]

    # Vocabulaire des 4-grammes de caractères les plus fréquents.
    char_freq: Counter = Counter()
    for text in df["lyrics_clean"]:
        char_freq.update(char_ngrams(text))
    vocab_char = [g for g, _ in char_freq.most_common(N_CHAR)]

    print("  vectorisation des mots...")
    counts_mfw = build_count_matrix(df["tokens"].tolist(), vocab_mfw)
    print("  vectorisation des 4-grammes...")
    counts_char = build_count_matrix(
        [char_ngrams(t) for t in df["lyrics_clean"]], vocab_char
    )
    print("  vectorisation du vocabulaire élargi...")
    counts_ext = build_count_matrix(df["tokens"].tolist(), vocab_ext)

    meta = df[["artist", "title", "year", "n_tokens", "pageviews"]].copy()
    meta["n_chargrams"] = np.asarray(counts_char.sum(axis=1)).ravel()

    meta.to_parquet(meta_path)
    sparse.save_npz(mfw_path, counts_mfw)
    sparse.save_npz(char_path, counts_char)
    sparse.save_npz(ext_path, counts_ext)
    np.savez(vocab_path, mfw=np.array(vocab_mfw, dtype=object),
             char=np.array(vocab_char, dtype=object),
             word_ext=np.array(vocab_ext, dtype=object))
    print(f"  cache écrit dans {CACHE_DIR}/")

    return {
        "meta": meta,
        "counts_mfw": counts_mfw,
        "counts_char": counts_char,
        "counts_word_ext": counts_ext,
        "vocab_mfw": vocab_mfw,
        "vocab_char": vocab_char,
        "vocab_word_ext": vocab_ext,
    }


if __name__ == "__main__":
    cache = build_cache(force=True)
    meta = cache["meta"]
    print()
    print("Mots les plus fréquents :", ", ".join(cache["vocab_mfw"][:15]))
    print("4-grammes les plus fréquents :", ", ".join(repr(g) for g in cache["vocab_char"][:8]))
    print()
    print("Ziak :", (meta["artist"] == "Ziak").sum(), "titres,",
          int(meta.loc[meta["artist"] == "Ziak", "n_tokens"].sum()), "tokens")
