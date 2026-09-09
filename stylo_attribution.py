#!/usr/bin/env python3
"""Moteur d'attribution d'auteur à taille contrôlée.

Deux principes gouvernent ce module :

1. **Taille contrôlée.** Chaque « document d'artiste » est un échantillon de
   `T` tokens exactement. Sans cette contrainte, la similarité cosinus mesure
   surtout la couverture de vocabulaire : un artiste prolifique paraît proche
   de tout le monde (cf. `02_diagnostic_biais_taille.py`).

2. **Distances calibrées.** La distance brute au meilleur candidat n'est pas
   interprétable seule. On la convertit en score standardisé par rapport à la
   distribution des distances aux autres candidats, ce qui rend comparables
   des requêtes différentes.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

# Traits stylométriques disponibles.
FEATURE_SETS = ("mfw", "char")


def sample_indices(song_ids: np.ndarray, n_tokens: np.ndarray, target: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Tire des chansons sans remise jusqu'à atteindre `target` tokens."""
    order = rng.permutation(len(song_ids))
    cumsum = np.cumsum(n_tokens[order])
    k = int(np.searchsorted(cumsum, target) + 1)
    k = min(k, len(order))
    return song_ids[order[:k]]


def make_docs(counts: sparse.csr_matrix, groups: dict[str, np.ndarray]
              ) -> tuple[list[str], np.ndarray]:
    """Somme les comptes par groupe, puis convertit en fréquences relatives."""
    names = list(groups)
    mat = np.vstack([
        np.asarray(counts[groups[name]].sum(axis=0)).ravel() for name in names
    ])
    totals = mat.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return names, mat / totals


def zscore(freqs: np.ndarray) -> np.ndarray:
    """Standardise chaque trait sur l'ensemble des documents de référence."""
    mu = freqs.mean(axis=0)
    sd = freqs.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (freqs - mu) / sd


def cosine_delta(query_z: np.ndarray, cand_z: np.ndarray) -> np.ndarray:
    """Cosine Delta (Smith & Aldridge 2011) : distance cosinus sur z-scores.

    Evert et al. (2017) montrent que cette variante domine le Delta de Burrows
    classique sur la plupart des corpus, l'angle étant moins sensible que la
    distance de Manhattan aux traits très dispersés.
    """
    q = query_z / (np.linalg.norm(query_z) + 1e-12)
    c = cand_z / (np.linalg.norm(cand_z, axis=1, keepdims=True) + 1e-12)
    return 1.0 - c @ q


def burrows_delta(query_z: np.ndarray, cand_z: np.ndarray) -> np.ndarray:
    """Delta de Burrows (1992) : distance de Manhattan moyenne sur z-scores."""
    return np.abs(cand_z - query_z).mean(axis=1)


DISTANCES = {"cosine_delta": cosine_delta, "burrows_delta": burrows_delta}


def rank_candidates(query_freq: np.ndarray, cand_freqs: np.ndarray,
                    metric: str = "cosine_delta") -> np.ndarray:
    """Distances requête -> candidats, standardisées sur le même corpus.

    Les z-scores sont estimés sur les candidats seuls : la requête est projetée
    dans ce référentiel, comme un texte anonyme confronté à un corpus connu.
    """
    mu = cand_freqs.mean(axis=0)
    sd = cand_freqs.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    cand_z = (cand_freqs - mu) / sd
    query_z = (query_freq - mu) / sd
    return DISTANCES[metric](query_z, cand_z)


def separation_score(dists: np.ndarray, best_idx: int) -> float:
    """À quel point le meilleur candidat se détache-t-il du reste ?

    Renvoie un z-score négatif : -3 signifie que le candidat retenu est à trois
    écarts-types sous la moyenne des distances aux autres candidats. C'est cette
    grandeur — et non la distance brute — qui est comparable d'une requête à
    l'autre, et qui permet de confronter le cas Ziak aux cas de contrôle.
    """
    others = np.delete(dists, best_idx)
    mu, sd = others.mean(), others.std(ddof=0)
    if sd < 1e-12:
        return 0.0
    return float((dists[best_idx] - mu) / sd)
