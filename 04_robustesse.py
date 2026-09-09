#!/usr/bin/env python3
"""Tests de robustesse du verdict sur Ziak.

Quatre objections méritent une réponse chiffrée :

1. *La méthode est-elle aussi puissante sur les artistes récents ?* Ziak
   appartient à une génération (2020-2024) dont le lexique est homogène ;
   si la puissance s'effondrait sur ce sous-groupe, le verdict serait fragile.

2. *Le seuil de 12 000 tokens n'exclut-il pas le bon candidat ?* Analyse de
   sensibilité en faisant varier la taille des documents candidats.

3. *Les candidats récurrents résistent-ils à un test par paire ?* Méthode des
   imposteurs (Koppel & Winter, 2014) : on ne demande plus « qui est le plus
   proche ? » mais « ce candidat précis est-il plus proche que le hasard ? ».

4. *Le verdict tient-il si l'on ne garde que les artistes contemporains ?*
   Restriction du pool aux artistes actifs en même temps que Ziak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stylo_attribution import make_docs, rank_candidates, sample_indices, separation_score
from stylo_features import build_cache

RESULT_DIR = Path("result")
TARGET = "Ziak"
SEED = 20260911


def puissance_par_generation(cache, t_query: int) -> pd.DataFrame:
    """Recall@1 des contrôles, ventilé par période d'activité médiane."""
    val = pd.read_csv(RESULT_DIR / "02_1_validation_brute.csv")
    meta = cache["meta"]
    annee_med = meta.groupby("artist")["year"].median()

    h1 = val[val["condition"] == "H1_jumeau_present"].copy()
    h1["annee_mediane"] = h1["artist"].map(annee_med)
    h1["generation"] = pd.cut(
        h1["annee_mediane"],
        [1989, 2005, 2012, 2017, 2020, 2025],
        labels=["1990-2005", "2006-2012", "2013-2017", "2018-2020", "2021-2024"],
    )
    out = (
        h1[h1["metric"] == "cosine_delta"]
        .groupby(["features", "generation"], observed=True)
        .agg(recall_at_1=("rank_twin", lambda s: (s == 1).mean()),
             recall_at_5=("rank_twin", lambda s: (s <= 5).mean()),
             n_essais=("rank_twin", "size"),
             n_artistes=("artist", "nunique"))
        .reset_index()
    )
    out.to_csv(RESULT_DIR / "04_1_puissance_par_generation.csv", index=False)
    return out


def sensibilite_seuil(cache, t_query: int) -> pd.DataFrame:
    """Le verdict dépend-il de la taille imposée aux documents candidats ?"""
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    totals = meta.groupby("artist")["n_tokens"].sum()
    ziak_songs = meta.index[meta["artist"] == TARGET].to_numpy()

    rows = []
    for t_cand in (5_000, 8_000, 12_000, 18_000, 25_000):
        pool = sorted(a for a in totals[totals >= t_cand].index if a != TARGET)
        pool_songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
        rng = np.random.default_rng(SEED)
        seps, tops = [], []
        for _ in range(10):
            groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], t_cand, rng)
                      for a in pool}
            names, freqs = make_docs(mat, groups)
            q = np.asarray(mat[ziak_songs].sum(axis=0)).ravel()
            q = q / max(q.sum(), 1.0)
            d = rank_candidates(q, freqs, "cosine_delta")
            b = int(np.argmin(d))
            seps.append(separation_score(d, b))
            tops.append(names[b])
        rows.append({
            "t_cand": t_cand, "n_candidats": len(pool),
            "sep_ziak_moy": float(np.mean(seps)),
            "top1_modal": pd.Series(tops).mode().iloc[0],
            "stabilite_top1": float(pd.Series(tops).value_counts().iloc[0] / len(tops)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT_DIR / "04_2_sensibilite_seuil.csv", index=False)
    return out


def test_imposteurs(cache, candidats: list[str], n_iter: int = 200) -> pd.DataFrame:
    """Méthode des imposteurs : vérification par paire contre un fond aléatoire.

    Pour chaque candidat C, on répète : tirer un sous-ensemble aléatoire de
    traits et un lot d'imposteurs, puis vérifier si C est plus proche de Ziak
    que tous les imposteurs. Le score est la proportion de victoires de C.
    Un alias authentique dépasse largement le seuil de 0,5 ; le hasard produit
    des scores proches de 1/(1 + nombre d'imposteurs).
    """
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    totals = meta.groupby("artist")["n_tokens"].sum()
    ziak_songs = meta.index[meta["artist"] == TARGET].to_numpy()

    pool = sorted(a for a in totals[totals >= 12_000].index if a != TARGET)
    pool_songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    rng = np.random.default_rng(SEED)

    groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], 12_000, rng)
              for a in pool}
    names, freqs = make_docs(mat, groups)
    index = {n: i for i, n in enumerate(names)}
    q = np.asarray(mat[ziak_songs].sum(axis=0)).ravel()
    q = q / max(q.sum(), 1.0)

    mu, sd = freqs.mean(axis=0), freqs.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    cand_z, q_z = (freqs - mu) / sd, (q - mu) / sd

    n_feat = freqs.shape[1]
    n_imp = 25
    rows = []
    for cand in candidats:
        ci = index[cand]
        pool_imp = [i for i in range(len(names)) if i != ci]
        wins = 0
        for _ in range(n_iter):
            cols = rng.choice(n_feat, size=int(0.4 * n_feat), replace=False)
            imps = rng.choice(pool_imp, size=n_imp, replace=False)
            qs = q_z[cols]
            qs = qs / (np.linalg.norm(qs) + 1e-12)
            block = cand_z[np.r_[ci, imps]][:, cols]
            block = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-12)
            dd = 1.0 - block @ qs
            if int(np.argmin(dd)) == 0:
                wins += 1
        rows.append({
            "candidat": cand,
            "score_imposteurs": wins / n_iter,
            "seuil_hasard": 1.0 / (n_imp + 1),
        })
    out = pd.DataFrame(rows).sort_values("score_imposteurs", ascending=False)
    out.to_csv(RESULT_DIR / "04_3_test_imposteurs.csv", index=False)
    return out


def imposteurs_reference(cache, n_controles: int = 40, n_iter: int = 200) -> pd.DataFrame:
    """Étalonne le test des imposteurs sur des jumeaux authentiques.

    Sans cet étalon, un score de 0,45 ne veut rien dire : il faut savoir ce que
    la méthode produit quand la bonne réponse est *réellement* présente.
    """
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    totals = meta.groupby("artist")["n_tokens"].sum()
    t_query = int(totals[TARGET])

    pool = sorted(a for a in totals[totals >= 12_000].index if a != TARGET)
    pool_songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    controls = [a for a in totals[totals >= t_query + 12_000].index if a != TARGET]
    rng = np.random.default_rng(SEED + 1)
    controls = list(rng.choice(controls, size=min(n_controles, len(controls)), replace=False))

    groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], 12_000, rng)
              for a in pool}
    names, freqs = make_docs(mat, groups)
    index = {n: i for i, n in enumerate(names)}
    n_feat, n_imp = freqs.shape[1], 25
    rows = []

    for ctrl in controls:
        songs = pool_songs[ctrl]
        perm = rng.permutation(len(songs))
        cum = np.cumsum(n_tokens[songs[perm]])
        k = int(np.searchsorted(cum, t_query) + 1)
        query_ids, rest = songs[perm[:k]], songs[perm[k:]]
        if rest.size == 0:
            continue
        twin_ids = sample_indices(rest, n_tokens[rest], 12_000, rng)

        q = np.asarray(mat[query_ids].sum(axis=0)).ravel()
        q = q / max(q.sum(), 1.0)
        twin = np.asarray(mat[twin_ids].sum(axis=0)).ravel()
        twin = twin / max(twin.sum(), 1.0)
        cand = freqs.copy()
        ci = index[ctrl]
        cand[ci] = twin

        mu, sd = cand.mean(axis=0), cand.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        cz, qz = (cand - mu) / sd, (q - mu) / sd
        pool_imp = [i for i in range(len(names)) if i != ci]
        wins = 0
        for _ in range(n_iter):
            cols = rng.choice(n_feat, size=int(0.4 * n_feat), replace=False)
            imps = rng.choice(pool_imp, size=n_imp, replace=False)
            qs = qz[cols]
            qs = qs / (np.linalg.norm(qs) + 1e-12)
            block = cz[np.r_[ci, imps]][:, cols]
            block = block / (np.linalg.norm(block, axis=1, keepdims=True) + 1e-12)
            if int(np.argmin(1.0 - block @ qs)) == 0:
                wins += 1
        rows.append({"artiste": ctrl, "score_imposteurs_jumeau": wins / n_iter})

    out = pd.DataFrame(rows).sort_values("score_imposteurs_jumeau", ascending=False)
    out.to_csv(RESULT_DIR / "04_4_imposteurs_reference.csv", index=False)
    return out


def main() -> None:
    cache = build_cache()
    totals = cache["meta"].groupby("artist")["n_tokens"].sum()
    t_query = int(totals[TARGET])

    print("=== 1. Puissance de la méthode par génération d'artistes ===")
    print(puissance_par_generation(cache, t_query).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== 2. Sensibilité au seuil de taille des candidats ===")
    print(sensibilite_seuil(cache, t_query).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    # Candidats récurrents relevés dans 03_attribution_ziak.py.
    cands = ["Zkr", "ISK", "Beendo Z", "Rimkus", "L’Animalerie", "Kerchak",
             "Werenoi", "SCH", "Lemon Haze", "UZI (FRA)"]
    dispo = set(cache["meta"]["artist"])
    cands = [c for c in cands if c in dispo]

    print("\n=== 3. Test des imposteurs sur les candidats récurrents ===")
    imp = test_imposteurs(cache, cands)
    print(imp.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== 4. Étalonnage : mêmes tests sur des jumeaux authentiques ===")
    ref = imposteurs_reference(cache)
    print(f"  n = {len(ref)} artistes de contrôle")
    print(f"  score médian d'un vrai jumeau : {ref['score_imposteurs_jumeau'].median():.3f}")
    print(f"  1er décile                    : {ref['score_imposteurs_jumeau'].quantile(0.1):.3f}")
    print(f"  meilleur score obtenu par Ziak: {imp['score_imposteurs'].max():.3f} "
          f"({imp.iloc[0]['candidat']})")
    part = (ref["score_imposteurs_jumeau"] <= imp["score_imposteurs"].max()).mean()
    print(f"  part des vrais jumeaux faisant moins bien que le meilleur candidat de Ziak : {part:.1%}")


if __name__ == "__main__":
    main()
