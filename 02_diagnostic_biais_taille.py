#!/usr/bin/env python3
"""Pourquoi l'approche intuitive désigne le mauvais artiste.

La démarche spontanée consiste à concaténer toutes les chansons de chaque
artiste, à vectoriser, puis à demander qui est le plus proche de Ziak. Elle
produit un classement d'apparence convaincante — et largement illusoire.

Ce script isole la contribution de chaque correction en évaluant trois
variantes avec *le même* protocole de vérité-terrain :

- **A. naïve** : corpus complets, fréquences relatives, distance cosinus.
- **B. taille contrôlée** : documents ramenés à un nombre de tokens identique.
- **C. protocole retenu** : taille contrôlée + standardisation des traits
  (Cosine Delta).

Le critère est le même pour les trois : retrouve-t-on l'auteur d'un texte dont
on connaît la réponse ?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stylo_attribution import make_docs, rank_candidates, sample_indices
from stylo_features import build_cache

RESULT_DIR = Path("export")
RESULT_DIR.mkdir(exist_ok=True)

TARGET = "Ziak"
T_CAND = 12_000
MIN_CAND_TOKENS = 12_000
N_REPEATS = 6
SEED = 20260912


def cosine_brut(query: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Distance cosinus sur fréquences relatives, sans standardisation."""
    q = query / (np.linalg.norm(query) + 1e-12)
    c = cands / (np.linalg.norm(cands, axis=1, keepdims=True) + 1e-12)
    return 1.0 - c @ q


def main() -> None:
    cache = build_cache()
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    totals = meta.groupby("artist")["n_tokens"].sum()

    pool = sorted(a for a in totals[totals >= MIN_CAND_TOKENS].index if a != TARGET)
    pool_songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    ziak_songs = meta.index[meta["artist"] == TARGET].to_numpy()
    t_query = int(totals[TARGET])

    # ------------------------------------------------------------------
    # 1. Le classement naïf de Ziak, et sa corrélation avec la taille
    # ------------------------------------------------------------------
    full_groups = {a: pool_songs[a] for a in pool}
    names, freqs_full = make_docs(mat, full_groups)
    q = np.asarray(mat[ziak_songs].sum(axis=0)).ravel()
    q = q / max(q.sum(), 1.0)

    d_naif = cosine_brut(q, freqs_full)
    classement = pd.DataFrame({
        "artiste": names,
        "distance_naive": d_naif,
        "n_tokens_corpus": [int(totals[a]) for a in names],
    }).sort_values("distance_naive").reset_index(drop=True)
    classement["rang_naif"] = np.arange(1, len(classement) + 1)

    rho, pval = stats.spearmanr(classement["distance_naive"],
                                classement["n_tokens_corpus"])

    # Le même classement, cette fois à taille contrôlée.
    rng = np.random.default_rng(SEED)
    groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], T_CAND, rng)
              for a in pool}
    names_c, freqs_c = make_docs(mat, groups)
    d_ctrl = rank_candidates(q, freqs_c, "cosine_delta")
    ctrl = pd.DataFrame({"artiste": names_c, "distance_controlee": d_ctrl})
    ctrl["rang_controle"] = ctrl["distance_controlee"].rank().astype(int)

    comp = classement.merge(ctrl, on="artiste")
    comp["ecart_de_rang"] = comp["rang_controle"] - comp["rang_naif"]
    comp.sort_values("rang_naif").to_csv(
        RESULT_DIR / "02_1_classement_naif_vs_controle.csv", index=False)

    rho_rangs, _ = stats.spearmanr(comp["rang_naif"], comp["rang_controle"])

    # ------------------------------------------------------------------
    # 2. Puissance comparée des trois variantes sur vérité-terrain
    # ------------------------------------------------------------------
    controls = [a for a in totals[totals >= t_query + T_CAND].index if a != TARGET]
    print(f"Évaluation sur {len(controls)} artistes de contrôle, "
          f"{N_REPEATS} répétitions\n")

    index_full = {n: i for i, n in enumerate(names)}
    rows = []
    for rep in range(N_REPEATS):
        groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], T_CAND, rng)
                  for a in pool}
        names_r, freqs_r = make_docs(mat, groups)
        index_r = {n: i for i, n in enumerate(names_r)}

        for ctrl_artist in controls:
            songs = pool_songs[ctrl_artist]
            perm = rng.permutation(len(songs))
            cum = np.cumsum(n_tokens[songs[perm]])
            k = int(np.searchsorted(cum, t_query) + 1)
            query_ids, rest = songs[perm[:k]], songs[perm[k:]]
            if rest.size == 0:
                continue
            twin_ids = sample_indices(rest, n_tokens[rest], T_CAND, rng)

            qq = np.asarray(mat[query_ids].sum(axis=0)).ravel()
            qq = qq / max(qq.sum(), 1.0)
            twin = np.asarray(mat[twin_ids].sum(axis=0)).ravel()
            twin = twin / max(twin.sum(), 1.0)

            # A. naïve : les autres candidats gardent leur corpus intégral,
            #    seul le jumeau est de taille réaliste.
            cand_a = freqs_full.copy()
            ia = index_full[ctrl_artist]
            cand_a[ia] = twin
            ra = int(np.argsort(cosine_brut(qq, cand_a)).tolist().index(ia)) + 1

            # B. taille contrôlée, sans standardisation.
            cand_b = freqs_r.copy()
            ib = index_r[ctrl_artist]
            cand_b[ib] = twin
            rb = int(np.argsort(cosine_brut(qq, cand_b)).tolist().index(ib)) + 1

            # C. taille contrôlée + Cosine Delta.
            rc = int(np.argsort(rank_candidates(qq, cand_b, "cosine_delta"))
                     .tolist().index(ib)) + 1

            rows.append({"rep": rep, "artiste": ctrl_artist,
                         "A_naive": ra, "B_taille_controlee": rb, "C_protocole": rc})

        print(f"  répétition {rep + 1}/{N_REPEATS}")

    ranks = pd.DataFrame(rows)
    ranks.to_csv(RESULT_DIR / "02_2_rangs_par_variante.csv", index=False)

    perf = []
    for col, lib in [("A_naive", "A. naïve (corpus complets)"),
                     ("B_taille_controlee", "B. taille contrôlée"),
                     ("C_protocole", "C. taille contrôlée + Cosine Delta")]:
        perf.append({
            "variante": lib,
            "recall_at_1": (ranks[col] == 1).mean(),
            "recall_at_5": (ranks[col] <= 5).mean(),
            "recall_at_20": (ranks[col] <= 20).mean(),
            "rang_median": ranks[col].median(),
            "rang_moyen": ranks[col].mean(),
        })
    perf_df = pd.DataFrame(perf)
    perf_df.to_csv(RESULT_DIR / "02_3_puissance_par_variante.csv", index=False)

    synth = pd.Series({
        "spearman_distance_vs_taille": rho,
        "p_value": pval,
        "part_variance_expliquee_par_taille": rho**2,
        "spearman_rang_naif_vs_controle": rho_rangs,
        "top1_naif": classement.iloc[0]["artiste"],
        "top1_controle": ctrl.sort_values("distance_controlee").iloc[0]["artiste"],
        "recall1_naif": perf_df.iloc[0]["recall_at_1"],
        "recall1_protocole": perf_df.iloc[2]["recall_at_1"],
    })
    synth.to_csv(RESULT_DIR / "02_4_synthese_diagnostic.csv")

    print("\n=== LE BIAIS DE TAILLE DANS L'APPROCHE NAÏVE ===")
    print(f"  Corrélation de Spearman entre la distance à Ziak et la taille")
    print(f"  du corpus du candidat : rho = {rho:.3f} (p = {pval:.1e})")
    print(f"  -> {rho**2:.0%} de la variance du classement s'explique par la")
    print(f"     seule quantité de texte disponible sur chaque artiste.")
    print(f"\n  Taille médiane du corpus des 20 « plus proches » : "
          f"{classement.head(20)['n_tokens_corpus'].median():,.0f} tokens")
    print(f"  Taille médiane sur l'ensemble du pool            : "
          f"{classement['n_tokens_corpus'].median():,.0f} tokens")

    print("\n=== LES DEUX CLASSEMENTS N'ONT RIEN EN COMMUN ===")
    print(f"  Corrélation des rangs naïf / contrôlé : rho = {rho_rangs:.3f}")
    print(f"  Premier candidat, approche naïve   : {synth['top1_naif']}")
    print(f"  Premier candidat, taille contrôlée : {synth['top1_controle']}")
    print("\n  Top 10 naïf (avec la taille du corpus) :")
    for _, r in comp.sort_values("rang_naif").head(10).iterrows():
        print(f"    {r['rang_naif']:>2}. {r['artiste']:<26} "
              f"{r['n_tokens_corpus']:>7,} tokens  "
              f"-> rang {r['rang_controle']:>3} une fois la taille contrôlée")

    print("\n=== PUISSANCE COMPARÉE SUR VÉRITÉ-TERRAIN ===")
    print(perf_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
