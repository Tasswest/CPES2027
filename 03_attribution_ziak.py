#!/usr/bin/env python3
"""Application du protocole validé au cas Ziak.

Trois questions, dans cet ordre :

1. **Le style de Ziak est-il seulement détectable ?** Contrôle positif : on
   coupe son corpus en deux moitiés disjointes et on cherche la seconde depuis
   la première. Si la méthode échoue ici, aucune conclusion n'est possible.

2. **Quels artistes sont les plus proches ?** Classement des candidats, sous
   quatre combinaisons de traits et de distances, avec rééchantillonnage.

3. **Ce meilleur candidat est-il crédible ?** Le score de séparation de Ziak
   est confronté aux distributions établies en validation : celle des cas où
   l'auteur est réellement dans le corpus (H1) et celle des cas où il en est
   absent (H0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stylo_attribution import make_docs, rank_candidates, sample_indices, separation_score
from stylo_features import build_cache

RESULT_DIR = Path("result")
RESULT_DIR.mkdir(exist_ok=True)

TARGET = "Ziak"
T_CAND = 12_000
MIN_CAND_TOKENS = 12_000
N_REPEATS = 30
SEED = 20260910

METRICS = ["cosine_delta", "burrows_delta"]
FEATURES = ["mfw", "char"]


def main() -> None:
    cache = build_cache()
    meta = cache["meta"]
    mats = {"mfw": cache["counts_mfw"], "char": cache["counts_char"]}
    n_tokens = meta["n_tokens"].to_numpy()

    totals = meta.groupby("artist")["n_tokens"].sum()
    t_query = int(totals[TARGET])
    ziak_songs = meta.index[meta["artist"] == TARGET].to_numpy()

    pool = sorted(a for a in totals[totals >= MIN_CAND_TOKENS].index if a != TARGET)
    pool_songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    print(f"{TARGET} : {len(ziak_songs)} titres, {t_query:,} tokens")
    print(f"Pool de candidats : {len(pool)} artistes ({T_CAND:,} tokens chacun)\n")

    rng = np.random.default_rng(SEED)
    rank_rows, sep_rows, selfcheck_rows = [], [], []

    for rep in range(N_REPEATS):
        groups = {
            a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], T_CAND, rng)
            for a in pool
        }

        # --- Contrôle positif : Ziak retrouve-t-il sa propre seconde moitié ? ---
        perm = rng.permutation(len(ziak_songs))
        cum = np.cumsum(n_tokens[ziak_songs[perm]])
        k = int(np.searchsorted(cum, T_CAND) + 1)
        half_a, half_b = ziak_songs[perm[:k]], ziak_songs[perm[k:]]

        for fname in FEATURES:
            mat = mats[fname]
            names, freqs = make_docs(mat, groups)

            # Requête = corpus complet de Ziak.
            q = np.asarray(mat[ziak_songs].sum(axis=0)).ravel()
            q = q / max(q.sum(), 1.0)

            # Requête du contrôle positif = moitié A ; le pool reçoit la moitié B.
            qa = np.asarray(mat[half_a].sum(axis=0)).ravel()
            qa = qa / max(qa.sum(), 1.0)
            hb = np.asarray(mat[half_b].sum(axis=0)).ravel()
            hb = hb / max(hb.sum(), 1.0)
            names_self = names + [f"{TARGET}__moitie_B"]
            cand_self = np.vstack([freqs, hb])

            for metric in METRICS:
                d = rank_candidates(q, freqs, metric)
                order = np.argsort(d)
                best = int(order[0])
                sep = separation_score(d, best)
                sep_rows.append({
                    "rep": rep, "features": fname, "metric": metric,
                    "top1": names[best], "distance_top1": float(d[best]),
                    "sep_top1": sep,
                    "marge_top1_top2": float(d[order[1]] - d[order[0]]),
                })
                for r, idx in enumerate(order[:25], start=1):
                    rank_rows.append({
                        "rep": rep, "features": fname, "metric": metric,
                        "rang": r, "artiste": names[idx], "distance": float(d[idx]),
                    })

                ds = rank_candidates(qa, cand_self, metric)
                os_ = np.argsort(ds)
                twin_pos = int(np.where(os_ == len(names_self) - 1)[0][0]) + 1
                selfcheck_rows.append({
                    "rep": rep, "features": fname, "metric": metric,
                    "rang_moitie_B": twin_pos,
                    "top1": names_self[int(os_[0])],
                    "sep_moitie_B": separation_score(ds, len(names_self) - 1),
                })

        if (rep + 1) % 10 == 0:
            print(f"  répétition {rep + 1}/{N_REPEATS}")

    ranks = pd.DataFrame(rank_rows)
    seps = pd.DataFrame(sep_rows)
    selfc = pd.DataFrame(selfcheck_rows)
    ranks.to_csv(RESULT_DIR / "03_1_classements_bruts_ziak.csv", index=False)
    seps.to_csv(RESULT_DIR / "03_2_separation_ziak.csv", index=False)
    selfc.to_csv(RESULT_DIR / "03_3_controle_positif_ziak.csv", index=False)

    # --- Stabilité du classement : fréquence d'apparition dans le top-5 ---
    top5 = (
        ranks[ranks["rang"] <= 5]
        .groupby(["features", "metric", "artiste"]).size()
        .rename("n_top5").reset_index()
    )
    top5["freq_top5"] = top5["n_top5"] / N_REPEATS
    top5 = top5.sort_values(["features", "metric", "freq_top5"], ascending=[True, True, False])
    top5.to_csv(RESULT_DIR / "03_4_stabilite_top5.csv", index=False)

    # --- Consensus toutes méthodes confondues : rang moyen ---
    consensus = (
        ranks.groupby("artiste")
        .agg(rang_moyen=("rang", "mean"), n_apparitions=("rang", "size"),
             meilleur_rang=("rang", "min"))
        .sort_values("rang_moyen").reset_index()
    )
    consensus["n_methodes"] = ranks.groupby("artiste")["metric"].nunique().reindex(
        consensus["artiste"]).to_numpy()
    consensus.to_csv(RESULT_DIR / "03_5_consensus_candidats.csv", index=False)

    # --- Confrontation aux seuils de décision de la validation ---
    seuils = pd.read_csv(RESULT_DIR / "02_3_seuils_decision.csv")
    verdict = []
    for (f, m), g in seps.groupby(["features", "metric"]):
        s = seuils[(seuils["features"] == f) & (seuils["metric"] == m)].iloc[0]
        obs = g["sep_top1"].mean()
        # Vraisemblance relative sous chaque hypothèse (densités normales).
        z_h1 = (obs - s["sep_H1_correct_moy"]) / s["sep_H1_sd"]
        z_h0 = (obs - s["sep_H0_moy"]) / s["sep_H0_sd"]
        ll_h1 = -0.5 * z_h1**2 - np.log(s["sep_H1_sd"])
        ll_h0 = -0.5 * z_h0**2 - np.log(s["sep_H0_sd"])
        verdict.append({
            "features": f, "metric": m,
            "sep_ziak": obs,
            "sep_H1_correct_moy": s["sep_H1_correct_moy"],
            "sep_H0_moy": s["sep_H0_moy"],
            "z_vs_H1": z_h1, "z_vs_H0": z_h0,
            "log_rapport_vraisemblance_H0_H1": ll_h0 - ll_h1,
            "hypothese_favorisee": "H0 (auteur absent du corpus)" if ll_h0 > ll_h1
                                   else "H1 (auteur présent)",
            "top1_modal": g["top1"].mode().iloc[0],
        })
    verdict_df = pd.DataFrame(verdict)
    verdict_df.to_csv(RESULT_DIR / "03_6_verdict_hypotheses.csv", index=False)

    print("\n=== CONTRÔLE POSITIF : Ziak retrouve-t-il sa propre moitié ? ===")
    ctrl = selfc.groupby(["features", "metric"]).agg(
        rang_median_moitie_B=("rang_moitie_B", "median"),
        taux_rang1=("rang_moitie_B", lambda s: (s == 1).mean()),
        sep_moyenne=("sep_moitie_B", "mean"),
    ).reset_index()
    print(ctrl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== CANDIDATS LES PLUS PROCHES DE ZIAK (top 8 par méthode) ===")
    for (f, m), g in top5.groupby(["features", "metric"]):
        head = g.head(8)
        print(f"\n[{f} / {m}]")
        for _, r in head.iterrows():
            print(f"   {r['artiste']:<28} présent dans le top-5 : {r['freq_top5']:.0%}")

    print("\n=== VERDICT ===")
    print(verdict_df[["features", "metric", "sep_ziak", "sep_H1_correct_moy",
                      "sep_H0_moy", "hypothese_favorisee", "top1_modal"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
