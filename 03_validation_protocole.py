#!/usr/bin/env python3
"""Validation du protocole d'attribution sur vérité-terrain.

Avant de demander « à qui ressemble Ziak ? », il faut savoir si la méthode est
capable de reconnaître un auteur *dont on connaît la réponse*, dans des
conditions strictement identiques à celles du cas Ziak.

Protocole
---------
Pour chaque artiste de contrôle disposant d'assez de texte :

- on prélève `T_QUERY` tokens (la taille exacte du corpus de Ziak) qui jouent
  le rôle du texte anonyme ;
- on prélève `T_CAND` tokens *disjoints* qui forment son « jumeau », placé dans
  le pool sous une autre étiquette ;
- tous les autres artistes sont représentés par `T_CAND` tokens, de sorte
  qu'aucun candidat n'est avantagé par la taille ;
- on classe les candidats et on relève le rang du jumeau.

Deux conditions sont évaluées :

- **H1 (jumeau présent)** : la bonne réponse est dans le pool. Mesure la
  puissance de la méthode.
- **H0 (jumeau retiré)** : la bonne réponse est absente. C'est exactement
  l'hypothèse à laquelle Ziak doit être confronté s'il n'est aucun des
  artistes du corpus.

La comparaison des scores de séparation sous H1 et sous H0 fournit le seuil de
décision utilisé ensuite pour Ziak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from stylo_attribution import make_docs, rank_candidates, sample_indices, separation_score
from stylo_features import build_cache, export_dir

RESULT_DIR = export_dir()

TARGET = "Ziak"
T_CAND = 12_000        # tokens par artiste candidat
MIN_CAND_TOKENS = 12_000
N_REPEATS = 10
SEED = 20260909

METRICS = ["cosine_delta", "burrows_delta"]
FEATURES = ["mfw", "char"]


def build_pool(meta: pd.DataFrame) -> tuple[list[str], dict[str, np.ndarray]]:
    """Artistes disposant d'assez de texte pour un document de `T_CAND` tokens."""
    totals = meta.groupby("artist")["n_tokens"].sum()
    eligible = sorted(totals[totals >= MIN_CAND_TOKENS].index)
    songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in eligible}
    return eligible, songs


def main() -> None:
    cache = build_cache()
    meta = cache["meta"]
    mats = {"mfw": cache["counts_mfw"], "char": cache["counts_char"]}

    totals = meta.groupby("artist")["n_tokens"].sum()
    t_query = int(totals[TARGET])
    print(f"Requête calibrée sur la taille de {TARGET} : {t_query:,} tokens")
    print(f"Candidats : {T_CAND:,} tokens chacun\n")

    pool, pool_songs = build_pool(meta)
    print(f"Pool de candidats : {len(pool)} artistes")

    # Un artiste de contrôle doit fournir requête ET jumeau, sur des titres disjoints.
    need = t_query + T_CAND
    controls = sorted(totals[totals >= need].index)
    controls = [c for c in controls if c != TARGET]
    print(f"Artistes de contrôle (>= {need:,} tokens) : {len(controls)}\n")

    n_tokens = meta["n_tokens"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []

    for rep in range(N_REPEATS):
        # Un échantillon de T_CAND tokens pour chaque artiste du pool.
        base_groups = {
            a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], T_CAND, rng)
            for a in pool
        }

        feats = {}
        for fname in FEATURES:
            names, freqs = make_docs(mats[fname], base_groups)
            feats[fname] = (names, freqs, {n: i for i, n in enumerate(names)})

        for ctrl in controls:
            songs = pool_songs[ctrl]
            perm = rng.permutation(len(songs))
            cum = np.cumsum(n_tokens[songs[perm]])
            k_q = int(np.searchsorted(cum, t_query) + 1)
            query_ids = songs[perm[:k_q]]
            rest = songs[perm[k_q:]]
            if rest.size == 0:
                continue
            twin_ids = sample_indices(rest, n_tokens[rest], T_CAND, rng)

            for fname in FEATURES:
                names, freqs, index = feats[fname]
                mat = mats[fname]

                q = np.asarray(mat[query_ids].sum(axis=0)).ravel()
                q = q / max(q.sum(), 1.0)
                twin = np.asarray(mat[twin_ids].sum(axis=0)).ravel()
                twin = twin / max(twin.sum(), 1.0)

                # Le jumeau remplace l'artiste de contrôle dans le pool :
                # la requête ne doit jamais pouvoir se comparer à elle-même.
                cand = freqs.copy()
                ci = index[ctrl]
                cand[ci] = twin

                for metric in METRICS:
                    d = rank_candidates(q, cand, metric)

                    # --- H1 : le jumeau est présent dans le pool ---
                    order = np.argsort(d)
                    rank_twin = int(np.where(order == ci)[0][0]) + 1
                    best = int(order[0])
                    rows.append({
                        "rep": rep, "artist": ctrl, "features": fname,
                        "metric": metric, "condition": "H1_jumeau_present",
                        "rank_twin": rank_twin,
                        "top1": names[best],
                        "top1_is_twin": best == ci,
                        "sep_top1": separation_score(d, best),
                        "sep_twin": separation_score(d, ci),
                        "n_candidates": len(names),
                    })

                    # --- H0 : le jumeau est retiré (auteur absent du corpus) ---
                    mask = np.ones(len(names), dtype=bool)
                    mask[ci] = False
                    d0 = rank_candidates(q, cand[mask], metric)
                    b0 = int(np.argmin(d0))
                    rows.append({
                        "rep": rep, "artist": ctrl, "features": fname,
                        "metric": metric, "condition": "H0_jumeau_absent",
                        "rank_twin": np.nan,
                        "top1": [n for n, m in zip(names, mask) if m][b0],
                        "top1_is_twin": False,
                        "sep_top1": separation_score(d0, b0),
                        "sep_twin": np.nan,
                        "n_candidates": int(mask.sum()),
                    })

        print(f"  répétition {rep + 1}/{N_REPEATS} terminée")

    res = pd.DataFrame(rows)
    res.to_csv(RESULT_DIR / "03_1_validation_brute.csv", index=False)

    # --- Puissance de la méthode sous H1 ---
    h1 = res[res["condition"] == "H1_jumeau_present"]
    perf = (
        h1.groupby(["features", "metric"])
        .agg(
            recall_at_1=("rank_twin", lambda s: (s == 1).mean()),
            recall_at_5=("rank_twin", lambda s: (s <= 5).mean()),
            recall_at_10=("rank_twin", lambda s: (s <= 10).mean()),
            recall_at_20=("rank_twin", lambda s: (s <= 20).mean()),
            rang_median=("rank_twin", "median"),
            sep_top1_moy=("sep_top1", "mean"),
            n_essais=("rank_twin", "size"),
        )
        .reset_index()
        .sort_values("recall_at_1", ascending=False)
    )
    perf.to_csv(RESULT_DIR / "03_2_puissance_methode.csv", index=False)

    # --- Seuils de décision : séparation sous H1 vs H0 ---
    seuils = []
    for (f, m), g in res.groupby(["features", "metric"]):
        a = g[g["condition"] == "H1_jumeau_present"]["sep_top1"]
        b = g[g["condition"] == "H0_jumeau_absent"]["sep_top1"]
        a_hit = g[(g["condition"] == "H1_jumeau_present") & g["top1_is_twin"]]["sep_top1"]
        seuils.append({
            "features": f, "metric": m,
            "sep_H1_moy": a.mean(), "sep_H1_sd": a.std(ddof=0),
            "sep_H1_correct_moy": a_hit.mean() if len(a_hit) else np.nan,
            "sep_H0_moy": b.mean(), "sep_H0_sd": b.std(ddof=0),
            "sep_H0_p05": b.quantile(0.05),
            "ecart_H1_H0": b.mean() - a.mean(),
        })
    pd.DataFrame(seuils).to_csv(RESULT_DIR / "03_3_seuils_decision.csv", index=False)

    print("\n=== PUISSANCE DE LA MÉTHODE (le jumeau est dans le pool) ===")
    print(perf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== SÉPARATION DU MEILLEUR CANDIDAT ===")
    print(pd.DataFrame(seuils).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
