#!/usr/bin/env python3
"""Ziak et Mikeysem sont-ils le même auteur ?

Mikeysem est le nom qui revient dans la rumeur entourant l'identité de Ziak.
Il ne figurait pas dans LRFAF — faute de page Wikipédia, il était hors du
critère d'inclusion du corpus — et l'étude `02_stylometrie_ziak.ipynb` ne
pouvait donc pas le tester. Ses 7 titres collectés (`mikeysem_lrfaf.csv`)
rendent l'hypothèse testable pour la première fois.

Une difficulté domine : Mikeysem ne pèse que 3 745 mots, contre les 12 000
utilisés comme taille de référence dans l'étude. Tout le protocole est donc
recalibré à cette taille, et sa puissance mesurée avant d'interpréter quoi que
ce soit. Un résultat négatif obtenu avec une méthode aveugle ne vaudrait rien.

Étapes :
1. puissance de la méthode quand les candidats sont réduits à 3 745 mots ;
2. rang de Mikeysem parmi les 493 candidats, pour les quatre combinaisons ;
3. test des imposteurs, étalonné sur des jumeaux authentiques de même taille ;
4. contrôle de redondance (deux titres de Mikeysem sont des versions du même
   morceau).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from stylo_attribution import make_docs, rank_candidates, sample_indices, separation_score
from stylo_features import build_cache, build_count_matrix, char_ngrams, clean_lyrics, tokenize

RESULT_DIR = Path("export")
RESULT_DIR.mkdir(exist_ok=True)

TARGET = "Ziak"
CHALLENGER = "Mikeysem"
SEED = 20260914
N_REPEATS = 30
METRICS = ["cosine_delta", "burrows_delta"]
FEATURES = ["mfw", "char"]


def add_mikeysem(cache: dict) -> tuple[dict, pd.DataFrame]:
    """Ajoute les titres de Mikeysem au cache, avec les vocabulaires du corpus.

    Les vocabulaires restent ceux estimés sur les 32 923 titres du corpus : les
    traits comparés sont donc rigoureusement les mêmes pour tous les artistes.
    """
    m = pd.read_csv("mikeysem_lrfaf.csv")
    m["lyrics_clean"] = m["lyrics"].map(clean_lyrics)
    m["tokens"] = m["lyrics_clean"].map(tokenize)
    m["n_tokens"] = m["tokens"].map(len)

    counts_mfw = build_count_matrix(m["tokens"].tolist(), cache["vocab_mfw"])
    counts_char = build_count_matrix(
        [char_ngrams(t) for t in m["lyrics_clean"]], cache["vocab_char"])

    meta = cache["meta"]
    add = pd.DataFrame({
        "artist": CHALLENGER, "title": m["title"], "year": m["year"].fillna(0).astype(int),
        "n_tokens": m["n_tokens"], "pageviews": m["pageviews"],
        "n_chargrams": np.asarray(counts_char.sum(axis=1)).ravel(),
    })
    new = {
        "meta": pd.concat([meta, add], ignore_index=True),
        "counts_mfw": sparse.vstack([cache["counts_mfw"], counts_mfw]).tocsr(),
        "counts_char": sparse.vstack([cache["counts_char"], counts_char]).tocsr(),
        "vocab_mfw": cache["vocab_mfw"], "vocab_char": cache["vocab_char"],
    }
    return new, m


def build_pool(meta: pd.DataFrame, t_cand: int, exclude: str):
    totals = meta.groupby("artist")["n_tokens"].sum()
    pool = sorted(a for a in totals[totals >= t_cand].index if a != exclude)
    songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    return pool, songs, totals


def validation(cache, t_cand: int, t_query: int, n_rep: int = 6) -> pd.DataFrame:
    """Puissance de la méthode avec des candidats réduits à `t_cand` mots."""
    meta = cache["meta"]
    mats = {"mfw": cache["counts_mfw"], "char": cache["counts_char"]}
    n_tokens = meta["n_tokens"].to_numpy()
    pool, pool_songs, totals = build_pool(meta, t_cand, TARGET)
    controls = [a for a in totals[totals >= t_query + t_cand].index
                if a not in (TARGET, CHALLENGER)]
    print(f"  {len(pool)} candidats, {len(controls)} artistes de contrôle")

    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(n_rep):
        groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], t_cand, rng)
                  for a in pool}
        feats = {}
        for f in FEATURES:
            names, freqs = make_docs(mats[f], groups)
            feats[f] = (names, freqs, {n: i for i, n in enumerate(names)})

        for ctrl in controls:
            songs = pool_songs[ctrl]
            perm = rng.permutation(len(songs))
            cum = np.cumsum(n_tokens[songs[perm]])
            k = int(np.searchsorted(cum, t_query) + 1)
            q_ids, rest = songs[perm[:k]], songs[perm[k:]]
            if rest.size == 0:
                continue
            twin_ids = sample_indices(rest, n_tokens[rest], t_cand, rng)

            for f in FEATURES:
                names, freqs, index = feats[f]
                mat = mats[f]
                q = np.asarray(mat[q_ids].sum(axis=0)).ravel(); q /= max(q.sum(), 1)
                tw = np.asarray(mat[twin_ids].sum(axis=0)).ravel(); tw /= max(tw.sum(), 1)
                cand = freqs.copy(); ci = index[ctrl]; cand[ci] = tw

                for m in METRICS:
                    d = rank_candidates(q, cand, m)
                    order = np.argsort(d)
                    rows.append({"features": f, "metric": m,
                                 "condition": "H1", "rank_twin": int(np.where(order == ci)[0][0]) + 1,
                                 "sep_top1": separation_score(d, int(order[0])),
                                 "top1_is_twin": int(order[0]) == ci})
                    mask = np.ones(len(names), bool); mask[ci] = False
                    d0 = rank_candidates(q, cand[mask], m)
                    rows.append({"features": f, "metric": m, "condition": "H0",
                                 "rank_twin": np.nan,
                                 "sep_top1": separation_score(d0, int(np.argmin(d0))),
                                 "top1_is_twin": False})
    return pd.DataFrame(rows)


def rank_mikeysem(cache, t_cand: int) -> pd.DataFrame:
    """Où se classe Mikeysem parmi tous les candidats, vu depuis Ziak ?"""
    meta = cache["meta"]
    mats = {"mfw": cache["counts_mfw"], "char": cache["counts_char"]}
    n_tokens = meta["n_tokens"].to_numpy()
    pool, pool_songs, _ = build_pool(meta, t_cand, TARGET)
    ziak = meta.index[meta["artist"] == TARGET].to_numpy()
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for rep in range(N_REPEATS):
        groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], t_cand, rng)
                  for a in pool}
        for f in FEATURES:
            mat = mats[f]
            names, freqs = make_docs(mat, groups)
            mi = names.index(CHALLENGER)
            q = np.asarray(mat[ziak].sum(axis=0)).ravel(); q /= max(q.sum(), 1)
            for m in METRICS:
                d = rank_candidates(q, freqs, m)
                order = np.argsort(d)
                rows.append({
                    "rep": rep, "features": f, "metric": m,
                    "rang_mikeysem": int(np.where(order == mi)[0][0]) + 1,
                    "n_candidats": len(names),
                    "sep_mikeysem": separation_score(d, mi),
                    "sep_top1": separation_score(d, int(order[0])),
                    "top1": names[int(order[0])],
                })
    return pd.DataFrame(rows)


def imposteurs(cache, t_cand: int, n_iter: int = 300) -> tuple[float, pd.Series]:
    """Ziak est-il plus proche de Mikeysem que d'imposteurs tirés au hasard ?

    Étalonné sur des jumeaux authentiques de même taille : c'est le seul moyen
    de savoir ce que vaut un score donné.
    """
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    pool, pool_songs, totals = build_pool(meta, t_cand, TARGET)
    ziak = meta.index[meta["artist"] == TARGET].to_numpy()
    rng = np.random.default_rng(SEED + 2)

    groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], t_cand, rng)
              for a in pool}
    names, freqs = make_docs(mat, groups)
    index = {n: i for i, n in enumerate(names)}
    q = np.asarray(mat[ziak].sum(axis=0)).ravel(); q /= max(q.sum(), 1)
    mu, sd = freqs.mean(axis=0), freqs.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    cz, qz = (freqs - mu) / sd, (q - mu) / sd
    n_feat, n_imp = freqs.shape[1], 25

    def score(ci, qvec, cmat):
        wins = 0
        others = [i for i in range(len(names)) if i != ci]
        for _ in range(n_iter):
            cols = rng.choice(n_feat, int(0.4 * n_feat), replace=False)
            imps = rng.choice(others, n_imp, replace=False)
            qs = qvec[cols]; qs = qs / (np.linalg.norm(qs) + 1e-12)
            blk = cmat[np.r_[ci, imps]][:, cols]
            blk = blk / (np.linalg.norm(blk, axis=1, keepdims=True) + 1e-12)
            if int(np.argmin(1.0 - blk @ qs)) == 0:
                wins += 1
        return wins / n_iter

    s_mikeysem = score(index[CHALLENGER], qz, cz)

    # Étalon : vrais jumeaux, même taille de candidat, même taille de requête.
    t_query = int(totals[TARGET])
    controls = [a for a in totals[totals >= t_query + t_cand].index
                if a not in (TARGET, CHALLENGER)]
    controls = list(rng.choice(controls, size=min(40, len(controls)), replace=False))
    ref = {}
    for ctrl in controls:
        songs = pool_songs[ctrl]
        perm = rng.permutation(len(songs))
        cum = np.cumsum(n_tokens[songs[perm]])
        k = int(np.searchsorted(cum, t_query) + 1)
        q_ids, rest = songs[perm[:k]], songs[perm[k:]]
        if rest.size == 0:
            continue
        tw_ids = sample_indices(rest, n_tokens[rest], t_cand, rng)
        qq = np.asarray(mat[q_ids].sum(axis=0)).ravel(); qq /= max(qq.sum(), 1)
        tw = np.asarray(mat[tw_ids].sum(axis=0)).ravel(); tw /= max(tw.sum(), 1)
        cand = freqs.copy(); ci = index[ctrl]; cand[ci] = tw
        m2, s2 = cand.mean(axis=0), cand.std(axis=0, ddof=0)
        s2 = np.where(s2 < 1e-12, 1.0, s2)
        ref[ctrl] = score(ci, (qq - m2) / s2, (cand - m2) / s2)
    return s_mikeysem, pd.Series(ref)


def controles_complementaires(cache, mike: pd.DataFrame, t_cand: int) -> None:
    """Deux objections à écarter avant de conclure.

    a) *Redondance* — « It's Mikeysem » et « . (Period) » sont deux versions du
       même morceau. On refait le classement sans la seconde.
    b) *Auto-cohérence à taille réduite* — on coupe le corpus de Mikeysem en
       deux et on cherche une moitié depuis l'autre. Attention : ce test est
       PLUS DUR que le test principal, où la requête compte 23 886 mots (tout
       Ziak) et non 1 870. Son échec ne condamne donc pas le test principal —
       la puissance pertinente est celle mesurée en section 1, avec une grande
       requête et des candidats de 3 745 mots. Il mesure simplement à quel
       point ce corpus est peu de matière.
    """
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    pool, pool_songs, totals = build_pool(meta, t_cand, TARGET)
    ziak = meta.index[meta["artist"] == TARGET].to_numpy()
    mike_idx = meta.index[meta["artist"] == CHALLENGER].to_numpy()
    rng = np.random.default_rng(SEED + 3)

    # a) sans le doublon
    dup = meta.loc[mike_idx]
    keep = mike_idx[(dup["title"] != ". (Period)").to_numpy()]
    rangs = []
    for _ in range(10):
        groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], t_cand, rng)
                  for a in pool}
        groups[CHALLENGER] = keep
        names, freqs = make_docs(mat, groups)
        q = np.asarray(mat[ziak].sum(axis=0)).ravel(); q /= max(q.sum(), 1)
        d = rank_candidates(q, freqs, "cosine_delta")
        order = np.argsort(d)
        rangs.append(int(np.where(order == names.index(CHALLENGER))[0][0]) + 1)
    print(f"  a) Sans le titre redondant ({int(n_tokens[keep].sum())} mots) : "
          f"rang médian {int(np.median(rangs))} / {len(pool)}")

    # b) Mikeysem retrouve-t-il sa propre moitié ?
    ok = 0
    n_try = 20
    for _ in range(n_try):
        groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], t_cand, rng)
                  for a in pool}
        names, freqs = make_docs(mat, groups)
        perm = rng.permutation(len(mike_idx))
        half = len(perm) // 2
        a_ids, b_ids = mike_idx[perm[:half]], mike_idx[perm[half:]]
        qa = np.asarray(mat[a_ids].sum(axis=0)).ravel(); qa /= max(qa.sum(), 1)
        hb = np.asarray(mat[b_ids].sum(axis=0)).ravel(); hb /= max(hb.sum(), 1)
        cand = np.vstack([freqs, hb])
        d = rank_candidates(qa, cand, "cosine_delta")
        if int(np.argmin(d)) == len(names):
            ok += 1
    print(f"  b) Auto-cohérence (moitié vers moitié, ~1 870 mots de chaque "
          f"côté) : {ok}/{n_try} tirages")
    print("     Test volontairement plus dur que le test principal ; il montre")
    print("     la faiblesse du volume, pas l'invalidité du classement.")

    # c) Mise en perspective : rangs des autres candidats, à la même taille.
    autres = [a for a in ["Kerchak", "Beendo Z", "Zkr", "Rimkus", "ISK", "Werenoi"]
              if a in pool]
    rows = []
    for _ in range(10):
        groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], t_cand, rng)
                  for a in pool}
        names, freqs = make_docs(mat, groups)
        q = np.asarray(mat[ziak].sum(axis=0)).ravel(); q /= max(q.sum(), 1)
        order = np.argsort(rank_candidates(q, freqs, "cosine_delta"))
        pos = {names[j]: i + 1 for i, j in enumerate(order)}
        rows.append({a: pos[a] for a in autres + [CHALLENGER]})
    perspective = pd.DataFrame(rows).median().sort_values()
    perspective.rename("rang_median").to_frame().assign(
        n_candidats=len(pool)).to_csv(RESULT_DIR / "10_5_rangs_compares.csv")
    print("\n  c) Rangs médians vus depuis Ziak, à taille égale (char/cosine) :")
    for a, r in perspective.items():
        mark = "  <-- hypothèse testée" if a == CHALLENGER else ""
        print(f"       {a:14} rang {int(r):3d} / {len(pool)}{mark}")


def main() -> None:
    cache0 = build_cache()
    cache, mike = add_mikeysem(cache0)
    t_cand = int(mike["n_tokens"].sum())
    t_query = int(cache["meta"].query("artist == @TARGET")["n_tokens"].sum())
    print(f"{CHALLENGER} : {len(mike)} titres, {t_cand} mots")
    print(f"{TARGET} : {t_query} mots (requête)\n")

    print("=== 1. PUISSANCE DE LA MÉTHODE À CETTE TAILLE ===")
    val = validation(cache, t_cand, t_query)
    val.to_csv(RESULT_DIR / "10_1_validation_petite_taille.csv", index=False)
    perf = (val[val.condition == "H1"].groupby(["features", "metric"])
            .agg(recall_at_1=("rank_twin", lambda s: (s == 1).mean()),
                 recall_at_5=("rank_twin", lambda s: (s <= 5).mean()),
                 recall_at_20=("rank_twin", lambda s: (s <= 20).mean()),
                 rang_median=("rank_twin", "median"))
            .reset_index().sort_values("recall_at_1", ascending=False))
    print(perf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== 2. RANG DE MIKEYSEM VU DEPUIS ZIAK ===")
    rk = rank_mikeysem(cache, t_cand)
    rk.to_csv(RESULT_DIR / "10_2_rang_mikeysem.csv", index=False)
    res = (rk.groupby(["features", "metric"])
           .agg(rang_median=("rang_mikeysem", "median"),
                rang_min=("rang_mikeysem", "min"),
                rang_max=("rang_mikeysem", "max"),
                sep_mikeysem=("sep_mikeysem", "mean"),
                n_candidats=("n_candidats", "first"),
                top1_modal=("top1", lambda s: s.mode().iloc[0]))
           .reset_index())
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Seuils H1/H0 recalculés à cette taille.
    seuils = []
    for (f, m), g in val.groupby(["features", "metric"]):
        h1 = g[(g.condition == "H1") & g.top1_is_twin]["sep_top1"]
        h0 = g[g.condition == "H0"]["sep_top1"]
        obs = rk[(rk.features == f) & (rk.metric == m)]["sep_mikeysem"].mean()
        seuils.append({"features": f, "metric": m, "sep_mikeysem": obs,
                       "sep_H1_correct": h1.mean(), "sep_H0": h0.mean(),
                       "z_vs_H0": (obs - h0.mean()) / (h0.std(ddof=0) + 1e-12)})
    sdf = pd.DataFrame(seuils)
    sdf.to_csv(RESULT_DIR / "10_3_seuils_petite_taille.csv", index=False)
    print("\n  Séparation de Mikeysem, comparée aux repères de cette taille :")
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== 3. TEST DES IMPOSTEURS ===")
    s_mike, ref = imposteurs(cache, t_cand)
    pd.DataFrame({"artiste": ["Mikeysem"] + list(ref.index),
                  "score": [s_mike] + list(ref.values),
                  "type": ["hypothèse"] + ["jumeau authentique"] * len(ref)}
                 ).to_csv(RESULT_DIR / "10_4_imposteurs.csv", index=False)
    print(f"  Ziak vs Mikeysem            : {s_mike:.3f}")
    print(f"  Jumeaux authentiques (n={len(ref)}) : médiane {ref.median():.3f}, "
          f"1er décile {ref.quantile(0.1):.3f}")
    print(f"  Hasard                      : {1/26:.3f}")
    print(f"  Part des vrais jumeaux sous le score de Mikeysem : "
          f"{100 * (ref <= s_mike).mean():.0f} %")

    print("\n=== 4. CONTRÔLES COMPLÉMENTAIRES ===")
    controles_complementaires(cache, mike, t_cand)


if __name__ == "__main__":
    main()
