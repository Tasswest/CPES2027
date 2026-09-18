#!/usr/bin/env python3
"""Que coûte un changement d'identité artistique à la détection stylométrique ?

Le test d'alias idéal — un artiste, deux noms, deux corpus séparés — est
irréalisable ici : **Genius fusionne lui-même les changements de nom**. Les
titres de la période *Joke* sont classés sous *Ateyaba*, il n'existe aucune page
« Joke », et LRFAF hérite de cette fusion. Vérification faite, les pages
« Peter Punk » et « Malsain » trouvées sur Genius sont des homonymes — un groupe
italien et un groupe de metal — et non Disiz ni Sinik.

Cette fusion rend possible un test différent, et plus instructif : découper ces
artistes **de part et d'autre de leur changement d'identité**. On interroge la
période d'avant et l'on cherche celle d'après, placée dans le pool sous une autre
étiquette. C'est la situation exacte d'un alias : même personne, identité
artistique renouvelée, volonté affichée de rompre — Ateyaba a explicitement
déclaré vouloir « tuer Joke ».

Pour savoir ce que ce changement coûte, les trois cas sont comparés à un groupe
de contrôle d'artistes découpés au même endroit de leur carrière (la médiane de
leur production) mais qui, eux, n'ont jamais changé de nom.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stylo_attribution import make_docs, rank_candidates, sample_indices, separation_score
from stylo_features import build_cache

RESULT_DIR = Path("export")
RESULT_DIR.mkdir(exist_ok=True)

T_CAND = 12_000
N_REPEATS = 10
N_CONTROLES = 45
SEED = 20260916

# Changements d'identité documentés, avec l'année de rupture.
RUPTURES = {
    "Ateyaba": (2016, "Joke → Ateyaba"),
    "Gims": (2019, "Maître Gims → Gims"),
    "Disiz": (2012, "Disiz la Peste → Disiz"),
}


def teste(artiste, annee, meta, mat, n_tokens, songs, pool_all, t_query, rng,
          n_rep=N_REPEATS):
    """Interroge la période antérieure, cherche la période postérieure."""
    d = meta[meta["artist"] == artiste]
    av = d.index[d["year"] < annee].to_numpy()
    ap = d.index[d["year"] >= annee].to_numpy()
    if n_tokens[av].sum() < 8000 or n_tokens[ap].sum() < T_CAND:
        return []
    pool = [a for a in pool_all if a != artiste]
    out = []
    for rep in range(n_rep):
        q_ids = sample_indices(av, n_tokens[av], t_query, rng)
        cible_ids = sample_indices(ap, n_tokens[ap], T_CAND, rng)
        groups = {a: sample_indices(songs[a], n_tokens[songs[a]], T_CAND, rng)
                  for a in pool}
        names, freqs = make_docs(mat, groups)
        q = np.asarray(mat[q_ids].sum(axis=0)).ravel(); q /= max(q.sum(), 1)
        cible = np.asarray(mat[cible_ids].sum(axis=0)).ravel()
        cible /= max(cible.sum(), 1)
        cand = np.vstack([freqs, cible])
        ci = len(names)
        d_ = rank_candidates(q, cand, "cosine_delta")
        order = np.argsort(d_)
        out.append({
            "artiste": artiste, "annee_rupture": annee, "rep": rep,
            "rang_periode_apres": int(np.where(order == ci)[0][0]) + 1,
            "n_candidats": len(names) + 1,
            "sep_cible": separation_score(d_, ci),
            "sep_top1": separation_score(d_, int(order[0])),
            "trouve_rang1": int(order[0]) == ci,
            "mots_avant": int(n_tokens[q_ids].sum()),
            "mots_apres": int(n_tokens[cible_ids].sum()),
        })
    return out


def main() -> None:
    cache = build_cache()
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    totals = meta.groupby("artist")["n_tokens"].sum()
    t_query = int(totals["Ziak"])

    pool_all = sorted(totals[totals >= T_CAND].index)
    songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool_all}
    rng = np.random.default_rng(SEED)

    print("=== CAS DE CHANGEMENT D'IDENTITÉ ===")
    rows = []
    for a, (an, lib) in RUPTURES.items():
        r = teste(a, an, meta, mat, n_tokens, songs, pool_all, t_query, rng)
        for x in r:
            x["groupe"] = "changement d'identité"; x["libelle"] = lib
        rows += r
        if r:
            rr = pd.DataFrame(r)
            print(f"  {lib:26} rang médian {int(rr.rang_periode_apres.median()):3d}"
                  f" / {int(rr.n_candidats.iloc[0])}   séparation {rr.sep_cible.mean():+.2f}"
                  f"   rang 1 : {100 * rr.trouve_rang1.mean():.0f} %")

    # --- Contrôles : même découpage, mais sans changement de nom ---
    elig = []
    for a in pool_all:
        if a in RUPTURES or a == "Ziak":
            continue
        d = meta[meta["artist"] == a]
        if len(d) < 10:
            continue
        med = d["year"].median()
        if (n_tokens[d.index[d["year"] < med].to_numpy()].sum() >= 8000
                and n_tokens[d.index[d["year"] >= med].to_numpy()].sum() >= T_CAND):
            elig.append((a, med))
    sel = [elig[i] for i in rng.choice(len(elig), min(N_CONTROLES, len(elig)),
                                       replace=False)]
    print(f"\n=== CONTRÔLES : {len(sel)} artistes sans changement de nom ===")
    ctrl_rows = []
    for a, med in sel:
        r = teste(a, med, meta, mat, n_tokens, songs, pool_all, t_query, rng, n_rep=4)
        for x in r:
            x["groupe"] = "sans changement"; x["libelle"] = a
        ctrl_rows += r
    rows += ctrl_rows

    res = pd.DataFrame(rows)
    res.to_csv(RESULT_DIR / "14_1_alias_temporel_bruts.csv", index=False)

    synth = (res.groupby(["groupe", "libelle"])
             .agg(rang_median=("rang_periode_apres", "median"),
                  sep_cible=("sep_cible", "mean"),
                  taux_rang1=("trouve_rang1", "mean"))
             .reset_index().sort_values(["groupe", "rang_median"]))
    synth.to_csv(RESULT_DIR / "14_2_alias_temporel_synthese.csv", index=False)

    print("\n=== COMPARAISON ===")
    for g, d in synth.groupby("groupe"):
        print(f"\n  {g} (n = {len(d)}) :")
        print(f"    rang médian de la période postérieure : {d.rang_median.median():.0f}")
        print(f"    retrouvée au rang 1                   : {100 * (d.rang_median == 1).mean():.0f} % des artistes")
        print(f"    retrouvée dans le top 20              : {100 * (d.rang_median <= 20).mean():.0f} %")
        print(f"    séparation moyenne                    : {d.sep_cible.mean():+.2f}")

    chg = synth[synth.groupe == "changement d'identité"]
    ctl = synth[synth.groupe == "sans changement"]
    print("\n=== LECTURE ===")
    print(f"  Un changement d'identité fait passer le rang médian de "
          f"{ctl.rang_median.median():.0f} à {chg.rang_median.median():.0f},")
    print(f"  et la séparation de {ctl.sep_cible.mean():+.2f} à {chg.sep_cible.mean():+.2f}.")
    pire = (ctl.rang_median > chg.rang_median.median()).mean()
    print(f"  {100 * pire:.0f} % des artistes sans changement de nom font moins bien")
    print(f"  que la médiane des artistes qui en ont changé.")

    # Position de Ziak sur la même échelle.
    z = pd.read_csv(RESULT_DIR / "04_2_separation_ziak.csv")
    z = z[(z.features == "char") & (z.metric == "cosine_delta")]
    print(f"\n  Pour mémoire, Ziak : séparation du meilleur candidat "
          f"{z.sep_top1.mean():+.2f} (aucun lien connu)")
    print(f"  Séparation du top-1 dans ces tests : "
          f"{res.sep_top1.mean():+.2f}")


if __name__ == "__main__":
    main()
