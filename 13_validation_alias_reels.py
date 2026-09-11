#!/usr/bin/env python3
"""Validation sur des recouvrements d'auteur réels, et non simulés.

La validation des sections 3 et 5 repose sur des « jumeaux » fabriqués : on coupe
l'œuvre d'un artiste en deux et l'on cherche une moitié depuis l'autre. C'est une
tâche *facile* — les deux moitiés partagent la même époque, les mêmes thèmes, le
même producteur. La puissance qu'on y mesure est donc une borne optimiste, et
c'est la principale objection qu'on peut opposer au verdict sur Ziak.

Ce script y répond avec des cas où la vérité est connue *hors* du corpus. Le rap
français change souvent de nom, mais LRFAF fusionne ces changements : il ne
contient qu'Ateyaba (et pas Joke), qu'un seul Gims. Aucun alias solo-solo n'y est
donc testable.

Restent les paires **solo / groupe**, où l'artiste solo a réellement écrit une
partie des textes du groupe. Le test est même plus sévère qu'un alias : dans un
trio, l'auteur ne signe qu'un tiers du texte, et le reste est du bruit écrit par
d'autres. Si la méthode retrouve le groupe malgré cette dilution, elle a de la
marge.

Protocole, identique à celui appliqué à Ziak : la requête est le corpus solo, les
candidats sont ramenés à 12 000 mots, et l'on relève le rang du groupe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stylo_attribution import make_docs, rank_candidates, sample_indices, separation_score
from stylo_features import build_cache

RESULT_DIR = Path("result")
RESULT_DIR.mkdir(exist_ok=True)

T_CAND = 12_000
N_REPEATS = 10
SEED = 20260915

# (solo, groupe, nombre de rappeurs du groupe) — le dernier champ sert à lire
# les résultats en fonction de la dilution attendue.
PAIRES = [
    ("Booba", "Lunatic", 2),
    ("Kool Shen", "Suprême NTM", 2),
    ("Gringe", "Casseurs Flowters", 2),
    ("Kery James", "Ideal J", 2),
    ("Rockin’ Squat", "Assassin (FRA)", 2),
    ("Akhenaton", "IAM", 3),
    ("Nekfeu", "S-Crew", 3),
    ("Tunisiano", "Sniper", 3),
    ("Jazzy Bazz", "L’Entourage", 5),
    ("Deen Burbigo", "L’Entourage", 5),
    ("Nekfeu", "1995", 6),
    ("Alpha Wann", "1995", 6),
    ("Black M", "Sexion d’Assaut", 8),
    ("Lefa", "Sexion d’Assaut", 8),
]


def main() -> None:
    cache = build_cache()
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    totals = meta.groupby("artist")["n_tokens"].sum()

    T_QUERY = int(totals["Ziak"])   # 23 886 mots : la requête de référence
    pool_all = sorted(totals[totals >= T_CAND].index)
    songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool_all}
    rng = np.random.default_rng(SEED)

    rows = []
    for solo, groupe, n_mc in PAIRES:
        if solo not in totals.index or groupe not in totals.index:
            print(f"  (ignoré : {solo} / {groupe} absent du pool)")
            continue
        # Le solo est retiré du pool : on cherche le groupe depuis ses textes,
        # exactement comme on cherche un alias de Ziak sans Ziak dans le pool.
        pool = [a for a in pool_all if a != solo]
        for rep in range(N_REPEATS):
            # La requête est ramenée à la taille du corpus de Ziak : sans cela,
            # Booba (114 983 mots) et Gringe (18 092) ne seraient pas comparables
            # entre eux, ni au cas qui nous intéresse. Le score de séparation
            # dépend directement de la quantité de texte interrogée.
            q_ids = sample_indices(songs[solo], n_tokens[songs[solo]], T_QUERY, rng)
            groups = {a: sample_indices(songs[a], n_tokens[songs[a]], T_CAND, rng)
                      for a in pool}
            names, freqs = make_docs(mat, groups)
            gi = names.index(groupe)
            q = np.asarray(mat[q_ids].sum(axis=0)).ravel()
            q = q / max(q.sum(), 1.0)
            d = rank_candidates(q, freqs, "cosine_delta")
            order = np.argsort(d)
            rows.append({
                "solo": solo, "groupe": groupe, "n_rappeurs": n_mc, "rep": rep,
                "mots_requete": int(n_tokens[q_ids].sum()),
                "rang_groupe": int(np.where(order == gi)[0][0]) + 1,
                "n_candidats": len(names),
                "sep_groupe": separation_score(d, gi),
                # Séparation du meilleur candidat, quel qu'il soit : c'est la
                # grandeur réellement comparable au cas Ziak, où l'on n'a pas de
                # partenaire désigné mais seulement un premier du classement.
                "sep_top1": separation_score(d, int(order[0])),
                "top1": names[int(order[0])],
                "top1_est_groupe": int(order[0]) == gi,
            })
        r = pd.DataFrame(rows)
        last = r[(r.solo == solo) & (r.groupe == groupe)]
        print(f"  {solo:16} -> {groupe:20} rang médian "
              f"{int(last.rang_groupe.median()):4d} / {int(last.n_candidats.iloc[0])}"
              f"   séparation {last.sep_groupe.mean():+.2f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULT_DIR / "13_1_alias_reels_bruts.csv", index=False)

    synth = (res.groupby(["solo", "groupe", "n_rappeurs"])
             .agg(rang_median=("rang_groupe", "median"),
                  rang_min=("rang_groupe", "min"),
                  rang_max=("rang_groupe", "max"),
                  sep_moy=("sep_groupe", "mean"),
                  sep_top1_moy=("sep_top1", "mean"),
                  taux_rang1=("top1_est_groupe", "mean"),
                  n_candidats=("n_candidats", "first"))
             .reset_index().sort_values("rang_median"))
    synth.to_csv(RESULT_DIR / "13_2_alias_reels_synthese.csv", index=False)

    print("\n=== SYNTHÈSE ===")
    print(synth.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # --- Comparaison avec Ziak et Mikeysem ---
    zsep = pd.read_csv(RESULT_DIR / "04_2_separation_ziak.csv")
    zsep = zsep[(zsep.features == "char") & (zsep.metric == "cosine_delta")]
    print("\n=== MISE EN PERSPECTIVE (4-grammes, Cosine Delta) ===")
    print(f"  Recouvrements réels : rang médian "
          f"{synth.rang_median.median():.0f}, séparation moyenne "
          f"{synth.sep_moy.mean():+.2f}")
    print(f"  dont top 1          : {100 * (synth.rang_median == 1).mean():.0f} % des paires")
    print(f"  dont top 20         : {100 * (synth.rang_median <= 20).mean():.0f} % des paires")
    duos = synth[synth.n_rappeurs == 2]
    print(f"\n  Restreint aux duos (cas le plus proche d'un alias, n={len(duos)}) :")
    print(f"    rang médian {duos.rang_median.median():.0f}, "
          f"top 20 pour {100 * (duos.rang_median <= 20).mean():.0f} % des paires")
    print("\n  COMPARAISON VALIDE — séparation du meilleur candidat du classement :")
    print(f"    cas à recouvrement réel : {res.sep_top1.mean():+.2f} "
          f"(écart-type {res.sep_top1.std():.2f})")
    print(f"    Ziak                    : {zsep.sep_top1.mean():+.2f}")
    print(f"    part des cas réels dont le top-1 sépare moins bien que Ziak : "
          f"{100 * (res.sep_top1 > zsep.sep_top1.mean()).mean():.0f} %")

    # Puissance en fonction de la dilution.
    par_taille = (synth.groupby("n_rappeurs")
                  .agg(rang_median=("rang_median", "median"),
                       sep=("sep_moy", "mean"), n=("solo", "size")).reset_index())
    par_taille.to_csv(RESULT_DIR / "13_3_effet_dilution.csv", index=False)
    print("\n=== EFFET DE LA DILUTION ===")
    print("  (plus le groupe compte de rappeurs, moins le solo y pèse)")
    print(par_taille.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
