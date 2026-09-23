#!/usr/bin/env python3
"""Les couplets d'invités faussent-ils l'attribution ?

Un couplet d'invité est écrit par l'invité. Le laisser dans le corpus de
l'artiste principal rapproche mécaniquement cet artiste de tous ceux qu'il a
invités — et l'hypothèse testée ici, « Ziak est un autre rappeur », porte
justement sur une proximité. Le soupçon est concret : Ziak pose un couplet chez
Kerchak, premier candidat du classement.

**Ce qui peut être nettoyé, et ce qui ne le peut pas.** LRFAF a supprimé les
balises de section avant publication, et ne conserve pas la liste des invités :
sur 400 textes tirés au hasard, aucun ne porte de balise, et 15 titres sur
37 307 signalent un featuring dans leur titre. Les couplets d'invités sont donc
*irrécupérables* dans le corpus. Ils restent en revanche identifiables dans les
paroles collectées directement sur Genius — celles de Ziak et de Mikeysem.

**D'où ce contrôle, en trois temps.**

1. *La requête.* On reconstruit la requête Ziak depuis Genius, invités retirés,
   et l'on rejoue le classement des 392 candidats. Une troisième variante — même
   source Genius, mais invités conservés — sépare l'effet du nettoyage de celui
   du changement de source.

2. *Les collaborateurs.* Si les featurings gouvernaient le classement, les
   artistes qui interviennent réellement chez Ziak devraient se détacher, puis
   reculer une fois nettoyés. On mesure ce déplacement.

3. *Le côté candidat.* Ziak est lui-même invité chez d'autres : ses couplets
   logés dans leur corpus les rapprocheraient de lui à tort. On retire des
   candidats les titres où il intervient, et l'on rejoue le classement.

4. *Le reste des featurings.* Aucun candidat ne peut être nettoyé de ses propres
   invités. On mesure donc ce que cette contamination coûte, en injectant chez
   tous les candidats une part croissante de texte étranger et en regardant à
   partir de quand la méthode cesse de retrouver un auteur connu.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import lrfaf_pipeline as P
from genius_sections import (LIGNE_BALISE, _norme, est_balise_de_section,
                             interpretes, retire_featurings)
from stylo_attribution import (make_docs, rank_candidates, sample_indices,
                               separation_score)
from stylo_features import (build_cache, build_count_matrix, char_ngrams,
                            clean_lyrics, export_dir, tokenize)

RESULT_DIR = export_dir()
ZIAK_RAW = Path(".cache_lex/ziak_raw.json")
MIKE_RAW = Path(".cache_lex/mikeysem_raw.json")
ZIAK_ID, MIKE_ID = 2113831, 3152412

T_CAND = 12_000
N_REPEATS = 30
SEED = 20260923
METRICS = ["cosine_delta", "burrows_delta"]
FEATURES = ["mfw", "char"]
# Années couvertes par LRFAF pour Ziak : la requête nettoyée doit interroger
# les mêmes titres que la requête d'origine, sans quoi on comparerait deux
# périodes plutôt que deux nettoyages.
ANNEES_LRFAF = (2020, 2024)


def cle_titre(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def textes_genius(path: Path, artiste_id: int, alias: set[str]) -> pd.DataFrame:
    """Un titre par ligne, en version complète et en version sans invités."""
    rows = []
    for s in json.load(open(path, encoding="utf-8")):
        if s["primary_artist_id"] != artiste_id or not s.get("lyrics_raw"):
            continue
        if s.get("language") not in (None, "fr"):
            continue
        brut = P.strip_genius_header(s["lyrics_raw"])
        sans, stats = retire_featurings(brut, alias)
        rows.append({
            "title": s["title"], "year": s.get("year"),
            "texte_complet": clean_lyrics(P.lg_clean(brut)),
            "texte_sans_invites": clean_lyrics(P.lg_clean(sans)),
            **stats,
        })
    df = pd.DataFrame(rows)
    return df[df.texte_sans_invites.str.len() > 0].reset_index(drop=True)


def invites_par_titre(path: Path, artiste_id: int, alias: set[str]) -> dict:
    """Noms normalisés des intervenants, titre par titre."""
    out = {}
    for s in json.load(open(path, encoding="utf-8")):
        if s["primary_artist_id"] != artiste_id or not s.get("lyrics_raw"):
            continue
        noms = set()
        for ligne in P.strip_genius_header(s["lyrics_raw"]).split("\n"):
            if est_balise_de_section(ligne):
                n = interpretes(LIGNE_BALISE.match(ligne).group(1))
                if n:
                    noms |= n
        out[cle_titre(s["title"])] = noms - {_norme(a) for a in alias}
    return out


def vecteur(textes: list[str], vocab_mfw: list[str], vocab_char: list[str]) -> dict:
    """Fréquences relatives d'un ensemble de textes, dans les vocabulaires du cache."""
    toks = [tokenize(t) for t in textes]
    grams = [char_ngrams(t) for t in textes]
    out = {}
    for nom, docs, vocab in [("mfw", toks, vocab_mfw), ("char", grams, vocab_char)]:
        v = np.asarray(build_count_matrix(docs, vocab).sum(axis=0)).ravel()
        out[nom] = v / max(v.sum(), 1.0)
    return out


def titres_ou_ziak_invite(path: Path, artiste_id: int, meta: pd.DataFrame) -> list[int]:
    """Indices LRFAF des titres d'autres artistes où Ziak pose un couplet.

    Sa page Genius liste aussi les morceaux où il est invité : l'artiste
    principal y est un autre, et c'est dans le corpus de celui-ci que le
    couplet de Ziak se retrouve une fois les balises supprimées.
    """
    cles = {(s["primary_artist"], cle_titre(s["title"]))
            for s in json.load(open(path, encoding="utf-8"))
            if s["primary_artist_id"] != artiste_id and s.get("lyrics_raw")}
    k = meta["title"].map(cle_titre)
    return [i for i in meta.index if (meta.at[i, "artist"], k[i]) in cles]


def simulation_contamination(meta: pd.DataFrame, mat, n_tokens: np.ndarray,
                             totals: pd.Series, taux=(0.0, 0.05, 0.10, 0.20, 0.35),
                             n_reps: int = 8, n_cibles: int = 40) -> pd.DataFrame:
    """Puissance de la méthode quand chaque candidat contient du texte étranger.

    Les couplets d'invités que LRFAF attribue à l'artiste principal ne peuvent
    pas être retirés. On les simule : chaque document candidat reçoit une part
    `f` de texte emprunté à d'autres artistes tirés au hasard, et l'on regarde
    si le vrai auteur d'une requête reste retrouvable.
    """
    pool = sorted(a for a in totals[totals >= 2 * T_CAND].index if a != "Ziak")
    songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    rng = np.random.default_rng(SEED + 1)
    cibles = list(rng.choice(pool, size=min(n_cibles, len(pool)), replace=False))
    t_query = int(totals["Ziak"])
    lignes = []
    for f in taux:
        rangs = []
        for _ in range(n_reps):
            # Moitiés disjointes pour les cibles : requête d'un côté, jumeau de l'autre.
            moities = {}
            for a in cibles:
                perm = rng.permutation(songs[a])
                cum = np.cumsum(n_tokens[perm])
                k = int(np.searchsorted(cum, t_query) + 1)
                moities[a] = (perm[:k], perm[k:])
            groups = {}
            for a in pool:
                propres = moities[a][1] if a in moities else songs[a]
                if len(propres) == 0:
                    propres = songs[a]
                idx = sample_indices(propres, n_tokens[propres],
                                     int((1 - f) * T_CAND), rng)
                if f > 0:
                    etrangers = np.concatenate(
                        [songs[b] for b in rng.choice(
                            [b for b in pool if b != a], size=3, replace=False)])
                    idx = np.concatenate([idx, sample_indices(
                        etrangers, n_tokens[etrangers], int(f * T_CAND), rng)])
                groups[a] = idx
            names, freqs = make_docs(mat, groups)
            pos = {a: i for i, a in enumerate(names)}
            for a in cibles:
                q = np.asarray(mat[moities[a][0]].sum(axis=0)).ravel()
                d = rank_candidates(q / max(q.sum(), 1.0), freqs, "cosine_delta")
                rangs.append(int((d < d[pos[a]]).sum()) + 1)
        r = np.array(rangs)
        lignes.append({"part_de_texte_etranger": f, "n_essais": len(r),
                       "rang_1": float((r == 1).mean()),
                       "top_20": float((r <= 20).mean()),
                       "rang_median": float(np.median(r))})
    return pd.DataFrame(lignes)


def main() -> None:
    cache = build_cache()
    meta = cache["meta"]
    mats = {"mfw": cache["counts_mfw"], "char": cache["counts_char"]}
    vocabs = {"mfw": cache["vocab_mfw"], "char": cache["vocab_char"]}
    n_tokens = meta["n_tokens"].to_numpy()

    totals = meta.groupby("artist")["n_tokens"].sum()
    pool = sorted(a for a in totals[totals >= T_CAND].index if a != "Ziak")
    pool_songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    ziak_songs = meta.index[meta["artist"] == "Ziak"].to_numpy()
    titres_lrfaf = {cle_titre(t) for t in meta.loc[ziak_songs, "title"]}

    # ------------------------------------------------------------------ #
    # Les trois requêtes                                                  #
    # ------------------------------------------------------------------ #
    gz = textes_genius(ZIAK_RAW, ZIAK_ID, {"Ziak"})
    gz = gz[gz.title.map(cle_titre).isin(titres_lrfaf)].copy()
    a0, a1 = ANNEES_LRFAF
    gz = gz[gz.year.between(a0, a1) | gz.year.isna()]
    part = 100 * gz.mots_retires.sum() / (gz.mots_gardes.sum() + gz.mots_retires.sum())
    print(f"Ziak : {len(gz)} des {len(ziak_songs)} titres LRFAF retrouvés sur Genius ; "
          f"{part:.1f} % des mots appartiennent à des invités\n")

    requetes = {
        "LRFAF (corpus publié)": {
            f: np.asarray(mats[f][ziak_songs].sum(axis=0)).ravel() for f in FEATURES},
        "Genius, invités inclus": vecteur(gz.texte_complet.tolist(), vocabs["mfw"], vocabs["char"]),
        "Genius, invités retirés": vecteur(gz.texte_sans_invites.tolist(), vocabs["mfw"], vocabs["char"]),
    }
    for f in FEATURES:  # la requête LRFAF est en comptes bruts, à normaliser
        v = requetes["LRFAF (corpus publié)"][f]
        requetes["LRFAF (corpus publié)"][f] = v / max(v.sum(), 1.0)

    # ------------------------------------------------------------------ #
    # Classements                                                         #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(SEED)
    lignes = []
    for rep in range(N_REPEATS):
        groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], T_CAND, rng)
                  for a in pool}
        for f in FEATURES:
            names, freqs = make_docs(mats[f], groups)
            noms = np.array(names)
            for variante, q in requetes.items():
                for m in METRICS:
                    d = rank_candidates(q[f], freqs, m)
                    ordre = np.argsort(d)
                    rangs = np.empty(len(d), int)
                    rangs[ordre] = np.arange(1, len(d) + 1)
                    lignes.append({"rep": rep, "variante": variante, "features": f,
                                   "metric": m, "top1": noms[ordre[0]],
                                   "sep_top1": separation_score(d, int(ordre[0])),
                                   **{f"rang::{a}": r for a, r in zip(noms, rangs)}})
    brut = pd.DataFrame(lignes)
    cols_rang = [c for c in brut.columns if c.startswith("rang::")]

    synth = (brut.groupby(["variante", "features", "metric"])
             .agg(sep_top1=("sep_top1", "mean"),
                  top1_modal=("top1", lambda s: s.mode().iat[0]))
             .reset_index())
    print("=== SÉPARATION DU MEILLEUR CANDIDAT, PAR VARIANTE DE REQUÊTE ===")
    print(synth.to_string(index=False), "\n")

    # Rangs des candidats habituels, par variante.
    suivis = ["Kerchak", "Beendo Z", "ISK", "Werenoi", "Zkr", "Rimkus"]
    suivis = [a for a in suivis if f"rang::{a}" in brut.columns]
    ref = brut[(brut.features == "char") & (brut.metric == "cosine_delta")]
    tab = (ref.groupby("variante")[[f"rang::{a}" for a in suivis]].median()
           .rename(columns=lambda c: c.replace("rang::", "")))
    print("=== RANG MÉDIAN DES CANDIDATS HABITUELS (char / cosine delta) ===")
    print(tab.to_string(), "\n")

    # ------------------------------------------------------------------ #
    # Les collaborateurs reculent-ils une fois nettoyés ?                 #
    # ------------------------------------------------------------------ #
    inv = invites_par_titre(ZIAK_RAW, ZIAK_ID, {"Ziak"})
    apparitions = Counter()
    for t in gz.title:
        apparitions.update(inv.get(cle_titre(t), set()))
    norm_pool = {_norme(a): a for a in pool}
    collaborateurs = sorted({norm_pool[n] for n in apparitions if n in norm_pool})
    print(f"=== COLLABORATEURS DE ZIAK PRÉSENTS DANS LE POOL ({len(collaborateurs)}) ===")
    print(", ".join(collaborateurs) or "aucun", "\n")

    colonnes_collab = [f"rang::{a}" for a in collaborateurs]
    autres = [c for c in cols_rang if c not in colonnes_collab]
    collab_rows = []
    for variante in requetes:
        sub = ref[ref.variante == variante]
        collab_rows.append({
            "variante": variante,
            "n_collaborateurs": len(collaborateurs),
            "rang_median_collaborateurs": float(sub[colonnes_collab].median().median())
            if colonnes_collab else np.nan,
            "rang_median_autres": float(sub[autres].median().median()),
            "meilleur_collaborateur": float(sub[colonnes_collab].median().min())
            if colonnes_collab else np.nan,
        })
    collab = pd.DataFrame(collab_rows)
    print("=== RANG DES COLLABORATEURS vs AUTRES (char / cosine delta) ===")
    print(collab.to_string(index=False), "\n")

    # ------------------------------------------------------------------ #
    # Les couplets de Ziak logés chez les candidats                       #
    # ------------------------------------------------------------------ #
    invite_chez = titres_ou_ziak_invite(ZIAK_RAW, ZIAK_ID, meta)
    print(f"=== TITRES DE CANDIDATS CONTENANT UN COUPLET DE ZIAK ({len(invite_chez)}) ===")
    for i in invite_chez:
        print(f"  {meta.at[i, 'artist']} — « {meta.at[i, 'title']} »")
    sans_ziak = {a: np.array([i for i in ids if i not in set(invite_chez)])
                 for a, ids in pool_songs.items()}
    q_prop = requetes["Genius, invités retirés"]
    rng2 = np.random.default_rng(SEED)
    avant, apres = [], []
    for _ in range(N_REPEATS):
        for cible, songs in [(avant, pool_songs), (apres, sans_ziak)]:
            groups = {a: sample_indices(songs[a], n_tokens[songs[a]], T_CAND, rng2)
                      for a in pool}
            names, freqs = make_docs(mats["char"], groups)
            d = rank_candidates(q_prop["char"], freqs, "cosine_delta")
            ordre = np.argsort(d)
            rangs = {names[j]: r for r, j in enumerate(ordre, 1)}
            cible.append({**{a: rangs[a] for a in suivis},
                          "sep_top1": separation_score(d, int(ordre[0]))})
    dz = pd.DataFrame([
        {"corpus des candidats": "tel quel", **pd.DataFrame(avant).median().to_dict()},
        {"corpus des candidats": "titres avec Ziak retirés",
         **pd.DataFrame(apres).median().to_dict()}])
    print("\n=== RANG MÉDIAN, AVANT ET APRÈS RETRAIT DE CES TITRES ===")
    print(dz.to_string(index=False), "\n")
    dz.to_csv(RESULT_DIR / "18_5_couplets_de_ziak.csv", index=False)

    # ------------------------------------------------------------------ #
    # Ce que coûterait une contamination généralisée des candidats        #
    # ------------------------------------------------------------------ #
    simu = simulation_contamination(meta, mats["char"], n_tokens, totals)
    print("=== PUISSANCE QUAND LES CANDIDATS SONT CONTAMINÉS (char / cosine delta) ===")
    print(simu.to_string(index=False), "\n")
    simu.to_csv(RESULT_DIR / "18_6_simulation_contamination.csv", index=False)

    brut.drop(columns=cols_rang).to_csv(RESULT_DIR / "18_1_requetes_bruts.csv", index=False)
    synth.to_csv(RESULT_DIR / "18_2_requetes_synthese.csv", index=False)
    tab.to_csv(RESULT_DIR / "18_3_rangs_par_requete.csv")
    collab.to_csv(RESULT_DIR / "18_4_effet_collaborateurs.csv", index=False)
    print(f"écrit : {RESULT_DIR}/18_1 à 18_6")


if __name__ == "__main__":
    main()
