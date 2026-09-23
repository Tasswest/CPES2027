#!/usr/bin/env python3
"""web7 (ex-7 Jaws) écrit-il les textes de Ziak ?

La rumeur, telle que la rapporte la presse musicale, répartit Ziak entre trois
personnes : web7, anciennement 7 Jaws, écrirait les textes ; Mikeysem porterait le
masque et les interpréterait ; le producteur Seezy serait « Hellboy ». Aucune
preuve n'est citée, et web7 a refusé d'en parler sans démentir.

C'est une hypothèse de *ghostwriting*, et elle ne se teste pas tout à fait comme
un alias :

- **Le nègre littéraire écrit pour la voix d'un autre.** Il adapte son registre au
  personnage, ce qui dilue sa signature — la section 6 de l'article a montré que
  la détection se dégrade quand l'auteur ne signe qu'une partie de ce qu'on lit.
- **L'interprète laisse sa marque dans les transcriptions.** Les ad-libs (« uh »,
  « huh », « gang ») sont les marqueurs les plus forts du corpus de Ziak ; ils
  appartiennent à la performance, pas à l'écriture. Genius les note surtout entre
  parenthèses (87 % des « uh », 95 % des « huh ») : une variante retire ces
  segments, pour *tous* les artistes afin que la comparaison reste équitable.
- **Les époques diffèrent.** web7 publie depuis 2016, Ziak depuis 2020 : une
  variante restreint web7 à sa production contemporaine de Ziak.

Ni web7 ni Mikeysem ne figurent dans LRFAF : faute de page Wikipédia, ils
échappaient au test principal de l'article.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import lrfaf_pipeline as P
from stylo_attribution import make_docs, rank_candidates, sample_indices, separation_score
from stylo_features import (CACHE_DIR, build_cache, build_count_matrix,
                            char_ngrams, clean_lyrics, export_dir, load_corpus,
                            tokenize)

RESULT_DIR = export_dir()

RAW = Path(".cache_lex/web7_raw.json")
WEB7_ID = 1078135
TARGET = "Ziak"
CHALLENGER = "web7"
T_CAND = 12_000
N_REPEATS = 30
SEED = 20260916
ANNEE_CONTEMPORAINE = 2019

# Voisins stylistiques de Ziak identifiés dans l'article : repères de rang.
VOISINS = ["Kerchak", "Beendo Z", "ISK", "Werenoi", "Zkr", "Rimkus"]

# Segments entre parenthèses : chœurs et ad-libs dans les conventions de Genius.
# Bornés à la ligne pour qu'une parenthèse non refermée n'avale pas le texte.
PARENTHESES = re.compile(r"\([^)\n]*\)")


def charge_web7() -> tuple[pd.DataFrame, list[str]]:
    """Titres de web7 retenus selon les critères LRFAF, et motifs d'exclusion."""
    raw = json.load(open(RAW, encoding="utf-8"))
    kept, motifs = [], []
    vus = set()
    for s in raw:
        feat = " ".join(s.get("featured_artists") or []).lower()
        if s["primary_artist_id"] != WEB7_ID:
            motifs.append(f"{s['title']} — artiste principal : {s['primary_artist']}")
        elif s.get("language") != "fr":
            motifs.append(f"{s['title']} — langue Genius : {s.get('language')}")
        elif not s.get("lyrics_raw"):
            motifs.append(f"{s['title']} — paroles absentes")
        elif "ziak" in feat or "mikeysem" in feat:
            # Un titre partagé mettrait les mêmes mots des deux côtés du test.
            motifs.append(f"{s['title']} — featuring avec {feat}")
        else:
            lyrics = P.strip_genius_header(P.lg_clean(s["lyrics_raw"]))
            cle = clean_lyrics(lyrics)[:300]
            if cle in vus:
                motifs.append(f"{s['title']} — doublon de paroles")
                continue
            vus.add(cle)
            kept.append({"title": s["title"], "year": s.get("year"), "lyrics": lyrics})
    return pd.DataFrame(kept), motifs


def matrices(cache: dict, web7: pd.DataFrame, sans_parentheses: bool):
    """Comptes de 4-grammes pour le corpus + web7, avec ou sans parenthèses."""
    vocab = cache["vocab_char"]

    def prepare(t):
        return clean_lyrics(PARENTHESES.sub(" ", t) if sans_parentheses else t)

    if sans_parentheses:
        path = CACHE_DIR / "counts_char_sans_parentheses.npz"
        tok_path = CACHE_DIR / "n_tokens_sans_parentheses.npy"
        if path.exists() and tok_path.exists():
            mat_c, ntok_c = sparse.load_npz(path), np.load(tok_path)
        else:
            print("  recalcul des comptes sans parenthèses (une fois, ~1 min)...")
            df = load_corpus()
            if list(df["artist"]) != list(cache["meta"]["artist"]):
                raise SystemExit("ordre du corpus différent du cache : reconstruire le cache")
            textes = [prepare(t) for t in df["lyrics"]]
            mat_c = build_count_matrix([char_ngrams(t) for t in textes], vocab)
            ntok_c = np.array([len(tokenize(t)) for t in textes])
            sparse.save_npz(path, mat_c)
            np.save(tok_path, ntok_c)
    else:
        mat_c, ntok_c = cache["counts_char"], cache["meta"]["n_tokens"].to_numpy()

    textes_w = [prepare(t) for t in web7["lyrics"]]
    mat_w = build_count_matrix([char_ngrams(t) for t in textes_w], vocab)
    ntok_w = np.array([len(tokenize(t)) for t in textes_w])

    meta = pd.concat([cache["meta"][["artist", "year"]],
                      pd.DataFrame({"artist": CHALLENGER, "year": web7["year"]})],
                     ignore_index=True)
    return (sparse.vstack([mat_c, mat_w]).tocsr(),
            np.concatenate([ntok_c, ntok_w]), meta)


def classement(mat, n_tok, meta, rng, requete=TARGET, cible=CHALLENGER,
               annee_min=None) -> pd.DataFrame:
    """Rang de `cible` et des voisins de référence, vus depuis `requete`."""
    ids = {a: meta.index[meta["artist"] == a].to_numpy()
           for a in meta["artist"].unique()}
    if annee_min is not None:
        c = ids[cible]
        ids[cible] = c[pd.to_numeric(meta.loc[c, "year"], errors="coerce").fillna(0)
                       .to_numpy() >= annee_min]
    totals = {a: int(n_tok[i].sum()) for a, i in ids.items()}
    t_query = min(totals[requete], totals[TARGET])
    pool = sorted(a for a, t in totals.items() if t >= T_CAND and a != requete)
    if cible not in pool:
        raise SystemExit(f"{cible} : {totals[cible]} mots, sous le seuil de {T_CAND}")

    rows = []
    for rep in range(N_REPEATS):
        q_ids = sample_indices(ids[requete], n_tok[ids[requete]], t_query, rng)
        groups = {a: sample_indices(ids[a], n_tok[ids[a]], T_CAND, rng) for a in pool}
        names, freqs = make_docs(mat, groups)
        q = np.asarray(mat[q_ids].sum(axis=0)).ravel()
        q = q / max(q.sum(), 1.0)
        d = rank_candidates(q, freqs, "cosine_delta")
        order = np.argsort(d)
        pos = {names[j]: k + 1 for k, j in enumerate(order)}
        row = {"rep": rep, "rang_cible": pos[cible], "n_candidats": len(names),
               "sep_cible": separation_score(d, names.index(cible)),
               "sep_top1": separation_score(d, int(order[0])),
               "top1": names[int(order[0])]}
        for v in VOISINS:
            if v in pos:
                row[f"rang_{v}"] = pos[v]
        rows.append(row)
    return pd.DataFrame(rows)


def resume(nom: str, r: pd.DataFrame) -> dict:
    out = {"variante": nom,
           "rang_median_web7": float(r["rang_cible"].median()),
           "rang_min": int(r["rang_cible"].min()),
           "rang_max": int(r["rang_cible"].max()),
           "top20_%": 100 * float((r["rang_cible"] <= 20).mean()),
           "sep_web7": float(r["sep_cible"].mean()),
           "n_candidats": int(r["n_candidats"].iloc[0]),
           "top1_modal": r["top1"].mode().iloc[0]}
    for v in VOISINS:
        if f"rang_{v}" in r:
            out[f"rang_{v}"] = float(r[f"rang_{v}"].median())
    return out


def main() -> None:
    if not RAW.exists():
        raise SystemExit(f"{RAW} absent : lancer `python3 collecte_genius.py {WEB7_ID} {RAW}`")
    web7, motifs = charge_web7()
    print(f"web7 : {len(web7)} titres retenus, {len(motifs)} exclus")
    for m in motifs[:12]:
        print(f"  exclu : {m}")
    if len(motifs) > 12:
        print(f"  ... et {len(motifs) - 12} autres")

    cache = build_cache()
    rng = np.random.default_rng(SEED)
    resumes, bruts = [], []

    for nom, sans_par, annee in [
        ("textes complets", False, None),
        ("sans ad-libs entre parenthèses", True, None),
        (f"web7 depuis {ANNEE_CONTEMPORAINE}, sans ad-libs", True, ANNEE_CONTEMPORAINE),
    ]:
        mat, n_tok, meta = matrices(cache, web7, sans_par)
        idx_w7 = meta.index[meta["artist"] == CHALLENGER]
        if annee is not None:
            ans = pd.to_numeric(meta.loc[idx_w7, "year"], errors="coerce").fillna(0)
            idx_w7 = idx_w7[ans.to_numpy() >= annee]
        mots_w7 = int(n_tok[idx_w7].sum())
        print(f"\n=== {nom} (web7 : {mots_w7:,} mots) ===".replace(",", " "))
        r = classement(mat, n_tok, meta, rng, annee_min=annee)
        r["variante"] = nom
        bruts.append(r)
        s = resume(nom, r)
        s["n_titres_web7"] = int(len(idx_w7))
        s["mots_web7"] = mots_w7
        resumes.append(s)
        print(f"  rang médian de web7 : {s['rang_median_web7']:.0f} / {s['n_candidats']}"
              f"  (min {s['rang_min']}, max {s['rang_max']})")
        print(f"  dans le top 20 : {s['top20_%']:.0f} % des tirages"
              f"   séparation : {s['sep_web7']:+.2f}")
        print("  repères — " + ", ".join(
            f"{v} {s[f'rang_{v}']:.0f}" for v in VOISINS if f"rang_{v}" in s))

    # Symétrie : depuis les textes de web7, Ziak ressort-il ?
    mat, n_tok, meta = matrices(cache, web7, True)
    inv = classement(mat, n_tok, meta, rng, requete=CHALLENGER, cible=TARGET)
    inv["variante"] = "inverse : Ziak vu depuis web7"
    bruts.append(inv)
    print(f"\n=== Test inverse (sans ad-libs) : rang de Ziak vu depuis web7 ===")
    print(f"  rang médian {inv['rang_cible'].median():.0f} / {inv['n_candidats'].iloc[0]}"
          f"   top 20 : {100 * (inv['rang_cible'] <= 20).mean():.0f} %")
    resumes.append({"variante": "inverse : Ziak vu depuis web7",
                    "rang_median_web7": float(inv["rang_cible"].median()),
                    "top20_%": 100 * float((inv["rang_cible"] <= 20).mean()),
                    "sep_web7": float(inv["sep_cible"].mean()),
                    "n_candidats": int(inv["n_candidats"].iloc[0])})

    pd.concat(bruts, ignore_index=True).to_csv(RESULT_DIR / "15_1_web7_bruts.csv", index=False)
    pd.DataFrame(resumes).to_csv(RESULT_DIR / "15_2_web7_synthese.csv", index=False)

    # Repères issus de la section 6 : ce que donne un lien d'auteur réel.
    reel = pd.read_csv(RESULT_DIR / "13_2_alias_reels_synthese.csv")
    temp = pd.read_csv(RESULT_DIR / "14_2_alias_temporel_synthese.csv")
    temp = temp[temp["groupe"] == "changement d'identité"]
    print("\n=== REPÈRES : liens d'auteur réels (section 6) ===")
    print(f"  artiste → son groupe      : rang médian {reel.rang_median.median():.0f},"
          f" top 20 pour {100 * (reel.rang_median <= 20).mean():.0f} % des paires")
    print(f"  artiste → son groupe (duo): rang médian "
          f"{reel[reel.n_rappeurs == 2].rang_median.median():.0f}")
    print(f"  changement d'identité     : rang médian {temp.rang_median.median():.0f},"
          f" top 20 pour {100 * (temp.rang_median <= 20).mean():.0f} % des cas")


if __name__ == "__main__":
    main()
