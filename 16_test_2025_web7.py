#!/usr/bin/env python3
"""Les titres de Ziak co-écrits avec web7 ressemblent-ils davantage à web7 ?

Genius crédite web7 (ex-7 Jaws) comme co-auteur de 10 titres de Ziak sur 75.
L'année 2025 offre une expérience presque naturelle : 23 titres du même artiste,
la même année, dont 8 crédités à web7 et 15 non. Si la contribution de web7 laisse
une trace stylométrique, les 8 titres crédités doivent être plus proches de ses
propres textes que ne l'est un ensemble quelconque de 8 titres de Ziak 2025.

**Test de permutation.** On mesure la distance entre le profil agrégé des 8 titres
crédités et le profil de web7, puis on la compare à celle de milliers de
sous-ensembles de 8 titres tirés au hasard parmi les 23. La taille du groupe est
ainsi identique sous l'hypothèse nulle, ce qui neutralise l'effet de taille.

**Contrôles de puissance.** Un échantillon de 8 titres est petit : un résultat
négatif ne vaut que si l'on sait ce que le test aurait détecté. On remplace donc
les titres crédités par des titres de web7 lui-même (100 % web7), puis par un
mélange moitié web7, moitié Ziak — ce que produirait une co-écriture à parts
égales dans le propre style de web7.

**Featurings et ad-libs retirés.** Les couplets d'invités sont retirés des paroles
de Ziak comme de web7 (voir `genius_sections.py`), ainsi que les segments entre
parenthèses, qui notent surtout les ad-libs de l'interprète. Les artistes de
LRFAF, dont les balises de section ont disparu, ne servent ici qu'à fixer
l'échelle de standardisation des traits : leurs featurings résiduels n'y pèsent
que marginalement.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import lrfaf_pipeline as P
from genius_sections import texte_selon_option
from stylo_attribution import make_docs, rank_candidates, sample_indices, separation_score
from stylo_features import (CACHE_DIR, OPTION_FEATURINGS, build_cache,
                            build_count_matrix, char_ngrams, clean_lyrics,
                            export_dir, tokenize)

RESULT_DIR = export_dir()
ZIAK_RAW = Path(".cache_lex/ziak_raw.json")
WEB7_RAW = Path(".cache_lex/web7_raw.json")
CREDITS = RESULT_DIR / "15_3_credits_auteurs_ziak.csv"
ZIAK_ID, WEB7_ID = 2113831, 1078135
ALIAS = {"Ziak": {"Ziak"},
         "web7": {"web7", "7 Jaws", "7Jaws", "JawsLee", "SeptMachoires"}}
PARENTHESES = re.compile(r"\([^)\n]*\)")
T_CAND = 12_000
N_PERM = 5000
N_CONTROLES = 200
SEED = 20260917


def prepare(raw: str, alias: set[str], sans_parentheses: bool,
            featured: list[str] | None = None) -> tuple[str | None, dict]:
    """Paroles brutes Genius -> texte d'analyse, sans invités ni ad-libs.

    None si le titre est écarté (option 1 : titre comportant un invité).
    """
    texte, stats = texte_selon_option(P.strip_genius_header(raw), alias,
                                      OPTION_FEATURINGS, featured)
    if texte is None:
        return None, stats
    texte = P.lg_clean(texte)
    if sans_parentheses:
        texte = PARENTHESES.sub(" ", texte)
    return clean_lyrics(texte), stats


def charge(path: Path, artiste_id: int, alias: set[str], sans_par: bool,
           exclure_feat: tuple[str, ...] = ()) -> pd.DataFrame:
    rows, vus = [], set()
    for s in json.load(open(path, encoding="utf-8")):
        feat = " ".join(s.get("featured_artists") or []).lower()
        if (s["primary_artist_id"] != artiste_id or s.get("language") != "fr"
                or not s.get("lyrics_raw") or any(e in feat for e in exclure_feat)):
            continue
        texte, st = prepare(s["lyrics_raw"], alias, sans_par, s.get("featured_artists"))
        if texte is None:
            continue
        cle = texte[:300]
        if len(tokenize(texte)) < 50 or cle in vus:
            continue
        vus.add(cle)
        rows.append({"title": s["title"], "url": s["url"], "year": s.get("year"),
                     "texte": texte, "n_tokens": len(tokenize(texte)), **st})
    return pd.DataFrame(rows)


def reference(cache, sans_par: bool, rng) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne et écart-type des traits sur les artistes de LRFAF (12 000 mots)."""
    if sans_par:
        mat = sparse.load_npz(CACHE_DIR / "counts_char_sans_parentheses.npz")
        ntok = np.load(CACHE_DIR / "n_tokens_sans_parentheses.npy")
    else:
        mat, ntok = cache["counts_char"], cache["meta"]["n_tokens"].to_numpy()
    meta = cache["meta"]
    ids = {a: meta.index[meta["artist"] == a].to_numpy() for a in meta["artist"].unique()}
    pool = [a for a, i in ids.items() if ntok[i].sum() >= T_CAND and a != "Ziak"]
    mus, sds = [], []
    for _ in range(5):
        groups = {a: sample_indices(ids[a], ntok[ids[a]], T_CAND, rng) for a in pool}
        _, freqs = make_docs(mat, groups)
        mus.append(freqs.mean(axis=0))
        sds.append(freqs.std(axis=0, ddof=0))
    sd = np.mean(sds, axis=0)
    return np.mean(mus, axis=0), np.where(sd < 1e-12, 1.0, sd)


class Espace:
    """Profils z-standardisés et distance Cosine Delta entre groupes de titres."""

    def __init__(self, counts: sparse.csr_matrix, mu: np.ndarray, sd: np.ndarray):
        # Quelques centaines de titres : la forme dense rend les milliers de
        # tirages des tests de permutation quasi instantanés.
        self.dense, self.mu, self.sd = counts.toarray(), mu, sd

    def profil(self, idx) -> np.ndarray:
        v = self.dense[np.asarray(idx)].sum(axis=0)
        z = (v / max(v.sum(), 1.0) - self.mu) / self.sd
        return z / (np.linalg.norm(z) + 1e-12)

    def distance(self, idx_a, idx_b) -> float:
        return float(1.0 - self.profil(idx_a) @ self.profil(idx_b))


def permutation(esp: Espace, groupe, population, web7_idx, rng, n=N_PERM):
    """Distance observée du groupe à web7, et p-valeur contre des tirages aléatoires."""
    ref = esp.profil(web7_idx)
    obs = float(1.0 - esp.profil(groupe) @ ref)
    k = len(groupe)
    nulle = np.array([1.0 - esp.profil(rng.choice(population, k, replace=False)) @ ref
                      for _ in range(n)])
    return obs, float((nulle <= obs).mean()), nulle


def main() -> None:
    for p in (ZIAK_RAW, WEB7_RAW, CREDITS):
        if not p.exists():
            raise SystemExit(f"{p} absent")
    cache = build_cache()
    vocab = cache["vocab_char"]
    credits = pd.read_csv(CREDITS)
    credite = dict(zip(credits["url"], credits["web7"]))
    rng = np.random.default_rng(SEED)
    lignes, distributions, controles = [], [], []

    for sans_par in (True, False):
        variante = "sans featurings ni ad-libs" if sans_par else "sans featurings"
        ziak = charge(ZIAK_RAW, ZIAK_ID, ALIAS["Ziak"], sans_par)
        web7 = charge(WEB7_RAW, WEB7_ID, ALIAS["web7"], sans_par,
                      exclure_feat=("ziak", "mikeysem"))
        ziak["web7_credite"] = ziak["url"].map(credite).fillna(False).astype(bool)

        textes = list(ziak["texte"]) + list(web7["texte"])
        counts = build_count_matrix([char_ngrams(t) for t in textes], vocab)
        nz = len(ziak)
        idx_z = np.arange(nz)
        idx_w = np.arange(nz, nz + len(web7))
        mu, sd = reference(cache, sans_par, rng)
        esp = Espace(counts, mu, sd)

        print(f"\n{'=' * 72}\nVARIANTE : {variante}\n{'=' * 72}")
        print(f"Featurings retirés — Ziak : {ziak.mots_retires.sum()} mots "
              f"({100 * ziak.mots_retires.sum() / (ziak.mots_retires.sum() + ziak.mots_gardes.sum()):.1f} %), "
              f"web7 : {web7.mots_retires.sum()} mots "
              f"({100 * web7.mots_retires.sum() / (web7.mots_retires.sum() + web7.mots_gardes.sum()):.1f} %)")

        # --- Expérience 2025 -------------------------------------------------
        z25 = ziak[pd.to_numeric(ziak["year"], errors="coerce") == 2025]
        cred = z25.index[z25["web7_credite"]].to_numpy()
        non = z25.index[~z25["web7_credite"]].to_numpy()
        pop = z25.index.to_numpy()
        print(f"\nZiak 2025 : {len(pop)} titres — {len(cred)} crédités à web7 "
              f"({int(ziak.loc[cred, 'n_tokens'].sum())} mots), {len(non)} non crédités "
              f"({int(ziak.loc[non, 'n_tokens'].sum())} mots)")
        obs, p, nulle = permutation(esp, cred, pop, idx_w, rng)
        d_non = esp.distance(non, idx_w)
        distributions.append(pd.DataFrame({"variante": variante, "distance_hasard": nulle}))
        print(f"  distance à web7 — titres crédités : {obs:.4f} | non crédités : {d_non:.4f}")
        print(f"  tirages aléatoires de {len(cred)} titres : médiane {np.median(nulle):.4f}")
        print(f"  p-valeur (crédités plus proches que le hasard) : {p:.3f}")
        tot = lambda d: int(d.mots_retires.sum() + d.mots_gardes.sum())
        lignes.append({"variante": variante, "test": "2025 : titres crédités à web7",
                       "n_titres": len(cred), "distance": obs,
                       "distance_mediane_hasard": float(np.median(nulle)), "p_valeur": p,
                       "mots_credites": int(ziak.loc[cred, "n_tokens"].sum()),
                       "n_non_credites": len(non),
                       "mots_non_credites": int(ziak.loc[non, "n_tokens"].sum()),
                       "distance_non_credites": d_non,
                       "n_titres_2025": len(pop),
                       "n_titres_web7": len(web7), "mots_web7": int(web7.n_tokens.sum()),
                       "pct_featurings_ziak": 100 * ziak.mots_retires.sum() / tot(ziak),
                       "pct_featurings_web7": 100 * web7.mots_retires.sum() / tot(web7)})

        # --- Contrôles de puissance ------------------------------------------
        k = len(cred)
        for part_web7, nom in [(1.0, "contrôle : 100 % web7"), (0.5, "contrôle : 50 % web7")]:
            n_w = int(round(k * part_web7))
            detect = []
            for _ in range(N_CONTROLES):
                tenus = rng.choice(idx_w, n_w, replace=False)
                ref_w = np.setdiff1d(idx_w, tenus)
                groupe = np.concatenate([tenus, rng.choice(non, k - n_w, replace=False)])
                pw = esp.profil(ref_w)
                o = 1.0 - esp.profil(groupe) @ pw
                nul = np.array([1.0 - esp.profil(rng.choice(pop, k, replace=False)) @ pw
                                for _ in range(400)])
                detect.append((nul <= o).mean())
            detect = np.array(detect)
            controles.append(pd.DataFrame({"variante": variante, "controle": nom,
                                           "p_valeur": detect}))
            print(f"  {nom:24} : détecté (p < 0,05) dans {100 * (detect < 0.05).mean():.0f} % "
                  f"des tirages, p médiane {np.median(detect):.3f}")
            lignes.append({"variante": variante, "test": nom, "n_titres": k,
                           "taux_detection_p05": float((detect < 0.05).mean()),
                           "p_valeur": float(np.median(detect))})

        # --- Toutes années confondues (époque non contrôlée) -----------------
        cred_all = ziak.index[ziak["web7_credite"]].to_numpy()
        obs_a, p_a, _ = permutation(esp, cred_all, idx_z, idx_w, rng, n=3000)
        print(f"\nToutes années : {len(cred_all)} titres crédités sur {nz}, "
              f"p-valeur {p_a:.3f} (époque non contrôlée)")
        lignes.append({"variante": variante, "test": "toutes années : titres crédités",
                       "n_titres": len(cred_all), "distance": obs_a, "p_valeur": p_a})

        # --- Classement général, invités retirés -----------------------------
        z_ante = idx_z[pd.to_numeric(ziak["year"], errors="coerce").fillna(0).to_numpy() <= 2024]
        if sans_par:
            mat = sparse.load_npz(CACHE_DIR / "counts_char_sans_parentheses.npz")
            ntok = np.load(CACHE_DIR / "n_tokens_sans_parentheses.npy")
        else:
            mat, ntok = cache["counts_char"], cache["meta"]["n_tokens"].to_numpy()
        meta = cache["meta"]
        ids = {a: meta.index[meta["artist"] == a].to_numpy() for a in meta["artist"].unique()}
        pool = [a for a, i in ids.items() if ntok[i].sum() >= T_CAND and a != "Ziak"]
        ntok_w = web7["n_tokens"].to_numpy()
        ntok_z = ziak["n_tokens"].to_numpy()
        rangs = []
        for _ in range(20):
            groups = {a: sample_indices(ids[a], ntok[ids[a]], T_CAND, rng) for a in pool}
            names, freqs = make_docs(mat, groups)
            w_ids = sample_indices(np.arange(len(web7)), ntok_w, T_CAND, rng)
            vw = np.asarray(counts[idx_w[w_ids]].sum(axis=0)).ravel()
            q_ids = sample_indices(z_ante, ntok_z[z_ante], 23_886, rng)
            vq = np.asarray(counts[q_ids].sum(axis=0)).ravel()
            cand = np.vstack([freqs, vw / max(vw.sum(), 1)])
            d = rank_candidates(vq / max(vq.sum(), 1), cand, "cosine_delta")
            rangs.append(int(np.where(np.argsort(d) == len(names))[0][0]) + 1)
        print(f"Classement général (Ziak 2020-2024 sans invités) : web7 au rang médian "
              f"{np.median(rangs):.0f} / {len(pool) + 1}, top 20 dans "
              f"{100 * np.mean(np.array(rangs) <= 20):.0f} % des tirages")
        lignes.append({"variante": variante, "test": "classement général sans invités",
                       "rang_median": float(np.median(rangs)),
                       "top20": float(np.mean(np.array(rangs) <= 20))})

    pd.DataFrame(lignes).to_csv(RESULT_DIR / "16_1_test_2025_web7.csv", index=False)
    pd.concat(distributions).to_csv(RESULT_DIR / "16_2_permutation_2025.csv", index=False)
    pd.concat(controles).to_csv(RESULT_DIR / "16_3_controles_puissance.csv", index=False)
    print(f"\n-> {RESULT_DIR / '16_1_test_2025_web7.csv'}")


if __name__ == "__main__":
    main()
