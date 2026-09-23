#!/usr/bin/env python3
"""Reconstruit le corpus en ne gardant, pour chaque titre, que le texte de l'artiste.

Entrée : les paroles balisées re-collectées par `19_collecte_corpus.py`.
Sortie : `corpus_sans_invites.csv`, de même schéma que LRFAF, dont la colonne
`lyrics` ne contient plus les strophes attribuées à quelqu'un d'autre.

**Le piège des groupes.** La règle naïve — « retirer toute section nommant un
autre que l'artiste » — vide les groupes de leur contenu : chez S-Crew, les
balises nomment Nekfeu, Framal, Mekra, dont les couplets *sont* le texte du
groupe. Un invité, lui, n'apparaît que sur un ou deux titres.

D'où la règle retenue : pour chaque artiste, un intervenant est réputé
**interne** s'il figure sur au moins `PART_MEMBRE` de ses titres balisés, et
**invité** sinon. Seules les sections des invités sont retirées. Le seuil est
reporté dans `export/20_2_membres_detectes.csv`, artiste par artiste, pour que
la décision reste vérifiable plutôt que cachée dans le code.

Deux garde-fous complètent la règle :

- un artiste dont le nettoyage retirerait plus de `GARDE_FOU` de ses mots est
  laissé intact et signalé — signe que la détection de membres a échoué ;
- un titre sans aucune balise est conservé tel quel : l'absence de balise
  n'est pas une preuve d'absence d'invité, et retirer ces titres biaiserait le
  corpus vers les artistes les mieux annotés.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import lrfaf_pipeline as P
from genius_sections import (LIGNE_BALISE, _norme, est_balise_de_section,
                             interpretes, retire_featurings)
from stylo_features import corpus_brut_csv, export_dir

COLLECTE = Path(".cache_lex/corpus_balises.jsonl")
SORTIE = Path("corpus_sans_invites.csv")
RESULT_DIR = export_dir()

PART_MEMBRE = 0.40   # part des titres d'un artiste au-delà de laquelle
                     # un intervenant est considéré comme membre, non invité
GARDE_FOU = 0.50     # au-delà, on renonce à nettoyer l'artiste


def charge_collecte() -> dict[str, dict]:
    """Paroles balisées, indexées par URL."""
    if not COLLECTE.exists():
        raise SystemExit(f"{COLLECTE} absent — lancer d'abord 19_collecte_corpus.py")
    out = {}
    with open(COLLECTE, encoding="utf-8") as f:
        for ligne in f:
            try:
                r = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            out[r["url"]] = r
    return out


def intervenants(texte: str) -> set[str]:
    """Noms normalisés nommés par les balises de section d'un titre."""
    noms = set()
    for ligne in texte.split("\n"):
        if est_balise_de_section(ligne):
            n = interpretes(LIGNE_BALISE.match(ligne).group(1))
            if n:
                noms |= n
    return noms


def membres_par_artiste(collecte: dict, urls: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pour chaque artiste, les intervenants récurrents — réputés internes."""
    par_artiste = defaultdict(Counter)
    n_titres = Counter()
    for url, artiste in zip(urls.url, urls.artist):
        r = collecte.get(url)
        if r is None:
            continue
        n_titres[artiste] += 1
        par_artiste[artiste].update(intervenants(P.strip_genius_header(r["lyrics_raw"])))

    membres, lignes = {}, []
    for artiste, compte in par_artiste.items():
        n = n_titres[artiste]
        internes = {_norme(artiste)}
        for nom, k in compte.items():
            if n and k / n >= PART_MEMBRE:
                internes.add(nom)
                lignes.append({"artiste": artiste, "intervenant": nom,
                               "titres": k, "sur": n, "part": k / n})
        membres[artiste] = internes
    return membres, pd.DataFrame(lignes).sort_values(["artiste", "part"],
                                                     ascending=[True, False])


def main() -> None:
    collecte = charge_collecte()
    corpus = pd.read_csv(corpus_brut_csv())
    print(f"{len(collecte):,} titres re-collectés ; corpus LRFAF : {len(corpus):,} lignes")

    couverts = corpus[corpus.url.isin(collecte)]
    membres, tab_membres = membres_par_artiste(collecte, couverts[["url", "artist"]])
    print(f"{len(membres)} artistes couverts ; "
          f"{len(tab_membres)} intervenants réputés membres\n")

    lyrics, stats = [], []
    for r in corpus.itertuples():
        rec = collecte.get(r.url)
        if rec is None:
            lyrics.append(r.lyrics)
            stats.append({"artist": r.artist, "url": r.url, "etat": "non recollecté",
                          "mots_gardes": 0, "mots_retires": 0})
            continue
        brut = P.strip_genius_header(rec["lyrics_raw"])
        sans, st = retire_featurings(brut, membres.get(r.artist, {_norme(r.artist)}))
        lyrics.append(P.lg_clean(sans))
        stats.append({"artist": r.artist, "url": r.url, "etat": "nettoyé", **st})

    st = pd.DataFrame(stats)
    # Garde-fou : un artiste amputé de plus de la moitié de ses mots signale que
    # la détection de membres a échoué ; on le laisse alors intact.
    part = st.groupby("artist").apply(
        lambda d: d.mots_retires.sum() / max(d.mots_gardes.sum() + d.mots_retires.sum(), 1),
        include_groups=False)
    suspects = set(part[part > GARDE_FOU].index)
    if suspects:
        print(f"garde-fou déclenché pour {len(suspects)} artiste(s) : "
              f"{', '.join(sorted(suspects)[:8])}\n")
    corpus["lyrics_sans_invites"] = lyrics
    garde = corpus.artist.isin(suspects) | corpus.url.map(lambda u: u not in collecte)
    corpus["lyrics"] = corpus.lyrics.where(garde, corpus.lyrics_sans_invites)
    corpus = corpus.drop(columns=["lyrics_sans_invites"])

    st["nettoye"] = ~st.artist.isin(suspects) & (st.etat == "nettoyé")
    total_g = st.loc[st.nettoye, "mots_gardes"].sum()
    total_r = st.loc[st.nettoye, "mots_retires"].sum()
    resume = pd.DataFrame([{
        "titres_corpus": len(corpus),
        "titres_recollectes": int((st.etat == "nettoyé").sum()),
        "artistes_nettoyes": int(st.loc[st.nettoye, "artist"].nunique()),
        "artistes_garde_fou": len(suspects),
        "mots_gardes": int(total_g), "mots_retires": int(total_r),
        "part_retiree": total_r / max(total_g + total_r, 1),
        "seuil_membre": PART_MEMBRE, "seuil_garde_fou": GARDE_FOU,
    }])
    print(resume.to_string(index=False))
    print(f"\n{100 * resume.part_retiree.iat[0]:.1f} % des mots retirés des titres nettoyés")

    corpus.to_csv(SORTIE, index=False)
    resume.to_csv(RESULT_DIR / "20_1_nettoyage_resume.csv", index=False)
    tab_membres.to_csv(RESULT_DIR / "20_2_membres_detectes.csv", index=False)
    (st.groupby("artist")
       .agg(mots_gardes=("mots_gardes", "sum"), mots_retires=("mots_retires", "sum"),
            titres=("url", "size"), nettoye=("nettoye", "any"))
       .assign(part_retiree=lambda d: d.mots_retires / (d.mots_gardes + d.mots_retires))
       .sort_values("part_retiree", ascending=False)
       .to_csv(RESULT_DIR / "20_3_nettoyage_par_artiste.csv"))
    print(f"écrit : {SORTIE}, {RESULT_DIR}/20_1 à 20_3")


if __name__ == "__main__":
    main()
