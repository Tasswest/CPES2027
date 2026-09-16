#!/usr/bin/env python3
"""Retire des paroles Genius les sections interprétées par d'autres artistes.

Un couplet d'invité est écrit par l'invité : le laisser dans le corpus d'un
artiste rapproche mécaniquement cet artiste de tous ceux qu'il a invités. Les
paroles brutes de Genius permettent de l'éviter, car chaque section y porte une
balise qui nomme, le cas échéant, son interprète :

    [Couplet 1]                          -> artiste principal (convention Genius)
    [Refrain : Captaine Roshi & 7 Jaws]  -> deux interprètes
    [Sofiane]                            -> balise réduite au nom de l'interprète

Ce nettoyage n'est possible que sur des paroles collectées soi-même : LRFAF a
supprimé ces balises avant publication, si bien que les artistes du corpus
gardent leurs featurings.

Règle retenue, volontairement stricte : une section n'est conservée que si elle
est attribuée au seul artiste principal, ou si elle n'est attribuée à personne.
Une section partagée (« Lacrim & Ziak ») est retirée, faute de pouvoir savoir
qui en a écrit quoi.
"""

from __future__ import annotations

import re
import unicodedata

# Libellés de section : une balise qui commence par l'un d'eux sans nommer
# d'interprète désigne une section de l'artiste principal.
LIBELLES = ("refrain", "couplet", "intro", "outro", "pont", "pre-refrain",
            "pre refrain", "post-refrain", "post refrain", "chorus", "verse",
            "hook", "bridge", "interlude", "break", "instrumental", "breakdown",
            "prechorus", "pre-chorus", "post-chorus", "refrain", "partie",
            "part", "skit", "spoken", "paroles de", "texte", "freestyle")

LIGNE_BALISE = re.compile(r"^\s*\[([^\[\]\n]+)\]\s*$")
SEPARATEURS = re.compile(r"\s*(?:&|,|\+|/|\bet\b|\band\b|\bx\b|\bfeat\.?|\bft\.?)\s*",
                         re.IGNORECASE)


def _norme(nom: str) -> str:
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", nom.lower())


def interpretes(balise: str) -> set[str] | None:
    """Interprètes nommés par une balise ; None si elle n'en nomme aucun."""
    contenu = re.sub(r"\([^)]*\)", "", balise).strip()      # « (x2) », « (Ziak) »…
    if ":" in contenu:
        noms = contenu.split(":", 1)[1]
    elif any(_norme(contenu).startswith(_norme(l)) for l in LIBELLES):
        return None
    else:
        noms = contenu                                        # « [Sofiane] »
    out = {_norme(n) for n in SEPARATEURS.split(noms) if _norme(n)}
    return out or None


def est_balise_de_section(ligne: str) -> bool:
    m = LIGNE_BALISE.match(ligne)
    # « [?] » et « [si beau?] » notent des mots inaudibles, pas une section.
    return bool(m) and "?" not in m.group(1)


def retire_featurings(texte: str, alias_principal: set[str]) -> tuple[str, dict]:
    """Paroles réduites aux sections de l'artiste principal, et statistiques.

    `alias_principal` : noms sous lesquels l'artiste principal peut apparaître
    dans les balises (« web7 », « 7 Jaws », « 7Jaws »…).
    """
    alias = {_norme(a) for a in alias_principal}
    garde, n_garde, n_retire = [], 0, 0
    courant_garde = True
    for ligne in texte.split("\n"):
        if est_balise_de_section(ligne):
            contenu = LIGNE_BALISE.match(ligne).group(1)
            if _norme(contenu).startswith("parolesde"):
                continue                                      # métadonnée Genius
            noms = interpretes(contenu)
            courant_garde = noms is None or noms <= alias
            if courant_garde:
                garde.append(ligne)
            continue
        n = len(re.findall(r"\w+", ligne))
        if courant_garde:
            garde.append(ligne)
            n_garde += n
        else:
            n_retire += n
    return "\n".join(garde), {"mots_gardes": n_garde, "mots_retires": n_retire}
