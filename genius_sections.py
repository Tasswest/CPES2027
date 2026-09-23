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

_LIBELLES_NORMES = {re.sub(r"[^a-z0-9]", "", l) for l in LIBELLES} | {"outro"}

LIGNE_BALISE = re.compile(r"^\s*\[([^\[\]\n]+)\]\s*$")
SEPARATEURS = re.compile(r"\s*(?:&|,|\+|/|\bet\b|\band\b|\bx\b|\bfeat\.?|\bft\.?)\s*",
                         re.IGNORECASE)


def _norme(nom: str) -> str:
    nom = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", nom.lower())


def _est_libelle(texte: str) -> bool:
    return any(_norme(texte).startswith(_norme(l)) for l in LIBELLES)


def _libelle_seul(nom: str) -> bool:
    """Un « nom » qui n'est qu'un libellé : « Outro », « Instrumental », « Couplet 1 »…"""
    n = re.sub(r"(x?\d+)+$", "", _norme(nom))
    return n in _LIBELLES_NORMES or n in ("instrumentale", "choeurs", "choeur")


def interpretes(balise: str) -> set[str] | None:
    """Interprètes nommés par une balise ; None si elle n'en nomme aucun."""
    contenu = re.sub(r"\([^)]*\)", "", balise).strip()      # « (x2) », « (Ziak) »…
    tiret = re.match(r"^([^:\-–—]+?)\s*[-–—]\s*(.+)$", contenu)
    if ":" in contenu:
        gauche, noms = contenu.split(":", 1)
        # Ordre inversé : « [Soprano : Outro] », « [Lefa : Couplet 1] ».
        if not _est_libelle(gauche) and _libelle_seul(noms):
            noms = gauche
    elif tiret and _est_libelle(tiret.group(1)):
        # Genius emploie le tiret aussi bien que le deux-points :
        # « [Couplet 3 - Kool Shen] ». Le libellé doit précéder, sinon
        # « Pre-refrain » ou « Jay-Z » seraient coupés en deux.
        noms = tiret.group(2)
    elif _est_libelle(contenu):
        return None
    else:
        noms = contenu                                        # « [Sofiane] »
    # « [Refrain - Outro] », « [Pont : Instrumental] » : deux libellés, aucun nom.
    out = {_norme(n) for n in SEPARATEURS.split(noms)
           if _norme(n) and not _libelle_seul(n)}
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
    garde, n_garde, n_retire, n_sections = [], 0, 0, 0
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
            else:
                n_sections += 1
            continue
        n = len(re.findall(r"\w+", ligne))
        if courant_garde:
            garde.append(ligne)
            n_garde += n
        else:
            n_retire += n
    return "\n".join(garde), {"mots_gardes": n_garde, "mots_retires": n_retire,
                              "sections_invites": n_sections}


def texte_selon_option(texte: str, alias_principal: set[str], option: str,
                       featured: list[str] | None = None) -> tuple[str | None, dict]:
    """Applique l'une des deux façons d'écarter les featurings.

    - « parties » (option 2) : seules les sections d'invités sont retirées ;
    - « titres » (option 1) : un titre comportant un invité est écarté en
      entier, et la fonction renvoie None. Un invité se reconnaît à une section
      qui le nomme, ou, quand Genius la fournit, à la liste `featured`.

    Les deux options partent du même repérage, si bien qu'elles ne diffèrent
    que par ce qu'elles font d'un titre partagé.
    """
    sans, st = retire_featurings(texte, alias_principal)
    alias = {_norme(a) for a in alias_principal}
    invites = {_norme(f) for f in featured or []} - alias
    if option == "titres":
        return (None if st["sections_invites"] or invites else texte), st
    if option == "parties":
        return sans, st
    raise ValueError(f"option inconnue : {option!r}")
