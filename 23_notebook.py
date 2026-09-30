#!/usr/bin/env python3
"""Construit le notebook de l'étude, puis l'exécute pour en remplir les sorties.

Le notebook est le livrable ; ce script en est la source. L'écrire ainsi plutôt
qu'à la main garantit qu'il reste exécutable de bout en bout — chaque cellule
est relue par `nbclient` à la génération, et une erreur interrompt la
construction au lieu de passer inaperçue.

Le notebook calcule réellement là où c'est possible : une passe d'attribution
sur les 393 candidats prend moins d'une seconde, si bien que le diagnostic du
biais de taille, le contrôle positif et le classement de Ziak sont refaits en
direct. Les résultats qui demandent des heures — la validation sur 177
artistes, la re-collecte des balises, l'étude rejouée sur trois corpus — sont
relus dans `export/`, et le notebook le dit à chaque fois.

Usage : python3 23_notebook.py [--sans-execution]
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

SORTIE = Path("02_stylometrie_ziak.ipynb")

C = []          # cellules, dans l'ordre


def md(texte: str) -> None:
    C.append(nbf.v4.new_markdown_cell(texte.strip("\n")))


def code(source: str) -> None:
    C.append(nbf.v4.new_code_cell(source.strip("\n")))


# =============================================================================
# Titre et plan
# =============================================================================
md("""
# Qui se cache derrière Ziak ?

**Enquête stylométrique sur l'identité d'un rappeur masqué — corpus LRFAF**

Tassilo Westphalen · CPES Sciences des données, Arts et Cultures — PSL / Louis-le-Grand

---

Ziak publie son premier morceau en 2020. Il apparaît cagoulé, ne donne aucune
identité civile, et trois hypothèses circulent parmi les auditeurs : ce serait
un rappeur déjà établi revenu sous un pseudonyme ; ce serait Mikeysem ; ou ses
textes seraient écrits par web7, anciennement 7 Jaws.

Ces trois hypothèses ont une propriété rare : **elles sont testables**. Les
habitudes d'écriture les moins conscientes d'un auteur — fréquence des mots
grammaticaux, enchaînements de caractères, élisions — forment une signature que
le changement de nom n'efface pas. Si un auteur en cache un autre, les textes
doivent porter la même signature.

L'enjeu de ce travail n'est cependant pas l'algorithme mais le **protocole**.
Une question d'attribution ne se règle pas en calculant des distances : elle se
règle en sachant ce que ces distances valent. La section 3 montre qu'une
démarche intuitive, appliquée aux mêmes données, désigne un artiste avec
assurance et se trompe trois fois sur quatre.

## Plan

1. Mise en place
2. Le corpus
3. Pourquoi l'approche intuitive échoue
4. Le protocole retenu
5. Ziak face aux 392 candidats
6. Validation sur des liens d'auteur réels
7. L'hypothèse Mikeysem
8. L'hypothèse web7
9. Les featurings faussent-ils le résultat ?
10. Portrait stylométrique de Ziak
11. Conclusion

**Ce qui est calculé ici, et ce qui est relu.** Une passe d'attribution sur les
393 candidats prend moins d'une seconde : les sections 3, 5 et 10 recalculent
donc tout en direct. Les résultats qui demandent des heures — la validation sur
177 artistes, la re-collecte de 34 666 pages, l'étude rejouée sur trois corpus —
sont relus dans `export/`, et chaque cellule concernée le signale.
""")

# =============================================================================
md("""
## 1. Mise en place

Trois modules portent tout le travail : `stylo_features` prépare le texte et
met les comptes en cache, `stylo_attribution` mesure les distances,
`genius_sections` sait lire les balises de Genius.
""")
code("""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stylo_features import build_cache
from stylo_attribution import (make_docs, rank_candidates, sample_indices,
                               separation_score)

EXPORT = Path("export")
# Les trois corpus comparés en section 9 : le publié, et les deux nettoyés.
CORPUS = {"publié": EXPORT,
          "sans strophes d'invités": EXPORT / "sans_invites",
          "sans titres à invités": EXPORT / "sans_feats"}
T_CAND = 12_000        # taille d'un document candidat, en mots
GRAINE = 20260930

def milliers(n, d=0):
    \"\"\"Nombre au format français : espace fine comme séparateur de milliers.\"\"\"
    return f"{n:,.{d}f}".replace(",", "\u202f")


pd.set_option("display.width", 120)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
""")

md("""
Le cache contient les comptes de traits **par chanson** : les 500 mots les plus
fréquents, les 3 000 suites de quatre caractères les plus fréquentes. C'est ce
découpage par chanson qui permet de recomposer un document d'artiste de
n'importe quelle taille par simple addition — sans quoi les centaines de
rééchantillonnages de ce notebook seraient inexécutables.
""")
code("""
cache = build_cache()
meta = cache["meta"]
n_tokens = meta["n_tokens"].to_numpy()
totaux = meta.groupby("artist")["n_tokens"].sum()

print(f"{milliers(len(meta))} titres retenus · {meta['artist'].nunique()} artistes")
print(f"{cache['counts_mfw'].shape[1]} mots · "
      f"{cache['counts_char'].shape[1]} 4-grammes de caractères")
""")

# =============================================================================
md("""
## 2. Le corpus

### 2.1 Vue d'ensemble

LRFAF rassemble 37 307 chansons issues de genius.com, croisées avec les
catégories Wikipédia et Wikidata. Après nettoyage — textes de moins de 100 mots
écartés, doublons de paroles entre artistes supprimés — il reste les titres
chargés ci-dessus.

Le seuil d'éligibilité de 12 000 mots vient du protocole : chaque candidat doit
pouvoir fournir un document de cette taille.
""")
code("""
eligibles = totaux[totaux >= T_CAND]
apercu = (pd.DataFrame({"titres": meta.groupby("artist").size(), "mots": totaux})
            .sort_values("mots", ascending=False))

print(f"Artistes éligibles (≥ {milliers(T_CAND)} mots) : {len(eligibles)}")
print(f"Corpus médian : {milliers(totaux.median())} mots")
apercu.head(8)
""")

md("""
### 2.2 Le cas Ziak

Ziak est représenté par 43 titres publiés entre 2020 et 2024. C'est un volume
modeste, mais très au-dessus du seuil de quelques milliers de mots requis par
les méthodes employées ici — la section 5.1 le vérifiera empiriquement plutôt
que de le supposer.
""")
code("""
ziak = meta[meta["artist"] == "Ziak"]
idx_ziak = ziak.index.to_numpy()

print(f"{len(ziak)} titres · {milliers(ziak['n_tokens'].sum())} mots · "
      f"{ziak['year'].min()}-{ziak['year'].max()}")
ziak[["title", "year", "n_tokens"]].sort_values("year").head(6)
""")

# =============================================================================
md("""
## 3. Pourquoi l'approche intuitive échoue

### 3.1 Le classement naïf

La démarche spontanée : concaténer toutes les chansons de chaque artiste,
vectoriser, puis classer les candidats par distance à Ziak. Elle produit un
palmarès d'apparence tout à fait convaincante.
""")
code("""
mat = cache["counts_char"]
pool = sorted(a for a in eligibles.index if a != "Ziak")
titres = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}

requete = np.asarray(mat[idx_ziak].sum(axis=0)).ravel()
requete = requete / requete.sum()

# Chaque artiste par son corpus ENTIER, distance cosinus brute : l'approche naïve.
noms, freqs = make_docs(mat, {a: titres[a] for a in pool})
cos = 1 - (freqs @ requete) / (np.linalg.norm(freqs, axis=1) * np.linalg.norm(requete))
naif = (pd.DataFrame({"artiste": noms, "distance": cos,
                      "mots du corpus": [totaux[a] for a in noms]})
          .sort_values("distance").reset_index(drop=True))
naif.head(10)
""")

md("""
### 3.2 Ce que ce classement mesure réellement

Le problème apparaît dès qu'on confronte ce palmarès à une variable qui n'a
rien à voir avec le style : **la simple quantité de texte disponible sur chaque
artiste**.

Le mécanisme est arithmétique, non musical. Plus un artiste a écrit, plus son
vecteur couvre de n-grammes, et plus il ressemble à n'importe quel texte. Un
corpus très fourni est comme un portrait-robot très vague : à force de tout
contenir, il finit par ressembler à tout le monde.
""")
code("""
from scipy.stats import spearmanr

rho, _ = spearmanr(naif["distance"], naif["mots du corpus"])
print(f"Corrélation de Spearman distance ↔ taille du corpus : ρ = {rho:.2f}")
print(f"Corpus médian des 20 « plus proches » : "
      f"{milliers(naif.head(20)['mots du corpus'].median())} mots")
print(f"Corpus médian de l'ensemble           : "
      f"{milliers(naif['mots du corpus'].median())} mots")

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.scatter(naif["mots du corpus"], naif["distance"], s=9, alpha=0.45, color="#2f6f9f")
ax.set_xscale("log")
ax.set_xlabel("Taille du corpus du candidat (mots, log)")
ax.set_ylabel("Distance à Ziak")
ax.set_title(f"Le classement naïf mesure surtout la taille (ρ = {rho:.2f})")
plt.show()
""")

md("""
### 3.3 L'arbitrage : quel classement a raison ?

L'argument décisif n'est pas qu'un classement change, mais qu'on peut **mesurer
lequel a raison**. On prélève chez un artiste un échantillon de la taille du
corpus de Ziak, on le traite comme un texte anonyme, et l'on regarde si la
méthode le rattache au reste de son œuvre.

Trois variantes sont soumises au même protocole de vérité-terrain, sur 177
artistes dont la réponse est connue.

> *Résultats relus dans `export/` : cette évaluation demande plusieurs minutes.*
""")
code("""
variantes = pd.read_csv(EXPORT / "02_3_puissance_par_variante.csv")
variantes
""")

md("""
L'approche naïve identifie le bon auteur dans **un quart des cas seulement**,
alors qu'elle produit un classement d'allure sérieuse. Ramener tous les
documents à une taille identique fait gagner près de 50 points, et standardiser
les traits près de 20 de plus.

**Sans étalonnage, un classement de distances ne permet aucune conclusion**,
quelle que soit sa netteté apparente. C'est le résultat méthodologique de ce
travail, et il vaut au-delà du cas Ziak.
""")

# =============================================================================
md("""
## 4. Le protocole retenu

Trois choix.

- **Les traits** : les 4-grammes de caractères et les 500 mots les plus
  fréquents. Des descripteurs qui captent des habitudes largement
  inconscientes plutôt que les thèmes abordés.
- **Les distances** : le Cosine Delta (Smith & Aldridge, 2011) et le Delta de
  Burrows (1992), calculées sur des traits standardisés.
- **La taille** : contrôlée. Chaque candidat est représenté par 12 000 mots
  exactement, échantillonnés parmi ses titres.

Chaque artiste de contrôle est traité *exactement* comme Ziak : on lui prélève
un texte-requête de la taille du corpus de Ziak, et un « jumeau » de 12 000
mots issu de titres **disjoints**, placé dans le pool sous une autre étiquette.
Deux conditions sont évaluées :

- **H₁** — le jumeau est présent : mesure la puissance de la méthode ;
- **H₀** — le jumeau est retiré : reproduit la situation d'un auteur absent du
  corpus, c'est-à-dire l'hypothèse à laquelle Ziak sera confronté.

> *Résultats relus : 1 770 essais sur 177 artistes.*
""")
code("""
puissance = pd.read_csv(EXPORT / "03_2_puissance_methode.csv")
puissance
""")

md("""
La meilleure combinaison — 4-grammes de caractères et Cosine Delta — retrouve
le bon auteur au premier rang dans 90 % des cas. **C'est cette puissance élevée
qui rendra un résultat négatif informatif** : si un alias existait dans le
corpus, la méthode aurait neuf chances sur dix de le désigner en tête.

Reste que la distance brute au meilleur candidat ne se lit pas seule. Un
classement désigne *toujours* un premier : ce n'est pas une information. On lui
substitue un **score de séparation** — de combien d'écarts-types le meilleur
candidat se détache-t-il des autres ? Cette grandeur, elle, est comparable
d'une requête à l'autre.
""")
code("""
seuils = pd.read_csv(EXPORT / "03_3_seuils_decision.csv")
seuils[["features", "metric", "sep_H1_correct_moy", "sep_H0_moy"]]
""")

# =============================================================================
md("""
## 5. Ziak face aux 392 candidats

### 5.1 Le style de Ziak est-il seulement détectable ?

Avant d'interpréter un échec d'identification, il faut écarter l'explication
triviale : un corpus trop petit ou trop hétérogène pour porter une signature.

On coupe donc le corpus de Ziak en deux moitiés disjointes et l'on cherche la
seconde depuis la première, parmi les 392 autres candidats. **Calculé ici.**
""")
code("""
rng = np.random.default_rng(GRAINE)
rangs_controle = []

for _ in range(10):
    perm = rng.permutation(idx_ziak)
    coupe = int(np.searchsorted(np.cumsum(n_tokens[perm]), T_CAND) + 1)
    moitie_a, moitie_b = perm[:coupe], perm[coupe:]

    groupes = {a: sample_indices(titres[a], n_tokens[titres[a]], T_CAND, rng)
               for a in pool}
    groupes["Ziak (moitié B)"] = moitie_b

    nm, fr = make_docs(mat, groupes)
    q = np.asarray(mat[moitie_a].sum(axis=0)).ravel()
    d = rank_candidates(q / q.sum(), fr, "cosine_delta")
    rangs_controle.append(int((d < d[nm.index("Ziak (moitié B)")]).sum()) + 1)

print(f"Rang de sa propre moitié, sur 10 tirages : {rangs_controle}")
print(f"→ retrouvée au premier rang dans "
      f"{100 * np.mean(np.array(rangs_controle) == 1):.0f} % des cas")
""")

md("""
Le style de Ziak est donc parfaitement détectable et son corpus suffisamment
homogène. **Aucun échec ultérieur ne pourra être imputé à une insuffisance de
matière.**

### 5.2 Le classement des 392 candidats

On classe maintenant les autres artistes, sur plusieurs rééchantillonnages et
avec les quatre combinaisons de traits et de distances. **Calculé ici.**
""")
code("""
resultats = []
for rep in range(15):
    groupes = {a: sample_indices(titres[a], n_tokens[titres[a]], T_CAND, rng)
               for a in pool}
    for traits, m in (("char", cache["counts_char"]), ("mfw", cache["counts_mfw"])):
        nm, fr = make_docs(m, groupes)
        q = np.asarray(m[idx_ziak].sum(axis=0)).ravel()
        q = q / q.sum()
        for dist in ("cosine_delta", "burrows_delta"):
            d = rank_candidates(q, fr, dist)
            j = int(np.argmin(d))
            resultats.append({"traits": traits, "distance": dist, "favori": nm[j],
                              "séparation": separation_score(d, j)})

res = pd.DataFrame(resultats)
(res.groupby(["traits", "distance"])
    .agg(favori_modal=("favori", lambda s: s.mode().iat[0]),
         n_favoris_différents=("favori", "nunique"),
         séparation_moyenne=("séparation", "mean")))
""")

md("""
Le résultat est éloquent **par son incohérence**. Chaque combinaison a son
favori, et aucun candidat ne s'impose d'une méthode à l'autre — alors que pour
les artistes de contrôle, dont la réponse est connue, les quatre combinaisons
convergent dans neuf cas sur dix.

### 5.3 Verdict

La séparation observée pour Ziak est maintenant confrontée aux deux
distributions de référence établies en section 4.

> *Relu : calcul du rapport de vraisemblance sur 30 rééchantillonnages.*
""")
code("""
verdict = pd.read_csv(EXPORT / "04_6_verdict_hypotheses.csv")
verdict[["features", "metric", "sep_ziak", "sep_H1_correct_moy", "sep_H0_moy",
         "hypothese_favorisee", "top1_modal"]]
""")

md("""
**Les quatre analyses concordent : l'hypothèse favorisée est H₀, celle d'un
auteur absent du corpus.** La séparation observée pour Ziak est non seulement
très loin de ce que produit un alias authentique, mais se situe dans la queue
de la distribution des auteurs absents.

Autrement dit, ce n'est pas seulement que la méthode ne trouve pas : **c'est
qu'elle trouve activement l'absence de correspondance**.
""")

# =============================================================================
md("""
## 6. Validation sur des liens d'auteur réels

Tout ce qui précède repose sur une validation par jumeaux *fabriqués* : on
coupe l'œuvre d'un artiste en deux et l'on cherche une moitié depuis l'autre.
C'est une tâche facile — les deux moitiés partagent la même époque, les mêmes
thèmes, le même producteur. **La puissance qu'on y mesure est donc une borne
optimiste**, et c'est l'objection la plus sérieuse qu'on puisse opposer au
verdict.

Deux réponses, avec des cas où la vérité est connue indépendamment du corpus.

### 6.1 Recouvrements entre un artiste et son groupe

Quatorze paires solo / groupe, où l'artiste a réellement écrit une partie des
textes du groupe. Le test est plus sévère qu'un alias : dans un trio, l'auteur
ne signe qu'un tiers du texte.
""")
code("""
alias_reels = pd.read_csv(EXPORT / "13_2_alias_reels_synthese.csv")
print(f"Lien retrouvé dans le top 20 : "
      f"{100 * (alias_reels.rang_median <= 20).mean():.0f} % des paires")
alias_reels.sort_values("rang_median")[["solo", "groupe", "n_rappeurs", "rang_median"]].head(8)
""")

md("""
L'effet de **dilution** est net : un duo se retrouve aisément, un groupe de huit
se perd dans le classement. Retrouver un auteur qui n'a écrit qu'une partie
d'un disque, c'est reconnaître une voix dans un chœur.

### 6.2 Un changement d'identité efface-t-il la signature ?

C'est l'objection de fond : un artiste qui se réinvente sous un autre nom
change peut-être aussi de manière d'écrire. Genius fusionnant les changements
de nom, on peut le tester en découpant ces artistes **de part et d'autre de
leur rupture**, puis en cherchant la période postérieure depuis l'antérieure.
""")
code("""
alias_temp = pd.read_csv(EXPORT / "14_2_alias_temporel_synthese.csv")
alias_temp[["libelle", "groupe", "rang_median", "sep_cible", "taux_rang1"]]
""")

md("""
**Joke → Ateyaba est retrouvé au premier rang sur 393 candidats, dans la
totalité des tirages** — alors même que le changement d'identité était
revendiqué : Ateyaba a publiquement déclaré vouloir « tuer Joke ».

Ces deux tests tirent dans des directions opposées, et il faut les lire
ensemble. La puissance est *plus faible* qu'annoncée dès lors que l'auteur ne
signe qu'une partie des textes ; mais elle ne s'effondre *pas* lorsqu'il change
d'identité — ce qui était la crainte principale, et ce qui correspond
précisément à l'hypothèse testée sur Ziak.
""")

# =============================================================================
md("""
## 7. L'hypothèse Mikeysem

La conclusion de la section 5 souffrait d'un angle mort : **le nom le plus
fréquemment avancé par les auditeurs ne figure pas dans LRFAF**. Ce n'est pas
un oubli du corpus mais une conséquence de son critère d'inclusion, qui part
des catégories Wikipédia — Mikeysem n'a ni article Wikipédia ni entrée
Wikidata. L'hypothèse la plus discutée était donc celle que l'étude ne pouvait
pas tester.

Ses titres ont été collectés sur Genius et passés dans le pipeline LRFAF
reconstitué. La couverture est partielle, et c'est la principale faiblesse de
cette section : sa discographie compte 21 titres, dont 7 seulement disposent
d'une page Genius, soit 3 745 mots.

**Cette petitesse impose de recalibrer avant d'interpréter quoi que ce soit.**
Un résultat négatif obtenu avec une méthode aveugle ne vaudrait rien.
""")
code("""
val_petite = pd.read_csv(EXPORT / "10_1_validation_petite_taille.csv")
h1 = val_petite[(val_petite.condition == "H1") & (val_petite.features == "char")
                & (val_petite.metric == "cosine_delta")]
print(f"À cette taille de corpus, un alias authentique est retrouvé :")
print(f"  au premier rang dans {100 * (h1.rank_twin == 1).mean():.0f} % des cas")
print(f"  dans le top 20 dans  {100 * (h1.rank_twin <= 20).mean():.0f} % des cas")
""")

md("""
La puissance baisse, mais un alias authentique resterait très majoritairement
détectable. Le test est donc interprétable.
""")
code("""
rangs_mike = pd.read_csv(EXPORT / "10_5_rangs_compares.csv", index_col=0)
rangs_mike
""")

md("""
**Mikeysem se classe derrière six artistes que personne ne soupçonne.** Son
score de séparation est même plus faible que celui d'un auteur typiquement
absent du corpus : il n'est pas un candidat ordinaire ayant manqué la première
place, mais un artiste particulièrement éloigné.

Une réserve accompagne ce résultat : un contrôle d'auto-cohérence — une moitié
de son corpus cherchant l'autre — échoue complètement. Ce test est plus
exigeant que le test principal, qui dispose des 23 886 mots de Ziak comme
requête, et ne remet donc pas en cause le classement ; mais avec la couverture
partielle signalée plus haut, il rappelle que **la conclusion de cette section
vaut comme faisceau convergent, non comme démonstration**.
""")

# =============================================================================
md("""
## 8. L'hypothèse web7

Une autre version de la rumeur, rapportée par la presse musicale, ne fait pas
de Ziak un seul artiste mais en répartit les rôles : web7, anciennement 7 Jaws,
écrirait les textes ; Mikeysem les interpréterait sous le masque.

C'est une hypothèse de *ghostwriting*, qui ne se teste pas tout à fait comme un
alias : un auteur qui écrit pour la voix d'un autre adapte son registre, ce qui
dilue sa signature.

### 8.1 Au niveau de l'artiste
""")
code("""
web7 = pd.read_csv(EXPORT / "15_2_web7_synthese.csv")
web7[["variante", "rang_median_web7", "n_candidats", "top20_%", "sep_web7"]]
""")

md("""
Quelle que soit la variante — textes complets, sans ad-libs, sans featurings,
ou restreinte à sa production contemporaine — **web7 se classe au-delà du
centième rang et n'entre jamais dans le top 20**. Dans le même temps, les
voisins habituels de Ziak restent dans les premiers rangs : la méthode
fonctionne normalement, elle ne trouve simplement pas web7.

### 8.2 Ce que disent les crédits, et l'expérience naturelle de 2025

L'examen de la collecte a pourtant fait apparaître un fait documentaire :
Genius crédite web7 comme co-auteur de 10 des 75 titres de Ziak. Or la section
6 a montré que la détection s'effondre quand l'auteur recherché ne signe qu'une
petite fraction des textes — un test mené au niveau de l'artiste ne pouvait
donc pas voir une contribution aussi minoritaire.

L'année 2025 offre alors une situation presque expérimentale : sur l'album
*Essonne History X*, 8 titres sont crédités à web7 et 15 ne le sont pas. Même
artiste, même année, même disque. On compare la distance du profil des 8 titres
crédités à celle de 5 000 tirages de 8 titres parmi les 23.
""")
code("""
t2025 = pd.read_csv(EXPORT / "16_1_test_2025_web7.csv")
t2025 = t2025[t2025.variante == "sans featurings ni ad-libs"]
t2025[["test", "n_titres", "distance", "distance_mediane_hasard", "p_valeur",
       "taux_detection_p05"]]
""")

md("""
**Les titres co-écrits avec web7 sont significativement plus proches de son
écriture que les autres titres de la période.** Les deux contrôles de puissance
montrent que le test pouvait le voir : un texte entièrement de web7 est détecté
dans 100 % des tirages, un mélange à parts égales dans plus de 80 %.

Crédits et stylométrie convergent donc vers une lecture sobre. **web7 est un
co-auteur réel d'une partie des titres, et rien n'indique qu'il écrive
l'essentiel de l'œuvre.** La leçon de méthode est nette : une attribution menée
au niveau de l'artiste ne voit pas un co-auteur minoritaire, qu'un contraste
ciblé à l'intérieur de l'œuvre permet de détecter.
""")

# =============================================================================
md("""
## 9. Les featurings faussent-ils le résultat ?

C'est l'objection la plus sérieuse qui reste, et elle mérite d'être prise au
sérieux : **un couplet d'invité est écrit par l'invité**. Le laisser dans le
corpus de l'artiste principal rapproche mécaniquement cet artiste de tous ceux
qu'il a invités — or l'hypothèse testée ici porte justement sur une proximité.

### 9.1 Ce que LRFAF a perdu

Sur Genius, chaque bloc de paroles porte le nom de celui qui le chante. Mais
**LRFAF a supprimé ces balises avant publication** : sur 400 textes tirés au
hasard, aucun n'en porte, et 15 titres sur 37 307 signalent un featuring dans
leur titre. L'information n'est pas récupérable depuis le corpus.

Elle l'est depuis Genius : le corpus conserve l'URL de chaque morceau, et la
page porte encore ses balises. Les 34 666 pages des artistes éligibles ont donc
été re-téléchargées — environ six heures de collecte.

### 9.2 Deux corpus propres

Reste à distinguer un membre de groupe d'un invité de passage. La règle naïve
viderait les groupes de leur contenu : chez IAM, les blocs nomment Akhenaton et
Shurik'n, dont les couplets *sont* le texte d'IAM. C'est la fréquence qui les
sépare — un nom qui revient dans une grande partie des morceaux est celui d'un
membre, et ses couplets sont gardés ; un nom qui n'apparaît qu'une ou deux fois
est celui d'un invité, et ses couplets sont retirés.

Deux corpus en sortent, et les deux sont testés.
""")
code("""
nettoyage = pd.read_csv(EXPORT / "20_1_nettoyage_resume.csv").iloc[0]
print(f"Pages re-collectées      : {milliers(nettoyage.titres_recollectes)}")
print(f"Artistes nettoyés        : {int(nettoyage.artistes_nettoyes)}")
print(f"Option 2 — mots retirés  : {100 * nettoyage.part_retiree:.1f} %")
print(f"Option 1 — titres écartés: {milliers(nettoyage.option1_titres_ecartes)}")
""")

md("""
### 9.3 Le verdict change-t-il ?

C'est le test décisif. Les trois corpus traversent **exactement** le même moteur
et les mêmes analyses, chacun avec son cache et son dossier de résultats.

Si le verdict est le même sur les trois, alors les featurings n'ont jamais
faussé l'analyse — et on le **sait**, au lieu de le supposer.
""")
code("""
lignes = []
for nom, dossier in CORPUS.items():
    v = pd.read_csv(dossier / "04_6_verdict_hypotheses.csv")
    mk = pd.read_csv(dossier / "10_2_rang_mikeysem.csv")
    mk = mk[(mk.features == "char") & (mk.metric == "cosine_delta")]
    w = pd.read_csv(dossier / "15_2_web7_synthese.csv").dropna(
        subset=["rang_median_web7"])
    lignes.append({
        "corpus": nom,
        "H₀ favorisée": f"{v.hypothese_favorisee.str.startswith('H0').sum()}/{len(v)}",
        "séparation de Ziak": f"{v.sep_ziak.mean():.2f}",
        "rang de Mikeysem": f"{mk.rang_mikeysem.median():.0f}",
        "meilleur rang de web7": f"{w.rang_median_web7.min():.0f}",
    })
pd.DataFrame(lignes).set_index("corpus")
""")

md("""
**Le verdict ne bouge pas.** Sur les trois corpus, les quatre analyses désignent
H₀ — auteur absent du corpus. Mikeysem reste loin, web7 reste au-delà du
centième rang.

Retirer les couplets d'invités ne change donc pas la réponse, et c'est un
résultat en soi : l'objection était légitime, elle est maintenant **mesurée et
écartée** plutôt qu'admise ou ignorée.

Une limite doit être signalée, parce qu'elle est réelle. Distinguer le membre
de l'invité par la seule fréquence a un coût : Freeman figure sur 6,7 % des
morceaux d'IAM dont il est membre, Rohff sur 6,3 % de ceux de 113 dont il est
seulement invité — **aucun seuil ne les sépare**. Le réglage penche donc vers la
conservation, garder le couplet d'un invité ne faisant que reproduire le défaut
du corpus d'origine, là où retirer celui d'un membre en fabriquerait un
nouveau. Les 12 % de mots retirés sont un **plancher**, pas une valeur exacte.
""")

# =============================================================================
md("""
## 10. Portrait stylométrique de Ziak

À défaut d'identifier Ziak, on peut le caractériser. Deux mesures doivent être
distinguées : la distance médiane à l'ensemble du corpus, qui dit s'il est
atypique ; et la distance à son plus proche voisin, qui dit s'il a un parent.
""")
code("""
exc = pd.read_csv(EXPORT / "06_1_excentricite.csv")
z = exc[exc.artiste == "Ziak"].iloc[0]
part_voisin = (exc.distance_min < z.distance_min).mean()

print(f"Excentricité : rang {int(z.rang_excentricite)} sur {len(exc)} "
      f"— au milieu exact du genre")
print(f"Proche parent : {100 * part_voisin:.0f} % des artistes ont un voisin "
      f"plus proche que le sien")
""")

md("""
C'est exactement la signature attendue d'un auteur qui écrit dans les codes
d'un courant **sans qu'aucun artiste du corpus ne soit sa seconde identité**.

Un mot sur ses marqueurs lexicaux, et sur un piège qu'ils recèlent.
""")
code("""
marqueurs = pd.read_csv(EXPORT / "06_2_marqueurs_lexicaux.csv")
marqueurs.nlargest(6, "z_log_odds")[["mot", "ratio", "z_log_odds"]]
""")

md("""
Ses marqueurs les plus robustes sont des **onomatopées** — les ad-libs qui
ponctuent ses morceaux — suivies d'un argot situé.

Mais le trait le plus spectaculaire de son corpus, à première vue, est un
sous-emploi massif de « je » : cinq fois moins que la moyenne. **C'est un
artefact.**
""")
code("""
elision = pd.read_csv(EXPORT / "06_5_controle_elision.csv")
elision.head(4)
""")

md("""
Il emploie « j' » 1,2 fois plus, et **une fois les deux formes additionnées
l'écart disparaît complètement**. La moitié des paires testées relève du même
phénomène.

Ce que l'on mesurait n'était pas une habitude d'écriture mais une convention de
transcription : les paroles de Genius sont saisies par des contributeurs
bénévoles, et « j'suis » ou « je suis » notent la même diction. C'est le genre
de faux marqueur qu'un contrôle explicite attrape, et qu'une lecture naïve des
scores aurait publié.
""")

# =============================================================================
md("""
## 11. Conclusion

**L'hypothèse du pseudonyme était testable, et le test est concluant.** Aucun
des 392 autres artistes éligibles ne présente la signature stylométrique de
Ziak, alors qu'un protocole validé sur 177 cas connus retrouve le bon auteur
neuf fois sur dix. Les quatre analyses convergent vers l'hypothèse d'un auteur
absent du corpus.

Les deux noms que la rumeur avance ont été testés séparément. **Mikeysem** se
classe derrière six artistes que personne ne soupçonne, même si la minceur de
son corpus interdit d'en faire plus qu'un faisceau convergent. **web7**
n'apparaît jamais parmi les proches de Ziak, mais il est crédité co-auteur de
10 titres, et sur l'album de 2025 les titres qu'il co-signe sont bien plus
proches de son écriture que les autres : la version forte de la rumeur ne tient
pas, une co-écriture ponctuelle est documentée et mesurable.

Ce verdict a été éprouvé sur des liens d'auteur **réels** et non plus simulés,
puis **sur trois corpus** — le publié et deux corpus dont les couplets
d'invités ont été retirés. Il ne bouge pas.

Ce que l'on observe à la place a sa propre valeur descriptive : Ziak écrit au
centre de son genre, sans excentricité mesurable, mais sans proche parent non
plus. Sa parenté avec Kerchak ou Beendo Z est celle d'une génération et d'un
sous-genre partagés, pas d'une main commune.

---

**La contribution la plus transposable de ce travail est cependant
méthodologique.** Une même question, posée au même corpus, reçoit deux réponses
opposées selon le protocole : l'approche intuitive désigne un artiste avec
assurance et se trompe trois fois sur quatre ; l'approche étalonnée ne désigne
personne, et sait dire pourquoi.

Entre les deux, il n'y a pas un algorithme plus sophistiqué, mais trois
exigences ordinaires — **contrôler une variable de confusion, valider sur des
cas connus, calibrer contre une hypothèse nulle**.

---

*Note éthique.* Cette étude porte sur une personne ayant fait le choix de
l'anonymat. Elle s'appuie exclusivement sur des données publiques de recherche
et des textes publiés. La stylométrie n'établit pas d'identité civile : une
correspondance forte aurait constitué un indice de parenté textuelle, jamais
une preuve — et elle aurait appelé, à ce titre, une retenue plus grande encore
dans sa publication que le résultat négatif obtenu ici.

*Reproductibilité.* `METHODE.md` expose la démarche, `CODE.md` commente le code
bloc par bloc, `PIPELINE.md` en donne le schéma. La chaîne complète s'exécute
par `./pipeline.sh`.
""")


def main() -> None:
    nb = nbf.v4.new_notebook(cells=C)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    })
    if "--sans-execution" not in sys.argv:
        from nbclient import NotebookClient
        print(f"exécution des {len(C)} cellules...")
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(Path.cwd())}}).execute()
    nbf.write(nb, str(SORTIE))
    n_code = sum(1 for c in C if c.cell_type == "code")
    print(f"{SORTIE} — {len(C)} cellules ({n_code} de code)")


if __name__ == "__main__":
    main()
