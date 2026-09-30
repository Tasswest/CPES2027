#!/usr/bin/env python3
"""Construit le notebook qui montre la chaîne de traitement, code à l'appui.

Là où `23_notebook.py` produit l'étude et ses résultats, celui-ci produit la
*mécanique* : pour chaque étape, le code réel qui l'exécute, puis une
démonstration sur un exemple concret.

Le code affiché est extrait des modules par `inspect.getsource` plutôt que
recopié. Un extrait recopié se désynchronise du jour où l'on touche au module,
et personne ne s'en aperçoit ; extrait, il est nécessairement celui qui tourne.

Les démonstrations de nettoyage portent sur un texte inventé pour l'occasion,
jamais sur des paroles réelles : le mécanisme se montre aussi bien, sans
reproduire d'œuvre protégée.

Usage : python3 24_notebook_pipeline.py [--sans-execution]
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

SORTIE = Path("00_pipeline_traitement.ipynb")
C = []


def md(t: str) -> None:
    C.append(nbf.v4.new_markdown_cell(t.strip("\n")))


def code(s: str) -> None:
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


def montre(objet: str, commentaire: str = "") -> None:
    """Cellule qui affiche le code source réel d'une fonction."""
    code(f'{commentaire}source({objet})' if commentaire else f"source({objet})")


# =============================================================================
md("""
# La chaîne de traitement, étape par étape

**Le code réel, à chaque étape, suivi de sa démonstration**

Tassilo Westphalen · CPES Sciences des données, Arts et Cultures — PSL / Louis-le-Grand

---

Ce notebook accompagne l'étude sur l'identité de Ziak
([`02_stylometrie_ziak.ipynb`](02_stylometrie_ziak.ipynb)). Il ne refait pas
l'analyse : il montre **comment elle est faite**, fonction par fonction.

Le code affiché n'est pas recopié : il est **extrait des modules à l'exécution**
par `inspect.getsource`. Un extrait recopié se désynchronise du jour où l'on
touche au module, et personne ne s'en aperçoit. Extrait, il est nécessairement
celui qui tourne.

## Plan

1. Mise en place
2. Récolte — d'où viennent les données
3. Prétraitement — du texte aux nombres
4. Balises — qui chante quelle strophe
5. Moteur d'attribution — mesurer une distance
6. Une attribution complète, de bout en bout
7. Les analyses et la restitution

> **Les démonstrations de nettoyage portent sur un texte inventé**, jamais sur
> des paroles réelles : le mécanisme se montre aussi bien, sans reproduire
> d'œuvre protégée.
""")

# =============================================================================
md("""
## 1. Mise en place

Un utilitaire d'abord : `source()` affiche le code d'une fonction, avec la
coloration syntaxique. C'est lui qui garantit que ce notebook montre le code
réel et non sa paraphrase.
""")
code("""
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Code, display

import genius_sections
import stylo_attribution
import stylo_features


def source(fonction) -> Code:
    \"\"\"Le code réel d'une fonction, extrait du module à l'exécution.\"\"\"
    return Code(inspect.getsource(fonction), language="python")


EXPORT = Path("export")
print("modules chargés :", stylo_features.__name__, "·",
      stylo_attribution.__name__, "·", genius_sections.__name__)
""")

# =============================================================================
md("""
## 2. Récolte

### 2.1 D'où viennent les données

Trois sources, et elles n'ont pas les mêmes propriétés.

| Source | Ce qu'elle donne | Volume |
|---|---|---|
| Hugging Face — jeu LRFAF | le corpus publié, **paroles sans balises** | 37 307 titres |
| genius.com, par artiste | Mikeysem, web7, Ziak, avec balises | 3 artistes |
| genius.com, par URL | les balises de tout le corpus | 34 666 pages |

La troisième ligne est celle qui a coûté six heures, et la section 4 dira
pourquoi elle était nécessaire.

### 2.2 Le corpus publié se charge, se filtre, se met en cache

`corpus_csv()` choisit le corpus selon la variante demandée — le publié, ou
l'un des deux nettoyés. C'est ce qui permet de rejouer l'étude trois fois sans
rien écraser.
""")
montre("stylo_features.corpus_csv")

md("""
### 2.3 La collecte par artiste

Pour les artistes absents du corpus. L'API de Genius ne sert pas les paroles —
elles sont licenciées séparément — donc les métadonnées viennent de l'endpoint
JSON et le texte de la page HTML. C'est la méthode qu'emploie `lyricsgenius`,
et celle qu'a employée LRFAF lui-même.

### 2.4 La re-collecte des balises

Le cœur du problème traité en section 4 : le corpus publié a perdu ses balises
de section. Elles sont encore sur Genius, dont le corpus conserve l'URL de
chaque morceau.
""")
code("""
import importlib.util

spec = importlib.util.spec_from_file_location("collecte", "19_collecte_corpus.py")
collecte = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collecte)

display(source(collecte.paroles))
""")

md("""
Trois détails de ce code méritent d'être relevés.

- **Le sélecteur** vise d'abord `data-lyrics-container`, l'attribut stable que
  Genius maintient pour son propre front-end. Les deux replis couvrent les
  balisages plus anciens, dont les noms de classes minifiés changent à chaque
  refonte — c'est sur eux que repose `lyricsgenius`, et ce qui le casse
  périodiquement. Sur 34 666 pages, zéro échec de parsing.
- **Le repli exponentiel** sur 429 / 403 / 5xx : la collecte dure six heures,
  un refus passager ne doit pas la faire échouer.
- **Le 404 n'est pas réessayé** : c'est une page supprimée, pas un incident.
  Il y en a eu 66.
""")

# =============================================================================
md("""
## 3. Prétraitement — du texte aux nombres

Quatre opérations, chacune montrée puis démontrée. La démonstration porte sur
un texte inventé pour l'occasion.
""")
code("""
# Texte d'exemple, écrit pour ce notebook — jamais des paroles réelles.
EXEMPLE = "J'ai vu l'aube, puis j'suis rentré... (uh) 100 % réveillé !"
print(EXEMPLE)
""")

md("""
### 3.1 Nettoyer

Deux caractères sont **délibérément conservés** : les accents, qui portent du
signal en français, et l'apostrophe, qui porte l'élision — donc une partie de
la grammaire.
""")
montre("stylo_features.clean_lyrics")
code("""
propre = stylo_features.clean_lyrics(EXEMPLE)
print("avant :", EXEMPLE)
print("après :", propre)
""")

md("""
### 3.2 Découper en mots

L'apostrophe finale est capturée **avec** le mot : `j'ai` donne `j'` puis `ai`.
Autrement dit `je` et `j'` deviennent deux traits distincts.

Ce choix, qui paraît anodin, est ce qui a permis de démasquer un faux marqueur :
Ziak semblait sous-employer « je » d'un facteur 5, alors qu'il compense
entièrement en « j' ».
""")
montre("stylo_features.tokenize")
code("""
mots = stylo_features.tokenize(propre)
print(mots)
print(f"\\n{len(mots)} tokens — noter « j' » isolé de « ai »")
""")

md("""
### 3.3 Les 4-grammes de caractères

Fenêtre glissante de quatre caractères, **espaces compris**. Les espaces
comptent : ils capturent les fins et débuts de mots, donc une part de la
syntaxe, sans passer par le lexique. C'est ce qui rend ces traits robustes au
sujet abordé.
""")
montre("stylo_features.char_ngrams")
code("""
grammes = stylo_features.char_ngrams(propre)
print(grammes[:12])
print(f"\\n{len(grammes)} 4-grammes pour {len(propre)} caractères")
""")

md("""
### 3.4 Filtrer le corpus

Deux filtres, et le second compte plus qu'il n'y paraît : un même morceau publié
sous deux artistes attribuerait le même texte à deux auteurs, ce qui fausse
mécaniquement les distances.
""")
montre("stylo_features.load_corpus")

md("""
### 3.5 Mettre en cache

Le choix structurant de tout le projet tient dans une ligne de cette fonction :
les comptes sont stockés **par chanson**, jamais par artiste.

Un document d'artiste de n'importe quelle taille se recompose ensuite par simple
addition de lignes. Sans cela, chaque rééchantillonnage exigerait de
revectoriser le corpus — et le protocole, qui en compte des centaines, serait
inexécutable.
""")
code("""
cache = stylo_features.build_cache()
meta = cache["meta"]

print("le cache contient :")
for cle, valeur in cache.items():
    forme = getattr(valeur, "shape", None) or f"{len(valeur)} entrées"
    print(f"  {cle:18} {forme}")
""")

# =============================================================================
md("""
## 4. Balises — qui chante quelle strophe

### 4.1 Le problème

Sur Genius, chaque bloc de paroles porte le nom de celui qui le chante. **LRFAF
a supprimé ces balises avant publication** : sur 400 textes tirés au hasard,
aucun n'en porte.

Or un couplet d'invité est écrit par l'invité. Le laisser dans le corpus de
l'artiste principal rapproche mécaniquement cet artiste de tous ceux qu'il a
invités — et l'hypothèse testée porte justement sur une proximité.

### 4.2 Lire une balise

`interpretes()` renvoie les interprètes nommés par une balise, ou `None` si elle
n'en nomme aucun.
""")
montre("genius_sections.interpretes")

md("""
L'ordre des branches est le correctif d'un bug réel : Genius emploie le
deux-points **et** le tiret, et seul le premier était lu, si bien que
`[Couplet 3 - Kool Shen]` passait pour une section de l'artiste principal.

Mais le tiret ne peut être accepté que si un libellé de section le précède —
sinon `Pre-refrain` et `Jay-Z` seraient coupés en deux. D'où la table de cas
ci-dessous, qui sert de test.
""")
code("""
cas = ["Couplet 1", "Refrain", "Couplet 2 : Akhenaton",
       "Couplet 3 - Kool Shen & Joey Starr", "Pre-refrain",
       "Post-refrain : Ziak", "Jay-Z", "Sofiane", "?"]
pd.DataFrame({
    "balise": [f"[{c}]" for c in cas],
    "interprètes nommés": [genius_sections.interpretes(c) or "— (artiste principal)"
                           for c in cas],
})
""")

md("""
### 4.3 Retirer les strophes d'invités

La règle est volontairement stricte : une section n'est conservée que si elle
n'est attribuée à personne, ou au seul artiste principal. **Une section partagée
est retirée**, faute de pouvoir savoir qui en a écrit quoi.
""")
montre("genius_sections.retire_featurings")
code("""
# Morceau fictif, écrit pour ce notebook.
MORCEAU = \"\"\"[Couplet 1]
première ligne de l'artiste principal
deuxième ligne de l'artiste principal

[Couplet 2 : Invité]
ligne écrite par quelqu'un d'autre
seconde ligne du même invité

[Refrain]
refrain non attribué, donc gardé\"\"\"

garde, stats = genius_sections.retire_featurings(MORCEAU, {"Principal"})
print(garde)
print("\\n→", stats)
""")

md("""
### 4.4 Membre de groupe ou invité de passage ?

C'est la difficulté du nettoyage. La règle naïve viderait les groupes de leur
contenu : chez IAM, les blocs nomment Akhenaton et Shurik'n, dont les couplets
*sont* le texte d'IAM.

C'est la **fréquence** qui les sépare — un nom qui revient dans une grande
partie des morceaux d'un artiste est celui d'un membre, et ses couplets sont
gardés ; un nom qui n'apparaît qu'une ou deux fois est celui d'un invité, et
ses couplets sont retirés.
""")
code("""
spec = importlib.util.spec_from_file_location("nettoyage", "20_corpus_sans_invites.py")
nettoyage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nettoyage)

display(source(nettoyage.membres_par_artiste))
""")

md("""
Le résultat de cette règle, artiste par artiste, est publié plutôt que caché
dans le code — voici ce qu'elle a retenu pour quelques groupes.
""")
code("""
membres = pd.read_csv(EXPORT / "20_2_membres_detectes.csv")
apercu = membres[membres.artiste.isin(["IAM", "Suprême NTM", "113"])]
apercu[["artiste", "intervenant", "titres", "sur", "part", "motif"]]
""")

md("""
**Cette règle a une limite, et elle est connue.** Freeman figure sur 6,7 % des
morceaux d'IAM dont il est membre ; Rohff sur 6,3 % de ceux de 113 dont il est
seulement invité. Aucun seuil ne les sépare.

Les deux erreurs possibles n'étant pas symétriques, le réglage penche vers la
conservation : garder le couplet d'un invité ne fait que reproduire le défaut du
corpus d'origine, tandis que retirer celui d'un membre en fabriquerait un
nouveau. Les 12 % de mots retirés sont donc un **plancher**, pas une valeur
exacte.
""")

# =============================================================================
md("""
## 5. Moteur d'attribution — mesurer une distance

Quatre fonctions, dans l'ordre où elles s'enchaînent.

### 5.1 Échantillonner à taille contrôlée

On échantillonne des **chansons entières**, pas des mots : découper au milieu
d'un morceau mélangerait des contextes. Sans cette contrainte de taille, la
distance mesure surtout la couverture de vocabulaire, et un artiste prolifique
paraît proche de tout le monde.
""")
montre("stylo_attribution.sample_indices")

md("""
### 5.2 Agréger en fréquences
""")
montre("stylo_attribution.make_docs")

md("""
### 5.3 Standardiser, puis mesurer

Le détail qui compte tient en deux lignes : **les z-scores sont estimés sur les
candidats seuls**, et la requête est projetée dans ce référentiel. C'est la
situation réelle d'un texte anonyme confronté à un corpus connu — inclure la
requête dans le calcul de la moyenne la ferait participer à sa propre
normalisation.
""")
montre("stylo_attribution.rank_candidates")

md("""
### 5.4 Le score de séparation

La fonction la plus importante du projet, et la plus courte.

Un classement désigne **toujours** un premier : ce n'est pas une information. La
question utile est de savoir de combien il se détache. Ce z-score négatif est la
seule grandeur comparable d'une requête à l'autre — donc la seule qui permette
de confronter Ziak à des cas dont on connaît la réponse.
""")
montre("stylo_attribution.separation_score")

# =============================================================================
md("""
## 6. Une attribution complète, de bout en bout

Toutes les pièces assemblées. Une quinzaine de lignes suffisent à poser la
question de l'étude : *qui, parmi les 392 candidats, écrit comme Ziak ?*
""")
code("""
mat = cache["counts_char"]
n_tokens = meta["n_tokens"].to_numpy()
totaux = meta.groupby("artist")["n_tokens"].sum()

pool = sorted(a for a in totaux[totaux >= 12_000].index if a != "Ziak")
titres = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
idx_ziak = meta.index[meta["artist"] == "Ziak"].to_numpy()
rng = np.random.default_rng(20260930)

# 1. chaque candidat ramené à 12 000 mots exactement
groupes = {a: stylo_attribution.sample_indices(
    titres[a], n_tokens[titres[a]], 12_000, rng) for a in pool}

# 2. agrégation en fréquences relatives
noms, freqs = stylo_attribution.make_docs(mat, groupes)

# 3. la requête : tout le corpus de Ziak
requete = np.asarray(mat[idx_ziak].sum(axis=0)).ravel()
requete = requete / requete.sum()

# 4. distances standardisées, puis séparation du premier
distances = stylo_attribution.rank_candidates(requete, freqs, "cosine_delta")
ordre = np.argsort(distances)
separation = stylo_attribution.separation_score(distances, int(ordre[0]))

classement = pd.DataFrame({"artiste": [noms[i] for i in ordre[:8]],
                           "distance": distances[ordre[:8]]})
print(f"{len(pool)} candidats · séparation du premier : {separation:.2f}")
classement
""")

md("""
Le premier du classement se détache de **moins de 3 écarts-types**. C'est peu :
un alias authentique produit une séparation autour de −4,6, et un auteur absent
du corpus autour de −3,0.

C'est tout le propos de la section 5.4 — sans ce repère, le tableau ci-dessus
désignerait un coupable avec assurance.
""")
code("""
seuils = pd.read_csv(EXPORT / "03_3_seuils_decision.csv")
seuils = seuils[(seuils.features == "char") & (seuils.metric == "cosine_delta")]
print(f"séparation observée pour Ziak    : {separation:.2f}")
print(f"repère H₁ — auteur bien présent  : {seuils.sep_H1_correct_moy.iloc[0]:.2f}")
print(f"repère H₀ — auteur absent        : {seuils.sep_H0_moy.iloc[0]:.2f}")
""")

# =============================================================================
md("""
## 7. Les analyses et la restitution

Le moteur ci-dessus est appelé par treize scripts, chacun répondant à une
question et écrivant sa réponse en CSV.
""")
code("""
analyses = pd.DataFrame([
    ("02", "l'approche intuitive a-t-elle raison ?", "25 % contre 91,5 %"),
    ("03", "que vaut la méthode quand on connaît la réponse ?", "90 % au premier rang"),
    ("04", "un artiste du corpus porte-t-il la signature de Ziak ?", "non — H₀ 4 fois sur 4"),
    ("05", "le verdict résiste-t-il aux objections ?", "oui, sur les quatre testées"),
    ("06", "à quoi ressemble son écriture ?", "banal au centre, sans proche parent"),
    ("08-09", "Mikeysem au format LRFAF", "mikeysem_lrfaf.csv"),
    ("10", "Ziak est-il Mikeysem ?", "non — 58ᵉ sur 493"),
    ("13-14", "la validation tient-elle sur des cas réels ?", "Joke → Ateyaba 1ᵉʳ sur 393"),
    ("15-16", "web7 écrit-il ses textes ?", "non — mais co-auteur de 10 titres"),
    ("18", "les featurings faussent-ils le classement ?", "non — mesuré, pas supposé"),
    ("19-20", "récupérer et retirer les strophes d'invités", "deux corpus propres"),
], columns=["script", "question", "réponse"])
analyses.set_index("script")
""")

md("""
L'ordre n'est pas décoratif : **la validation précède les tests**. Un résultat
négatif n'a de valeur que si l'on a d'abord établi que la méthode sait trouver
quand il y a quelque chose à trouver.

### La restitution

`07_figures.py` fabrique les figures depuis les CSV. `article_contenu.py` porte
le texte et **relit tous les chiffres dans les résultats au moment de la
génération** : aucune valeur n'est écrite à la main, donc aucune ne peut se
désynchroniser de l'analyse qu'elle cite. Deux moteurs partagent ce texte, l'un
vers le PDF, l'autre vers le Word.

### Tout enchaîner

```bash
./pipeline.sh                    # les huit étapes
./pipeline.sh --depuis variantes # à partir de celle-ci
./pipeline.sh balises --oui      # autorise l'étape longue (~6 h)
```

Chaque étape est idempotente : ce qui existe n'est pas refait, et une relance
reprend où elle s'est arrêtée.

---

**Pour aller plus loin** — [`02_stylometrie_ziak.ipynb`](02_stylometrie_ziak.ipynb)
présente l'étude et ses résultats ; [`METHODE.md`](METHODE.md) expose la
démarche ; [`CODE.md`](CODE.md) commente le code bloc par bloc ;
[`PIPELINE.md`](PIPELINE.md) en donne le schéma.
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
    print(f"{SORTIE} — {len(C)} cellules "
          f"({sum(1 for c in C if c.cell_type == 'code')} de code)")


if __name__ == "__main__":
    main()
