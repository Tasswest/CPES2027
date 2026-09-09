# ProjetRap — Analyse du corpus RapFr

## Contexte

Ce projet s'inscrit dans le cadre du cours d'application des méthodes de **data science et d'intelligence artificielle à l'analyse de corpus volumineux**, du parcours **CPES Sciences des données, Arts et Cultures (DAC)**, porté par l'[Université PSL](https://psl.eu/) et le [Lycée Louis-le-Grand](https://www.louislegrand.fr/) :
- Page du parcours : [psl.eu/formation/cpes-psl-louis-le-grand](https://psl.eu/formation/cpes-psl-louis-le-grand)

Ce cours mobilise également les ressources de l'[Institut ACSS-PSL](https://acss-dig.psl.eu/) (Applied Computational Social Sciences), dont l'auteur de ce projet est membre.

Le corpus étudié (`RapFr.csv`) rassemble des textes de chansons de rap français ainsi qu'un ensemble de variables quantitatives et qualitatives associées (popularité, longueur des textes, tonalité, informations sur les artistes, etc.).

## Source des données

`RapFr.csv` correspond au corpus **LRFAF** (37 307 chansons de rap français issues de genius.com, croisées avec Wikipédia/Wikidata), constitué par Benoît de Courson (regicid) :
- Jeu de données : [huggingface.co/datasets/regicid/LRFAF](https://huggingface.co/datasets/regicid/LRFAF)
- Article associé : Benoît de Courson, *« LRFAF : une exploration numérique du rap français depuis les années 1990 »* — [researchgate.net/publication/379061284](https://www.researchgate.net/publication/379061284_LRFAF_une_exploration_numerique_du_rap_francais_depuis_les_annees_1990)
- Exploration interactive des fréquences lexicales du corpus : [Gallicagram, corpus « Rap »](https://shiny.ens-paris-saclay.fr/app/gallicagram)

Ce corpus est distribué pour un usage de recherche, sans licence formelle (les ayants droit restant les artistes).

## Installation

`RapFr.csv` (~114 Mo) dépasse la limite de 100 Mo par fichier de GitHub : il n'est donc **pas versionné** dans ce dépôt (voir `.gitignore`). Pour reconstituer le fichier avant d'exécuter les notebooks, télécharger le corpus depuis Hugging Face et le placer à la racine du dépôt sous le nom `RapFr.csv` :

```powershell
Invoke-WebRequest -Uri "https://huggingface.co/datasets/regicid/LRFAF/resolve/main/corpus.csv?download=true" -OutFile "RapFr.csv"
```

## État d'avancement

Les notebooks livrés dans ce dépôt sont des **premiers jets**. Ils posent une structure d'analyse de base (chargement, description, distributions, évolution temporelle) mais ne sont pas définitifs.

**Le travail demandé aux étudiants est de :**
- affiner les scripts fournis (choix des variables clés, filtres, granularité temporelle, présentation des graphiques...) ;
- surtout, **mettre en valeur des faits remarquables** repérés dans les données, en les commentant et en les interprétant.

## Organisation du dépôt

- Les scripts et notebooks (`.py`, `.ipynb`) restent à la racine du dépôt.
- `images/` : figures générées par les notebooks (graphiques exportés en `.png`).
- `result/` : tables et CSV de résultats produits par les notebooks (résumés, agrégats).

Voir les skills [`notebook-authoring`](.github/skills/notebook-authoring/SKILL.md) et [`export-organization`](.github/skills/export-organization/SKILL.md) pour les conventions détaillées de rédaction des notebooks et d'organisation des exports.

## Notebooks

- [`01_description_variables_quantitatives.ipynb`](01_description_variables_quantitatives.ipynb) : description des variables quantitatives du corpus (statistiques descriptives, distributions, évolution des moyennes annuelles).
- [`02_stylometrie_ziak.ipynb`](02_stylometrie_ziak.ipynb) : **article** — Ziak est-il un autre rappeur ? Test stylométrique de l'hypothèse du pseudonyme.

## Étude : l'identité stylométrique de Ziak

Ziak est apparu en 2020 sans identité civile publique, ce qui a nourri l'hypothèse
d'un artiste déjà établi rappant sous un pseudonyme. Cette hypothèse est testable :
si Ziak est le second nom d'un rappeur du corpus, ses textes doivent porter la même
signature statistique.

**Résultat** — aucun des 392 autres artistes éligibles ne correspond, avec une méthode qui
retrouve le bon auteur dans 90 % des cas quand la réponse est connue (97,7 % pour la
génération de Ziak). Les quatre combinaisons de traits et de distances testées
convergent vers l'hypothèse « auteur absent du corpus ».

**Le cas Mikeysem** — le nom le plus souvent avancé par les auditeurs ne figurait pas
dans LRFAF : sans page Wikipédia, il échappait au critère d'inclusion du corpus. Ses
titres ont été collectés et passés dans le pipeline LRFAF reconstitué
(voir [`RAPPORT_LRFAF.md`](RAPPORT_LRFAF.md)), puis testés. Il se classe 58ᵉ sur 493,
derrière six artistes que personne ne soupçonne, et obtient au test par paire un score
à peine supérieur au hasard. Son corpus (3 745 mots) reste mince : le résultat vaut
comme faisceau convergent, pas comme démonstration.

**Point méthodologique** — l'approche intuitive (concaténer les chansons de chaque
artiste, puis comparer) donne un classement d'apparence convaincante mais se trompe
trois fois sur quatre : elle mesure surtout la quantité de texte disponible sur chaque
artiste. Le protocole retenu contrôle la taille des documents, se valide sur des cas
de vérité-terrain et se calibre contre une hypothèse nulle explicite.

### Pipeline

Les scripts s'exécutent dans l'ordre ; le premier construit un cache de compteurs
(`.cache_stylo/`, non versionné) qui rend les suivants quasi instantanés.

```bash
python3 stylo_features.py             # cache des compteurs par chanson (~1 min)
python3 02_diagnostic_biais_taille.py  # pourquoi l'approche naïve échoue
python3 03_validation_protocole.py     # validation sur vérité-terrain
python3 04_attribution_ziak.py         # application à Ziak, verdict
python3 05_robustesse.py               # sensibilité, imposteurs, générations
python3 06_profil_stylistique.py       # portrait : marqueurs, excentricité
python3 08_ajout_mikeysem.py           # ajoute Mikeysem au format LRFAF
python3 09_controle_reproduction.py    # contrôles du pipeline reconstitué
python3 10_test_ziak_mikeysem.py       # test de l'hypothèse Mikeysem
python3 07_figures.py                  # figures de l'article
```

Le pipeline LRFAF reconstitué ([`lrfaf_pipeline.py`](lrfaf_pipeline.py)) et son rapport
de reproductibilité ([`RAPPORT_LRFAF.md`](RAPPORT_LRFAF.md)) documentent, colonne par
colonne, ce qui est reproduit exactement, approximé ou impossible à retrouver.

Modules partagés : [`stylo_features.py`](stylo_features.py) (nettoyage, tokenisation,
cache) et [`stylo_attribution.py`](stylo_attribution.py) (Cosine Delta, Delta de
Burrows, échantillonnage à taille contrôlée, scores de séparation).
