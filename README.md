# ProjetRap — Qui se cache derrière Ziak ?

## Contexte

Ce projet s'inscrit dans le cadre du cours d'application des méthodes de **data science et d'intelligence artificielle à l'analyse de corpus volumineux**, du parcours **CPES Sciences des données, Arts et Cultures (DAC)**, porté par l'[Université PSL](https://psl.eu/) et le [Lycée Louis-le-Grand](https://www.louislegrand.fr/) :
- Page du parcours : [psl.eu/formation/cpes-psl-louis-le-grand](https://psl.eu/formation/cpes-psl-louis-le-grand)

Ce cours mobilise également les ressources de l'[Institut ACSS-PSL](https://acss-dig.psl.eu/) (Applied Computational Social Sciences).

Le travail demandé aux étudiants est de **mettre en valeur des faits remarquables** repérés
dans les données, en les commentant et en les interprétant. Ce dépôt le fait en posant une
question unique et vérifiable : *Ziak, rappeur masqué apparu en 2020, est-il un autre artiste ?*

## Source des données

Le corpus correspond à **LRFAF** (37 307 chansons de rap français issues de genius.com, croisées avec Wikipédia/Wikidata), constitué par Benoît de Courson (regicid) :
- Jeu de données : [huggingface.co/datasets/regicid/LRFAF](https://huggingface.co/datasets/regicid/LRFAF)
- Article associé : Benoît de Courson, *« LRFAF : une exploration numérique du rap français depuis les années 1990 »* — [researchgate.net/publication/379061284](https://www.researchgate.net/publication/379061284_LRFAF_une_exploration_numerique_du_rap_francais_depuis_les_annees_1990)
- Exploration interactive : [Gallicagram, corpus « Rap »](https://shiny.ens-paris-saclay.fr/app/gallicagram)

Ce corpus est distribué pour un usage de recherche, sans licence formelle (les ayants droit restant les artistes).

## Installation

Le corpus (~114 Mo) dépasse la limite de 100 Mo par fichier de GitHub : il n'est donc
**pas versionné** (voir `.gitignore`). Le télécharger à la racine du dépôt sous le nom
`corpus.csv` :

```bash
curl -L "https://huggingface.co/datasets/regicid/LRFAF/resolve/main/corpus.csv?download=true" -o corpus.csv
```

Les scripts acceptent aussi l'ancien nom `RapFr.csv`, pour les copies déjà téléchargées.

## Organisation du dépôt

Conventions du dépôt amont (voir les skills [`export-organization`](.github/skills/export-organization/SKILL.md)
et [`notebook-authoring`](.github/skills/notebook-authoring/SKILL.md)) :

- les scripts et notebooks (`.py`, `.ipynb`) restent à la racine ;
- `images/` : figures ;
- `export/` : tables et CSV de résultats.

**Deux écarts assumés**, signalés ici plutôt que passés sous silence :

1. `images/` reste versionné. Le `.gitignore` amont n'y conserve que les figures `R00_*`
   et `*_full_*` ; les neuf figures de l'article sont des livrables de corpus complet et
   doivent accompagner le PDF.
2. Le skill [`rap-eda-hip-hop-aesthetic`](.github/skills/rap-eda-hip-hop-aesthetic/SKILL.md)
   (fond noir, texte doré) n'est pas appliqué aux figures de l'article : il vise les
   visualisations exploratoires, alors que ces figures sont destinées à l'impression sur
   page blanche, où un fond noir nuit à la lisibilité.

## L'article

- 📝 **[`article_ziak_stylometrie.docx`](article_ziak_stylometrie.docx)** — l'article rédigé, au format Word pour être retravaillé (16 pages, 9 figures, 10 tableaux).
- 📄 [`article_ziak_stylometrie.pdf`](article_ziak_stylometrie.pdf) — la même version en PDF.

Il se lit **à deux niveaux** : la section 1 répond à la question en français courant, sans
prérequis ; les sections suivantes exposent la méthode et les chiffres, chaque passage
technique étant suivi d'un encadré « En clair », un lexique fermant l'article.

Le texte vit dans [`article_contenu.py`](article_contenu.py), partagé par
[`12_article_pdf.py`](12_article_pdf.py) et [`17_article_docx.py`](17_article_docx.py) :
les deux formats ne peuvent pas diverger, et tous les chiffres sont relus dans `export/`.
Une fois le Word modifié à la main, c'est lui qui fait foi — relancer le script
écraserait ces modifications.

## L'étude

Ziak est apparu en 2020 sans identité civile publique, ce qui a nourri trois hypothèses.
Toutes trois sont testables : si un auteur en cache un autre, les textes doivent porter la
même signature statistique.

| Hypothèse | Résultat | Solidité |
|---|---|---|
| Ziak est un rappeur du corpus, sous un autre nom | aucun des 392 candidats ne porte sa signature | forte |
| Ziak est Mikeysem | 58ᵉ sur 493, derrière six artistes non suspectés | limitée (3 745 mots) |
| web7 (ex-7 Jaws) écrit ses textes | jamais parmi ses proches, mais co-auteur crédité de 10 titres, avec une trace mesurable sur l'album 2025 (p = 0,04) | moyenne |

**Validation** — la méthode retrouve le bon auteur dans 90 % des cas sur 177 artistes dont
la réponse est connue, et 97,7 % pour la génération de Ziak. Confrontée à des liens réels
plutôt que simulés — 14 recouvrements auteur/groupe, 3 changements de nom documentés —
elle retrouve *Joke → Ateyaba* au premier rang sur 393 malgré un changement d'identité
revendiqué, mais perd sa capacité de détection quand l'auteur ne signe qu'une fraction des
textes.

**Le cas Mikeysem** — le nom le plus souvent avancé ne figurait pas dans LRFAF : sans page
Wikipédia, il échappait au critère d'inclusion du corpus. Ses titres ont été collectés et
passés dans le pipeline LRFAF reconstitué (voir [`RAPPORT_LRFAF.md`](RAPPORT_LRFAF.md)),
puis testés. Son corpus reste mince : le résultat vaut comme faisceau convergent, pas
comme démonstration.

**Point méthodologique** — l'approche intuitive (concaténer les chansons de chaque artiste,
puis comparer) donne un classement d'apparence convaincante mais se trompe trois fois sur
quatre : elle mesure surtout la quantité de texte disponible sur chaque artiste. Le
protocole retenu contrôle la taille des documents, se valide sur des cas de vérité-terrain
et se calibre contre une hypothèse nulle explicite.

## Pipeline

Les scripts s'exécutent dans l'ordre ; le premier construit un cache de compteurs
(`.cache_stylo/`, non versionné) qui rend les suivants quasi instantanés.

```bash
python3 stylo_features.py              # cache des compteurs par chanson (~1 min)
python3 02_diagnostic_biais_taille.py  # pourquoi l'approche naïve échoue
python3 03_validation_protocole.py     # validation sur vérité-terrain
python3 04_attribution_ziak.py         # application à Ziak, verdict
python3 05_robustesse.py               # sensibilité, imposteurs, générations
python3 06_profil_stylistique.py       # portrait : marqueurs, excentricité
python3 collecte_genius.py 3152412 .cache_lex/mikeysem_raw.json   # Mikeysem
python3 08_ajout_mikeysem.py           # ajoute Mikeysem au format LRFAF
python3 09_controle_reproduction.py    # contrôles du pipeline reconstitué
python3 10_test_ziak_mikeysem.py       # test de l'hypothèse Mikeysem
python3 13_validation_alias_reels.py   # validation sur recouvrements réels
python3 14_test_alias_temporel.py      # coût d'un changement d'identité
python3 collecte_genius.py 1078135 .cache_lex/web7_raw.json       # web7 (ex-7 Jaws)
python3 collecte_genius.py 2113831 .cache_lex/ziak_raw.json       # Ziak, avec balises
python3 15_test_ziak_7jaws.py          # web7 au niveau de l'artiste, crédits
python3 16_test_2025_web7.py           # expérience naturelle 2025, sans featurings
python3 07_figures.py                  # figures de l'article
python3 12_article_pdf.py              # article_ziak_stylometrie.pdf
python3 17_article_docx.py             # article_ziak_stylometrie.docx
```

Le pipeline LRFAF reconstitué ([`lrfaf_pipeline.py`](lrfaf_pipeline.py)) et son rapport
de reproductibilité ([`RAPPORT_LRFAF.md`](RAPPORT_LRFAF.md)) documentent, colonne par
colonne, ce qui est reproduit exactement, approximé ou impossible à retrouver.

Modules partagés : [`stylo_features.py`](stylo_features.py) (nettoyage, tokenisation,
cache), [`stylo_attribution.py`](stylo_attribution.py) (Cosine Delta, Delta de Burrows,
échantillonnage à taille contrôlée, scores de séparation) et
[`genius_sections.py`](genius_sections.py) (retrait des couplets d'invités à partir des
balises de section de Genius).

## Notebooks

- [`01_description_variables_quantitatives.ipynb`](01_description_variables_quantitatives.ipynb) : description des variables quantitatives du corpus (notebook d'origine du dépôt amont).
- [`02_stylometrie_ziak.ipynb`](02_stylometrie_ziak.ipynb) : la version reproductible de l'étude, avec le code et ses sorties. Il couvre toute l'étude sauf les tests sur web7 (section 9 de l'article), qui ne vivent que dans les scripts `15_` et `16_`.
