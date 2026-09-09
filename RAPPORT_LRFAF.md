# Reproduire le pipeline LRFAF pour ajouter un artiste

Rapport de rétro-ingénierie du corpus [regicid/LRFAF](https://huggingface.co/datasets/regicid/LRFAF)
et d'application à l'artiste **Mikeysem**.

---

## 0. Résultat en bref

Le code de production de LRFAF **n'a jamais été publié**. Le pipeline a donc été
reconstitué à partir de la prose de l'article, du code source de `lyricsgenius`
et de la rétro-ingénierie des relations arithmétiques du corpus. Sur 33 colonnes :

| Statut | Nombre | Colonnes |
|---|---|---|
| **Reproduites exactement** | 14 | `artist`, `title`, `year`, `lyrics`, `pageviews`, `contributors`, `url`, `n_words`, `n_unique_words`, `means_word_length`, `n_je`, `pageviews_corrected`, `pageviews_2`, `pageview_mean` |
| **Reproduites par relation interne** | 2 | `n_non_french_words`, `age_artist` |
| **Approximées** (méthode connue, ressource dérivée) | 6 | `n_french_words`, `n_verlan`, `n_argot`, `n_sexe`, `n_profanity`, `n_lines` |
| **Identifiées mais non exécutées ici** | 2 | `hate`, `sexism` |
| **Non reproductibles** | 9 | `topic`, `topic_clean`, `ranking`, `n_onomatopee`, `n_positive`, `n_negative`, `sentiment2`, `birthdate_artist`, `born_in_france` |

Deux constats conditionnent la lecture de tout ce qui suit, et il faut les
énoncer avant le détail :

1. **Les paroles de Genius ont massivement changé depuis la collecte de LRFAF
   (début 2024).** Sur 22 titres du corpus re-scrapés en septembre 2026, **un
   seul** a des paroles strictement identiques à celles publiées. Aucune
   reproduction *à la valeur près* des lignes existantes n'est donc possible
   aujourd'hui, quelle que soit la qualité du pipeline.

2. **Le corpus n'est pas cohérent avec lui-même.** Les compteurs de LRFAF n'ont
   pas été calculés sur les paroles qu'il publie : `n_words` ne correspond au
   texte de la colonne `lyrics` que dans **10,2 %** des lignes, avec un écart
   médian de −4 mots. Ces 4 mots sont l'en-tête du site Genius
   (« *8 ContributorsMes étoiles Lyrics* »), présent lors du comptage et retiré
   avant publication. Le pipeline reconstitué reproduit ce comportement.

---

## A. Tableau des colonnes

`Repro.` : **Oui** = formule/source retrouvée et vérifiée · **Approx.** = méthode
connue, ressource ou variante non identique · **Non** = méthode indisponible.

### Métadonnées de collecte

| Colonne | Signification | Source | Méthode | Repro. | Référence |
|---|---|---|---|---|---|
| `artist` | Nom de l'artiste | Genius API | `primary_artist.name`. LRFAF ne retient que les titres dont l'artiste est le *primary* (vérifié : 99,1 % des URL du corpus commencent par le slug de l'artiste), soit le défaut `include_features=False` de `lyricsgenius` | **Oui** | [rap.Rmd §collecte](https://huggingface.co/datasets/regicid/LRFAF/blob/main/article/rap.Rmd) |
| `title` | Titre | Genius API | `song.title` | **Oui** | idem |
| `year` | Année de sortie | Genius API | `release_date_components.year`. Vide si Genius ne date pas le titre (9,7 % du corpus) | **Oui** | idem |
| `lyrics` | Paroles | Scraping genius.com | Voir §B.1 | **Oui** (97,9 %) | code de `lyricsgenius` |
| `pageviews` | Vues de la page | Genius API | `stats.pageviews`. Genius ne publie le compteur qu'**au-delà de 5 000 vues** : LRFAF code l'absence par `0` (71,7 % du corpus) | **Oui** | [README HF](https://huggingface.co/datasets/regicid/LRFAF) |
| `contributors` | Contributeurs | Genius API | `stats.contributors` | **Oui** | — |
| `url` | URL Genius | Genius API | `song.url` | **Oui** | — |

> **Valeurs datées.** `pageviews` et `contributors` sont des compteurs vivants.
> Les valeurs collectées aujourd'hui ne sont pas comparables à celles de 2024 :
> elles sont exactes en méthode, décalées en niveau.

### Topic modeling

| Colonne | Signification | Source | Méthode | Repro. | Référence |
|---|---|---|---|---|---|
| `topic` | Cluster thématique brut | Calculé | Plongements **Solon-large** (OrdalieTech) puis clustering **Bunka** (Charles de Dampierre) | **Non** | rap.Rmd §topic modeling |
| `topic_clean` | Nom lisible du topic | Manuel | Noms attribués *a posteriori* par l'auteur aux clusters (« Rap conscient », « Gangsta rap »…) | **Non** | idem |
| `ranking` | Centralité dans le topic | Bunka | Rang du titre au sein de son topic ; `NA` = forte incertitude (17,8 % du corpus) | **Non** | README HF |

> **Pourquoi non reproductible.** Les topics sont les clusters d'un modèle ajusté
> sur *l'ensemble* du corpus. Les reproduire supposerait de réexécuter le
> clustering complet, sans que ni les hyperparamètres, ni la graine, ni le
> nommage manuel des clusters ne soient documentés. Classer 7 titres nouveaux
> par proximité aux centroïdes existants donnerait une étiquette *plausible*,
> pas la valeur qu'aurait produite le pipeline : ces colonnes sont laissées vides.

### Compteurs lexicométriques

| Colonne | Signification | Source | Méthode | Repro. | Référence |
|---|---|---|---|---|---|
| `n_words` | Nombre de mots | Calculé | Apostrophes → espace, minuscules, tokens `\w+` (équivalent de `tokenizers::tokenize_words` en R) | **Oui** | `get_complexity`, rap.Rmd L60 |
| `n_unique_words` | Mots distincts | Calculé | `length(unique(words))` sur la même tokenisation | **Oui** | idem |
| `means_word_length` | Longueur moyenne des mots | Calculé | `sum(nchar(words))/length(words)`. Le terme `sum(str_count(words,"'"))` du code R est toujours nul, les apostrophes ayant déjà été remplacées | **Oui** | idem |
| `n_je` | Occurrences de « je » | Calculé | `count("je") + count("j")` — l'élision « j' » produit le token « j » | **Oui** (82,9 %) | rap.Rmd §lexiques |
| `n_lines` | Nombre de lignes | Calculé | Non documenté. Meilleure variante trouvée : lignes contenant **au moins deux mots** (84,5 % contre 68 % pour « lignes non vides ») | **Approx.** | — |
| `n_french_words` | Mots du dictionnaire | Morphalou 3.1 (ATILF) | Tokens présents dans les formes fléchies. **Corrélation 0,988** avec la colonne publiée : ressource confirmée, version 2024 légèrement différente | **Approx.** | rap.Rmd §lexiques |
| `n_non_french_words` | Hors dictionnaire | Calculé | `n_words − n_french_words` — **exact sur 100 % du corpus** | **Oui** (dérivée) | vérifié |
| `n_verlan` | Mots en verlan | Wiktionnaire | [Catégorie:Verlan](https://fr.wiktionary.org/wiki/Catégorie:Verlan) (312 entrées aujourd'hui) | **Approx.** | rap.Rmd, note |
| `n_argot` | Mots d'argot | Wiktionnaire | [Annexe:Liste de termes argotiques](https://fr.wiktionary.org/wiki/Annexe:Liste_de_termes_argotiques_en_français) (1 260 entrées) | **Approx.** | idem |
| `n_sexe` | Lexique sexuel | Wiktionnaire | [Catégorie:Lexique en français de la sexualité](https://fr.wiktionary.org/wiki/Catégorie:Lexique_en_français_de_la_sexualité) (1 101 entrées) | **Approx.** | idem |
| `n_profanity` | Insultes | GitHub | [MauriceButler/badwords](https://github.com/MauriceButler/badwords) + « fuck », « nigga », « bitch » | **Approx.** | idem |
| `n_onomatopee` | Onomatopées | ? | **Aucune source documentée** : ni l'article, ni le README, ni le dépôt ne mentionnent ce lexique | **Non** | — |
| `n_positive` / `n_negative` | Tonalité | LIWC-fr | Dictionnaires d'émotions du projet LIWC, version française de Piolat et al. (2011) | **Non** | rap.Rmd §lexiques |

> **Pourquoi les lexiques ne sont qu'approximés.** Trois raisons distinctes :
> - *Ressources vivantes.* Les catégories du Wiktionnaire sont éditées en
>   continu ; leur contenu de 2024 n'est pas archivé. Testées sur le corpus,
>   elles ne reproduisent les compteurs que dans 8 à 46 % des cas.
> - *Ressource sous licence.* **LIWC est un dictionnaire commercial**, non
>   redistribuable. `n_positive` et `n_negative` ne peuvent pas être recalculés
>   sans licence — c'est une impossibilité de droit, pas de méthode.
> - *Imprécision de l'article.* Celui-ci présente la liste d'insultes comme
>   « basée sur le Wiktionnaire » ; son dépôt indique une liste **anglaise**
>   issue du projet Google « what do you love ». Aucune liste d'insultes
>   françaises n'est donc employée, ce qui explique la faible corrélation (0,33)
>   entre la liste récupérée et la colonne publiée.
>
> Une reconstruction des lexiques *par inférence* depuis le corpus (élimination
> par les titres à compteur nul) a été tentée et **écartée** : elle produit des
> ensembles aberrants (29 000 mots pour les onomatopées), le vocabulaire du
> corpus étant trop vaste et les paroles trop instables.

### Popularité — relations retrouvées par rétro-ingénierie

| Colonne | Signification | Méthode | Repro. |
|---|---|---|---|
| `pageview_mean` | Moyenne annuelle des vues | Constante par année (37 valeurs). Repli à **0,80089547** pour les titres sans année | **Oui** (relue dans LRFAF) |
| `pageviews_2` | Vues normalisées (log) | `log(pageviews / pageview_mean[year])`, et **0** quand `pageviews = 0` | **Oui** (100 %) |
| `pageviews_corrected` | Vues corrigées de l'année | `log(pageviews + 10) + f(year)`, `f` étant un décalage annuel. Vaut `NA` si l'année est inconnue | **Oui** (100 %) |

> Ces trois formules ont été retrouvées numériquement, non documentées. Celle de
> `pageviews_corrected` est exacte **au bruit machine près** (écart-type
> intra-année de 1e−14). La forme du lissage `f(year)` n'est pas publiée : les
> valeurs sont donc *relues* dans le corpus année par année plutôt que
> réestimées — ce qui les rend exactes pour toute année déjà couverte.

### Modèles de discours

| Colonne | Modèle | Méthode | Repro. |
|---|---|---|---|
| `hate` | [`Hate-speech-CNERG/dehatebert-mono-french`](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-french) | Probabilité par **ligne**, agrégée au titre | **Identifié, non exécuté** |
| `sexism` | [`annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal`](https://huggingface.co/annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal) | idem | **Identifié, non exécuté** |
| `sentiment2` | ? | **Aucun modèle documenté.** La colonne n'apparaît dans le code de l'article que sous forme commentée (`#positivity = weighted.mean(sentiment2, n_lines)`) | **Non** |

> Les deux modèles sont nommés dans l'article, publics, et ont été téléchargés ;
> le code d'inférence est fourni (`lrfaf_pipeline.score_lines`). Leur exécution
> n'a pas abouti dans cet environnement (blocage de PyTorch au chargement), et
> les colonnes sont donc laissées vides plutôt que remplies par un substitut.
> Deux réserves subsisteraient de toute façon : la version des poids employée en
> 2024 n'est pas épinglée, et **l'agrégation ligne → titre n'est pas
> documentée** (la moyenne simple est une hypothèse ; seule l'agrégation
> titre → artiste est explicitement une moyenne pondérée par `n_lines`).

### Métadonnées d'artiste

| Colonne | Source | Méthode | Repro. |
|---|---|---|---|
| `birthdate_artist` | Wikidata | Année de naissance ; pour un groupe, **moyenne des membres** | **Non** (voir ci-dessous) |
| `age_artist` | Calculé | `year − birthdate_artist` — exact sur 100 % du corpus | **Oui** (dérivée) |
| `born_in_france` | Wikidata | **Proportion des membres nés en France** — d'où les valeurs fractionnaires du corpus (6/7 ≈ 0,857 ; 2/3 ; 1/2) | **Non** (voir ci-dessous) |

> Ces colonnes sont vides pour Mikeysem par *absence de donnée source* :
> **il n'a ni entrée Wikidata ni article Wikipédia**. Ce n'est pas un défaut du
> pipeline. Le corpus lui-même ne renseigne `birthdate_artist` que pour 40,9 %
> de ses lignes — l'absence est le cas courant, y compris pour PNL, Damso ou
> OrelSan.

---

## B. Détail des deux mécanismes clés

### B.1 Nettoyage des paroles — reproduit exactement

Le corpus porte la signature de `lyricsgenius` appelé avec
`remove_section_headers=True` :

```python
lyrics = re.sub(r'(\[.*?\])*', '', lyrics)   # retire [Couplet 1], [Refrain]…
lyrics = re.sub('\n{2}', '\n', lyrics)       # exactement DEUX sauts -> UN
```

Le détail décisif est le second appel : il remplace les paires de sauts de ligne,
**et non les séquences de trois ou plus**. Une règle naïve (`\n{3,}` → `\n\n`)
tombe juste sur certains titres et faux sur d'autres ; celle-ci reproduit les
deux cas. Validation sur 25 587 titres appariés à l'archive brute de février
2024 : **97,9 %** de correspondance exacte ou en préfixe (les textes d'archive
sont tronqués), contre 83 % pour la règle naïve.

À cela s'ajoute le retrait de l'en-tête Genius (`^.*?Lyrics`), **postérieur au
calcul des compteurs** — c'est ce décalage qui explique le constat n°2 du §0.

### B.2 Tokenisation — reproduite

L'article donne la fonction R `get_complexity`, qui remplace les apostrophes par
des espaces avant d'appeler `tokenize_words`. L'équivalent retenu est
`re.findall(r"\w+", re.sub(r"['’]", " ", text).lower())`.

**Preuve** : sur les titres où `n_words` est reproduit exactement,
`n_unique_words` l'est aussi dans **99,9 %** des cas. Les deux colonnes reposent
donc bien sur cette tokenisation ; les écarts résiduels viennent du texte, pas
de la méthode.

---

## C. Contrôle de reproduction (`09_controle_reproduction.py`)

### Relations arithmétiques internes — 100 % exactes

| Relation | n | Exact |
|---|---|---|
| `n_non_french_words = n_words − n_french_words` | 37 307 | **100 %** |
| `pageviews_corrected = log(pageviews+10) + f(year)` | 33 696 | **100 %** |
| `pageviews_2 = log(pageviews / pageview_mean)` | 10 547 | **100 %** |
| `pageviews_2 = 0` quand `pageviews = 0` | 26 760 | **100 %** |
| `age_artist = year − birthdate_artist` | 14 064 | **100 %** |

### Contrôle A — pipeline appliqué aux textes bruts d'archive

25 587 titres appariés ; **97,9 %** de paroles reproduites. Sur les 2 374 titres
au texte strictement identique :

| Colonne | Exact | Écart médian | \|écart\| ≤ 2 |
|---|---|---|---|
| `n_lines` | 84,5 % | 0 | 97,7 % |
| `n_je` | 82,9 % | 0 | 88,7 % |
| `n_unique_words` | 71,0 % | 0 | 89,6 % |
| `n_words` | 60,1 % | 0 | 74,3 % |
| `means_word_length` | 59,6 % | −0,0 | 100 % |

Les écarts sont **centrés sur zéro** et petits : ils reflètent le fait que même
ces textes d'archive constituent une troisième version, distincte de celle ayant
servi aux compteurs.

### Contrôle B — cohérence interne du corpus publié

| Colonne | Exact | Écart médian |
|---|---|---|
| `n_words` | 10,1 % | −3 |
| `n_unique_words` | 9,7 % | −4 |
| `n_french_words` | 30,1 % | +2 |
| `n_je` | 79,8 % | 0 |

Ce tableau ne mesure pas une erreur du pipeline : il mesure le décalage interne
du corpus, décrit au §0.

---

## D. Application à Mikeysem

### Vérification d'identité

- **Un seul artiste** nommé Mikeysem sur Genius : identifiant **3152412**, 9 titres,
  tous rattachés à ce même identifiant. Aucun homonyme.
- Rappeur francophone de l'Essonne ; page sur [Le Plan, Ris-Orangis](http://leplan.com/site/mikeysem/) ;
  SoundCloud `junkee-officiel`.
- **Ni entrée Wikidata, ni article Wikipédia.** La recherche « Mikeysem » sur
  Wikipédia FR ne renvoie que l'article *Ziak*, où il figure en infobox avec la
  mention explicite « spéculation car identité non dévoilée ».

> **Conséquence méthodologique** : Mikeysem n'aurait **pas pu** figurer dans
> LRFAF. Le corpus part des catégories Wikipédia et de la liste Genius des
> comptes vérifiés ; sans page Wikipédia, l'artiste n'était pas dans la liste de
> départ. L'ajouter étend le corpus au-delà de son critère d'inclusion d'origine
> — ce qui est le but ici, mais doit être déclaré.

### Sélection des titres

9 titres sur Genius, **7 retenus** par les critères LRFAF :

| Titre exclu | Motif | Fidèle à LRFAF ? |
|---|---|---|
| `Cavale` | Artiste principal = Tsonpa | Oui — `include_features=False` |
| `Phone` | `language = "en"` selon Genius | Oui, **mais le texte est en français** |

> `Phone` est un **faux négatif connu** du filtre de langue : ses paroles sont
> françaises, mais Genius les déclare anglaises. L'article LRFAF signale
> exactement ce cas pour « Ma Benz » et l'assume : « il vaut mieux manquer
> quelques titres que polluer le corpus ». Le titre est donc exclu du CSV
> principal par fidélité, et conservé dans `result/08_mikeysem_collecte.csv`.

Autre particularité : `Phone` utilise des accolades (`{Refrain}`) et non des
crochets pour ses balises de section — que le nettoyage de `lyricsgenius` ne
retire donc pas. Le corpus original présente la même anomalie sur 0,1 % de ses
titres ; le comportement est conservé tel quel.

### Livrables

| Fichier | Contenu |
|---|---|
| `mikeysem_lrfaf.csv` | 7 lignes, 33 colonnes, ordre et types de LRFAF. **Aucune valeur estimée** : les colonnes non reproductibles sont vides |
| `result/08_mikeysem_estimations.csv` | Colonnes approximées, suffixées `_est` |
| `result/08_mikeysem_collecte.csv` | Données brutes Genius des 9 titres, avant tout calcul |

---

## E. Différences méthodologiques à déclarer

1. **Date de collecte.** Paroles, `pageviews` et `contributors` datent de
   septembre 2026 ; le reste du corpus de début 2024. Les compteurs de
   popularité ne sont pas comparables en niveau.
2. **Version des lexiques.** Wiktionnaire et Morphalou dans leur version
   actuelle, non celle de 2024.
3. **`n_lines`.** Définition inférée (lignes ≥ 2 mots), exacte à 84,5 %.
4. **`n_profanity`.** Liste anglaise, conformément au dépôt réellement cité par
   l'article — et non une liste française comme sa prose le laisse entendre.
5. **Colonnes vides.** 9 colonnes non renseignées, dont 3 par absence de source
   (Wikidata) et 6 par indisponibilité de la méthode.
6. **Aucune valeur inventée.** Aucune colonne du fichier principal ne contient
   d'estimation, d'imputation ou de valeur par défaut non conforme à LRFAF.

---

## F. Sources

**Corpus et documentation**
- Jeu de données : <https://huggingface.co/datasets/regicid/LRFAF> (DOI 10.57967/hf/3316)
- Code de l'article : [`article/rap.Rmd`](https://huggingface.co/datasets/regicid/LRFAF/blob/main/article/rap.Rmd) — *code d'analyse, non de production*
- Préprint : <https://osf.io/preprints/socarxiv/d96tr>
- Dépôt GitHub : <https://github.com/regicid/LRFAF> (README seul)
- Textes bruts et listes d'artistes : <https://github.com/regicid/genius_french_rap_corpus> (aucun code)
- Version brute : <https://huggingface.co/datasets/regicid/lrfaf_v2> (85 023 lignes, colonnes de collecte uniquement)

**Outils et ressources**
- `lyricsgenius` (John W. Miller) — <https://github.com/johnwmillr/LyricsGenius>
- Morphalou 3.1, ATILF/ORTOLANG, licence LGPL-LR — <https://www.ortolang.fr/market/lexicons/morphalou/v3.1>
- Bunka (Charles de Dampierre) — <https://github.com/charlesdedampierre/BunkaTopics>
- Solon-large (OrdalieTech) ; LIWC-fr (Piolat et al., 2011)
- Wiktionnaire : catégories Verlan, Sexualité, Annexe des termes argotiques
- `MauriceButler/badwords` — <https://github.com/MauriceButler/badwords>

**Reproduire**

```bash
python3 collecte_mikeysem.py       # collecte Genius (9 titres)
python3 08_ajout_mikeysem.py       # produit mikeysem_lrfaf.csv
python3 09_controle_reproduction.py # contrôles A, B et arithmétique
```

---

## G. Note sur l'usage

Ce travail prolonge l'étude stylométrique de `02_stylometrie_ziak.ipynb`, dont
la conclusion était qu'aucun des 392 artistes éligibles de LRFAF ne portait la
signature de Ziak. **Mikeysem ne figurait pas dans ce pool** — faute de page
Wikipédia, il était hors du critère d'inclusion du corpus. Ces 7 titres
permettent donc de tester une hypothèse qui, jusqu'ici, ne pouvait pas l'être.

Deux réserves avant d'y voir un test décisif. D'abord le volume : environ
3 800 mots, très en deçà des 12 000 utilisés comme seuil dans l'étude, ce qui
réduit fortement la puissance de la méthode. Ensuite le statut de l'hypothèse :
le rapprochement entre les deux noms est une spéculation d'auditeurs, que
l'intéressé a démentie, et qu'aucune source vérifiable n'étaye.
