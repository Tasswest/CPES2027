# Méthode, de bout en bout

Ce document raconte dans l'ordre ce que fait le projet : d'où viennent les
données, comment elles sont préparées, quel moteur les analyse, et quelle
analyse répond à quelle question. Il complète l'article, qui expose les
résultats, en exposant la mécanique.

Le fil directeur tient en une phrase : **une distance ne veut rien dire tant
qu'on ne sait pas ce qu'elle vaut**. Tout le reste en découle — le contrôle de
la taille, la validation sur des cas connus, la calibration contre une
hypothèse nulle.

---

## 0. Le point de départ

| | |
|---|---|
| Corpus | **LRFAF** (de Courson, 2024), 37 307 chansons de rap français |
| Source | genius.com, croisé avec les catégories Wikipédia / Wikidata |
| Question | Ziak, rappeur masqué apparu en 2020, est-il un autre artiste ? |

Trois hypothèses sont testées : Ziak serait un rappeur du corpus sous un autre
nom ; Ziak serait Mikeysem ; les textes de Ziak seraient écrits par web7
(ex-7 Jaws).

---

## 1. La récolte

### 1.1 Le corpus publié

Téléchargé tel quel depuis Hugging Face (`corpus.csv`, 114 Mo). Il n'est pas
versionné : trop volumineux pour GitHub, et régénérable par une commande.

### 1.2 Les artistes absents du corpus — `collecte_genius.py`

Mikeysem et web7 ne figurent pas dans LRFAF : sans page Wikipédia, ils tombent
hors de son critère d'inclusion. Il a fallu les collecter.

```bash
python3 collecte_genius.py <artist_id> <sortie.json>
```

Le script liste les morceaux d'un artiste via l'endpoint JSON interne de Genius
(`genius.com/api/artists/<id>/songs`, sans authentification), récupère les
métadonnées de chaque titre (date, album, vues, contributeurs, invités), puis
**scrape la page HTML** pour les paroles — l'API ne les renvoie pas, Genius les
licenciant séparément. C'est la méthode qu'emploie la bibliothèque
`lyricsgenius`, et celle qu'a employée LRFAF lui-même.

Aucun filtre n'est appliqué à la collecte : les critères d'inclusion de LRFAF
(artiste principal, langue française) sont appliqués plus tard, au moment de
l'analyse, pour que les exclusions restent visibles et motivées.

**Vérification d'identité.** Un homonyme ruinerait le test. Les pages ont été
vérifiées une à une : « Peter Punk » sur Genius est un groupe italien,
« Malsain » un groupe de metal — ni Disiz ni Sinik, contrairement à ce qu'on
pouvait espérer. web7 a été confirmé par le champ `alternate_names`, qui
contient « 7 Jaws ».

| Artiste | ID Genius | Titres retenus |
|---|---|---|
| Ziak | 2113831 | 91 collectés, 41 appariés aux titres LRFAF |
| web7 (ex-7 Jaws) | 1078135 | 78 satisfaisant aux critères du corpus |
| Mikeysem | 3152412 | 9 avec paroles, sur 21 titres de discographie |

### 1.3 La re-collecte du corpus entier — `19_collecte_corpus.py`

**Le problème.** LRFAF a supprimé les balises de section (`[Couplet 1 : X]`)
avant publication. Sans elles, impossible de savoir quelle strophe revient à
l'artiste principal et laquelle appartient à un invité : le couplet de l'invité
est attribué au propriétaire du morceau. Comme l'hypothèse testée porte
précisément sur une proximité entre artistes, cette confusion fausse la mesure.

Vérification chiffrée : **aucune balise sur 400 textes tirés au hasard**, et
15 titres sur 37 307 signalent un featuring dans leur titre. L'information
n'est pas récupérable depuis le corpus.

**La solution.** LRFAF conserve l'URL Genius de chaque titre, et la page porte
encore ses balises. Le script re-télécharge les 34 732 pages des 393 artistes
éligibles.

| Réglage | Valeur | Pourquoi |
|---|---|---|
| `N_FILS` | 4 | ~1,7 page/s au total, courtois pour un site de cette taille |
| `PAUSE` | 1,0 s + aléa | par fil, pour lisser le débit |
| `MAX_ESSAIS` | 4 | repli exponentiel sur 429 / 403 / 5xx |
| Sortie | JSONL en ajout seul | relancer le script reprend où il s'est arrêté |

La reprise n'est pas un luxe : la collecte dure environ six heures.

---

## 2. Le prétraitement

### 2.1 Nettoyage des paroles — `stylo_features.clean_lyrics`

Dans l'ordre :

1. neutralisation des balises `[...]` résiduelles ;
2. normalisation des apostrophes (`’`, `‘`, `` ` ``, `´` → `'`) ;
3. passage en minuscules ;
4. suppression des retours à la ligne ;
5. suppression de tout ce qui n'est ni lettre, ni accent, ni apostrophe.

Deux choix comptent. **Les accents sont conservés** — ils portent du signal en
français. **L'apostrophe est conservée** — elle porte l'élision, donc une part
de la grammaire.

### 2.2 Tokenisation et n-grammes

```python
tokenize(text)    # [a-zàâä…]+'?   →  « j'ai » devient « j' » + « ai »
char_ngrams(text) # 4-grammes de caractères, espaces compris
```

L'élision est séparée volontairement : `j'` et `je` deviennent deux traits
distincts. C'est ce qui a permis de démasquer un faux marqueur (§ 4.5).

### 2.3 Filtres du corpus — `load_corpus`

| Filtre | Effet |
|---|---|
| Textes de moins de 100 tokens | écartés : trop courts pour porter une signature |
| Doublons de paroles entre artistes | dédoublonnés sur les 300 premiers caractères |

Le dédoublonnage compte : un même texte publié sous deux artistes (featuring,
réédition) attribuerait le même texte à deux auteurs, ce qui fausse
mécaniquement les distances.

**37 307 → 32 923 titres, 596 artistes.**

### 2.4 Vocabulaires et cache — `build_cache`

| Famille | Taille | Usage |
|---|---|---|
| `counts_mfw` | 500 mots les plus fréquents | Delta de Burrows |
| `counts_char` | 3 000 4-grammes les plus fréquents | Cosine Delta |
| `counts_word_ext` | 6 000 mots | marqueurs lexicaux seulement |

Les comptes sont stockés **par chanson**, jamais par artiste. C'est le choix
structurant : il permet de recomposer n'importe quel « document d'artiste » de
taille voulue par simple somme, ce qui rend possibles les centaines de
rééchantillonnages dont dépend toute la validation. Sans ce cache, le protocole
serait inexécutable en temps raisonnable.

### 2.5 Retrait des strophes d'invités — `genius_sections.py` et `20_corpus_sans_invites.py`

Une balise Genius nomme, le cas échéant, l'interprète de la section :

```
[Couplet 1]                          → artiste principal (convention)
[Refrain : Captaine Roshi & 7 Jaws]  → deux interprètes
[Couplet 3 - Kool Shen]              → tiret, équivalent au deux-points
[Sofiane]                            → balise réduite au nom
[?]                                  → mot inaudible, pas une section
```

**Le piège des groupes.** La règle naïve — retirer toute section nommant
quelqu'un d'autre — vide les groupes de leur contenu : chez Suprême NTM, les
balises nomment Kool Shen et Joey Starr, dont les couplets *sont* le texte du
groupe. La règle retenue distingue le membre de l'invité par sa récurrence :

| Clause | Seuil | Exemple |
|---|---|---|
| Part du répertoire | ≥ 40 % des titres | Akhenaton, 64 titres d'IAM sur 105 |
| Intervenant récurrent | ≥ 4 titres **et** ≥ 5 % | Freeman, 7 titres sur 105 — membre lui aussi |
| Garde-fou | > 50 % de mots retirés | l'artiste est laissé intact, et signalé |

**Les deux erreurs possibles ne se valent pas.** Garder par erreur le couplet
d'un invité ne fait que reproduire le défaut de LRFAF ; retirer par erreur le
couplet d'un membre ampute un artiste de sa propre écriture, et fabrique un
défaut que le corpus publié n'a pas. Le réglage penche donc vers la
conservation.

Un titre sans balise est conservé tel quel : l'absence de balise ne prouve pas
l'absence d'invité, et écarter ces titres biaiserait le corpus vers les
artistes les mieux annotés.

### 2.6 Le pipeline LRFAF reconstitué — `lrfaf_pipeline.py`

Pour ajouter Mikeysem au corpus, il fallait produire ses lignes **dans le
format exact de LRFAF**, dont le code de production n'a jamais été publié. Il a
été reconstitué depuis la prose de l'article, le code source de `lyricsgenius`
et de la rétro-ingénierie arithmétique sur le corpus lui-même.

```python
def lg_clean(raw):            # le nettoyage exact de lyricsgenius
    t = re.sub(r"(\[.*?\])*", "", raw)
    return re.sub("\n{2}", "\n", t)

def strip_genius_header(text): # « 12 ContributorsTitre Lyrics » en tête de page
    return re.sub(r"^.*?Lyrics", "", text, count=1, flags=re.S)
```

Cinq relations arithmétiques internes sont reproduites **à 100 %**. Le détail
colonne par colonne — reproduit exactement, approximé, ou impossible à
retrouver — est dans [`RAPPORT_LRFAF.md`](RAPPORT_LRFAF.md). Rien n'a été
inventé : les colonnes irreproductibles (LIWC-fr, commercial ; Bunka topics)
restent vides, et les estimations vivent dans un fichier séparé.

---

## 3. Le moteur d'attribution — `stylo_attribution.py`

Quatre opérations, dans cet ordre.

**1. Échantillonnage à taille contrôlée.** Chaque document d'artiste vaut
exactement `T_CAND` tokens, tirés sans remise parmi ses chansons. Sans cette
contrainte, la distance mesure surtout la couverture de vocabulaire : un
artiste prolifique paraît proche de tout le monde.

**2. Agrégation en fréquences.** `make_docs` somme les comptes par groupe puis
divise par le total.

**3. Distance standardisée.** Les z-scores sont estimés **sur les candidats
seuls**, et la requête est projetée dans ce référentiel — comme un texte
anonyme confronté à un corpus connu.

```python
cosine_delta   # 1 − cosinus des z-scores  (Smith & Aldridge 2011)
burrows_delta  # Manhattan moyenne des z-scores  (Burrows 1992)
```

**4. Score de séparation.** La distance brute au meilleur candidat n'est pas
interprétable seule. On la convertit :

> de combien d'écarts-types le premier se détache-t-il des autres ?

C'est cette grandeur, et non la distance, qui est comparable d'une requête à
l'autre — et donc la seule qui permette de confronter Ziak à des cas de
contrôle.

---

## 4. Les analyses, dans l'ordre d'exécution

### 4.1 `02_diagnostic_biais_taille.py` — pourquoi l'approche intuitive échoue

Trois variantes jugées au **même** protocole de vérité-terrain :

| Variante | Rang 1 |
|---|---|
| A. naïve — corpus complets, cosinus | 25,0 % |
| B. taille contrôlée | 73,5 % |
| C. taille contrôlée + standardisation | 91,5 % |

L'approche naïve produit un classement d'allure sérieuse et se trompe trois
fois sur quatre. C'est le résultat méthodologique de l'article.

### 4.2 `03_validation_protocole.py` — que vaut la méthode ?

Pour chaque artiste de contrôle : on prélève un texte-requête **de la taille
exacte du corpus de Ziak** (23 886 tokens), et un « jumeau » de 12 000 tokens
issu de titres *disjoints*, placé dans le pool sous une autre étiquette.

Deux conditions :

- **H₁** — le jumeau est présent : mesure la puissance ;
- **H₀** — le jumeau est retiré : reproduit un auteur absent du corpus.

1 770 essais sur 177 artistes. Meilleure combinaison (4-grammes + Cosine
Delta) : **90 % au premier rang**. C'est cette puissance qui rendra un résultat
négatif informatif.

### 4.3 `04_attribution_ziak.py` — le verdict

Trois questions enchaînées :

1. *Le style de Ziak est-il seulement détectable ?* Contrôle positif : sa
   propre seconde moitié est retrouvée **dans 100 % des tirages**. Aucun échec
   ultérieur ne pourra être imputé à un manque de matière.
2. *Qui est le plus proche ?* 30 rééchantillonnages × 4 combinaisons.
3. *Ce meilleur candidat est-il crédible ?* Sa séparation est confrontée aux
   distributions H₁ et H₀ établies à l'étape précédente.

Les quatre combinaisons **ne s'accordent même pas sur un favori** — alors
qu'elles convergent neuf fois sur dix quand la réponse existe. Toutes penchent
vers H₀.

### 4.4 `05_robustesse.py` — quatre objections

Puissance par génération (maximale sur celle de Ziak), sensibilité au seuil de
12 000 mots, **test des imposteurs** (Koppel & Winter 2014 : un candidat précis
affronte des inconnus tirés au sort), restriction aux contemporains.

### 4.5 `06_profil_stylistique.py` — le portrait

Excentricité, marqueurs lexicaux par **log-odds à prior informatif** (Monroe
et al. 2008), stabilité temporelle, voisinage.

C'est ici qu'un faux marqueur a été démasqué : Ziak semblait sous-employer
« je » d'un facteur 5. En réalité il emploie « j' » 1,2 fois plus, et **une
fois les deux formes additionnées l'écart disparaît**. Ce n'était pas une
habitude d'écriture mais une convention de transcription. La moitié des paires
d'élision testées relève du même phénomène.

### 4.6 `08_` et `09_` — Mikeysem au format LRFAF, et ses contrôles

Production des lignes, puis vérification arithmétique contre le corpus publié.

Au passage, une découverte sur LRFAF lui-même : **la colonne `n_words`
correspond aux paroles stockées dans seulement 10,2 % des lignes**, avec un
écart médian de −4 mots — la signature de l'en-tête Genius. Le corpus est
interne­ment incohérent sur ce point.

### 4.7 `10_test_ziak_mikeysem.py` — l'hypothèse Mikeysem

Une difficulté domine : Mikeysem ne pèse que 3 745 mots. **Tout le protocole
est donc recalibré à cette taille, et sa puissance mesurée avant toute
interprétation** — un résultat négatif obtenu avec une méthode aveugle ne
vaudrait rien.

### 4.8 `13_` et `14_` — la validation sur des liens réels

L'objection la plus sérieuse au verdict : les « jumeaux » des étapes 4.2 et 4.3
sont *fabriqués*, donc faciles à retrouver.

- **`13_`** : 14 paires solo / groupe, où l'artiste a réellement écrit une part
  des textes du groupe. Plus sévère qu'un alias — dans un trio, l'auteur ne
  signe qu'un tiers du texte.
- **`14_`** : 3 changements de nom documentés, testés en découpant l'artiste de
  part et d'autre de sa rupture. **Joke → Ateyaba est retrouvé au premier rang
  sur 393, dans la totalité des tirages** — malgré un changement d'identité
  revendiqué.

Les deux tirent en sens opposés, et il faut les lire ensemble : la puissance
est *plus faible* qu'annoncée quand l'auteur ne signe qu'une partie des textes,
mais elle ne s'effondre *pas* quand il change d'identité — ce qui est
précisément la situation testée sur Ziak.

### 4.9 `15_` et `16_` — l'hypothèse web7

`15_` teste au niveau de l'artiste, en trois variantes, plus un test inverse
(Ziak vu depuis web7). `16_` exploite une **expérience naturelle** : sur
l'album *Essonne History X*, 8 titres sont crédités à web7 et 15 ne le sont
pas. Même artiste, même année, même disque.

Test de permutation : la distance du profil des 8 titres crédités au profil de
web7, comparée à 5 000 tirages de 8 titres parmi les 23. **Deux contrôles de
puissance** établissent ce que le test aurait su voir — on remplace les titres
crédités par du texte entièrement de web7, puis par un mélange à parts égales.

Les featurings et les ad-libs sont retirés des deux côtés.

### 4.10 `18_controle_featurings.py` — les invités faussent-ils le classement ?

Trois mesures, avant que la re-collecte complète ne soit disponible :

1. la requête Ziak reconstruite depuis Genius, invités retirés — avec une
   troisième variante (même source, invités conservés) qui **sépare l'effet du
   nettoyage de celui du changement de source** ;
2. les titres du corpus contenant un couplet de Ziak, retirés du corpus des
   candidats ;
3. une simulation : du texte étranger injecté chez *tous* les candidats, à
   taux croissant, pour mesurer à partir de quand la méthode lâche.

### 4.11 `19_` et `20_` — le corpus nettoyé

Décrits en § 1.3 et § 2.5. Les deux corpus coexistent : la variable
d'environnement `CORPUS_VARIANTE` choisit lequel analyser et **isole cache et
résultats**, pour que l'étude puisse être rejouée sur le corpus propre sans
écraser la version publiée.

```bash
CORPUS_VARIANTE=sans_invites python3 04_attribution_ziak.py
```

---

## 5. La restitution

| Script | Produit |
|---|---|
| `07_figures.py` | les 10 figures, depuis les CSV de `export/` |
| `article_contenu.py` | le texte de l'article, **relisant tous les chiffres dans `export/`** |
| `12_article_pdf.py` | le PDF (reportlab) |
| `17_article_docx.py` | le Word (python-docx) |

Le texte étant partagé, les deux formats **ne peuvent pas diverger**. Et aucun
chiffre n'est écrit à la main dans la prose : tout est relu depuis les
résultats au moment de la génération, si bien qu'une analyse rejouée met
l'article à jour.

---

## 6. Les paramètres, en un tableau

| Paramètre | Valeur | Où |
|---|---|---|
| Mots les plus fréquents | 500 | `stylo_features` |
| 4-grammes de caractères | 3 000 | `stylo_features` |
| Vocabulaire élargi | 6 000 | `stylo_features` |
| Longueur minimale d'un titre | 100 tokens | `load_corpus` |
| Clé de dédoublonnage | 300 premiers caractères | `load_corpus` |
| Taille d'un document candidat | 12 000 tokens | partout |
| Taille de la requête Ziak | 23 886 tokens (son corpus entier) | `03_`, `04_` |
| Artistes éligibles | 393 (≥ 12 000 tokens) | partout |
| Rééchantillonnages | 6 à 30 selon le script | — |
| Permutations (test 2025) | 5 000 | `16_` |
| Graines aléatoires | fixées, une par script | reproductibilité |

Chaque script fixe sa propre graine : deux exécutions donnent exactement les
mêmes chiffres.

---

## 7. Ce qui est en cours

La re-collecte des 34 732 titres tourne. À son terme :

1. `20_corpus_sans_invites.py` reconstruit le corpus ;
2. l'étude est rejouée sur la variante `sans_invites` ;
3. l'article compare les trois hypothèses **avant et après** nettoyage.

Un correctif de parsing récent — Genius sépare le libellé de ses interprètes
par un deux-points *ou* par un tiret, et seul le premier était lu — impose de
rejouer aussi `15_` et `16_`. Les chiffres publiés sur web7 pourront bouger
légèrement.
