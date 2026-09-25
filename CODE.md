# Le code, bloc par bloc

[`METHODE.md`](METHODE.md) raconte *ce que* fait le projet et *pourquoi*. Ce
document-ci explique *comment* : chaque module, chaque bloc, ce qu'il lit, ce
qu'il calcule, ce qu'il écrit.

Le code se lit en trois couches :

| Couche | Fichiers | Rôle |
|---|---|---|
| **Moteur** | `stylo_features.py`, `stylo_attribution.py`, `genius_sections.py` | tout le reste s'appuie dessus |
| **Analyses** | `02_` à `20_` | une question par script, un CSV par réponse |
| **Restitution** | `07_figures.py`, `article_contenu.py`, `12_`, `17_` | figures et article |

---

# 1. Le moteur

## 1.1 `stylo_features.py` — préparer le texte

### Le sélecteur de corpus

```python
VARIANTE = os.environ.get("CORPUS_VARIANTE", "brut")
CACHE_DIR = Path(".cache_stylo" if VARIANTE == "brut" else f".cache_stylo_{VARIANTE}")
```

Trois corpus coexistent : le corpus publié, celui dont les strophes d'invités
sont retirées, celui dont les titres à invités sont écartés. Une variable
d'environnement choisit lequel, et **isole cache et résultats**. C'est ce qui
permet de rejouer toute l'étude sur un corpus nettoyé sans écraser les chiffres
de la version publiée — donc de comparer.

### `clean_lyrics(text)` — normaliser

Cinq opérations, dans l'ordre : neutraliser les balises `[...]` résiduelles,
normaliser les apostrophes typographiques vers `'`, passer en minuscules,
aplatir les retours à la ligne, supprimer tout ce qui n'est ni lettre, ni
accent, ni apostrophe.

```python
NON_LETTERS = re.compile(r"[^a-zàâäéèêëïîôöùûüÿçœæ'\s]+")
```

Deux caractères sont **délibérément conservés** : les accents, qui portent du
signal en français, et l'apostrophe, qui porte l'élision — donc de la
grammaire.

### `tokenize(text)` — découper

```python
return re.findall(r"[a-zàâäéèêëïîôöùûÿçœæ]+'?", text)
```

L'apostrophe finale est capturée avec le mot : `j'ai` donne `j'` puis `ai`.
**`je` et `j'` deviennent donc deux traits distincts.** C'est ce choix qui a
permis de démasquer un faux marqueur : Ziak paraissait sous-employer « je »
d'un facteur 5, alors qu'il compense entièrement en « j' ».

### `char_ngrams(text, n=4)` — les 4-grammes

Fenêtre glissante de quatre caractères, **espaces compris**. Les espaces
comptent : ils capturent les fins et débuts de mots, donc une partie de la
syntaxe, sans passer par le lexique.

### `load_corpus(min_tokens=100)` — filtrer

```python
df = df[df["n_tokens"] >= min_tokens]
df["dup_key"] = df["lyrics_clean"].str.slice(0, 300)
df = df.drop_duplicates(subset="dup_key", keep="first")
```

Deux filtres. Les textes de moins de 100 tokens sont écartés : trop courts pour
porter une signature. Les doublons de paroles entre artistes — un même morceau
publié sous deux noms — sont supprimés, **sans quoi le même texte serait
attribué à deux auteurs**, ce qui fausse mécaniquement les distances.

Résultat : 37 307 → 32 923 titres, 596 artistes.

### `build_count_matrix(docs, vocab)` — vectoriser

Construit une matrice creuse (documents × vocabulaire) de comptes bruts, en
n'indexant que les tokens présents dans le vocabulaire. Le format creux est
imposé par la taille : 33 000 documents × 3 000 4-grammes.

### `build_cache(force=False)` — le cœur

Trois vocabulaires, calculés **sur tout le corpus** :

| Matrice | Taille | Usage |
|---|---|---|
| `counts_mfw` | 500 mots les plus fréquents | Delta de Burrows |
| `counts_char` | 3 000 4-grammes les plus fréquents | Cosine Delta |
| `counts_word_ext` | 6 000 mots | marqueurs lexicaux seulement |

**Le point décisif : les comptes sont stockés par chanson, jamais par artiste.**
Un « document d'artiste » de n'importe quelle taille se recompose ensuite par
simple somme de lignes. Sans cela, les centaines de rééchantillonnages dont
dépend toute la validation demanderaient de revectoriser le corpus à chaque
tirage — le protocole serait inexécutable.

## 1.2 `stylo_attribution.py` — mesurer une distance

### `sample_indices(song_ids, n_tokens, target, rng)`

```python
order = rng.permutation(len(song_ids))
cumsum = np.cumsum(n_tokens[order])
k = int(np.searchsorted(cumsum, target) + 1)
```

Tire des chansons **sans remise** jusqu'à atteindre le nombre de tokens voulu.
On échantillonne des chansons entières, pas des mots : découper au milieu d'un
morceau mélangerait des contextes.

### `make_docs(counts, groups)`

Somme les comptes par groupe, puis divise par le total → fréquences relatives.

### `rank_candidates(query_freq, cand_freqs, metric)`

```python
mu = cand_freqs.mean(axis=0)
sd = cand_freqs.std(axis=0, ddof=0)
cand_z = (cand_freqs - mu) / sd
query_z = (query_freq - mu) / sd
```

Le détail qui compte : **les z-scores sont estimés sur les candidats seuls**, et
la requête est projetée dans ce référentiel. C'est la situation réelle d'un
texte anonyme confronté à un corpus connu — inclure la requête dans le calcul
de la moyenne la ferait participer à sa propre normalisation.

### `cosine_delta` et `burrows_delta`

```python
cosine_delta   # 1 − cosinus des z-scores      (Smith & Aldridge 2011)
burrows_delta  # Manhattan moyenne des z-scores (Burrows 1992)
```

Deux distances plutôt qu'une, croisées avec deux jeux de traits : quatre
analyses indépendantes. Leur **désaccord** est un résultat en soi — sur les cas
de contrôle elles convergent neuf fois sur dix ; sur Ziak, elles divergent.

### `separation_score(dists, best_idx)`

```python
others = np.delete(dists, best_idx)
return float((dists[best_idx] - others.mean()) / others.std(ddof=0))
```

La question n'est pas « qui est premier ? » — il y a toujours un premier — mais
« de combien se détache-t-il ? ». Ce z-score négatif est **la seule grandeur
comparable d'une requête à l'autre**, donc la seule qui permette de confronter
Ziak à des cas dont on connaît la réponse.

## 1.3 `genius_sections.py` — savoir qui chante quoi

### `interpretes(balise)`

Lit une balise Genius et renvoie les interprètes nommés, ou `None` si elle n'en
nomme aucun.

```python
tiret = re.match(r"^([^:\-–—]+?)\s*[-–—]\s*(.+)$", contenu)
if ":" in contenu:              noms = contenu.split(":", 1)[1]
elif tiret and _est_libelle(tiret.group(1)):  noms = tiret.group(2)
elif _est_libelle(contenu):     return None
else:                           noms = contenu
```

L'ordre des branches est le correctif d'un bug réel. Genius emploie le
deux-points **et** le tiret : `[Couplet 3 - Kool Shen]` passait pour une section
de l'artiste principal. Mais le tiret ne peut être accepté que si un libellé de
section le précède, sinon `Pre-refrain` et `Jay-Z` seraient coupés en deux.

### `est_balise_de_section(ligne)`

```python
return bool(m) and "?" not in m.group(1)
```

Une ligne entre crochets contenant un `?` note un mot inaudible, pas une
section. Sans ce garde, `[?]` ouvrirait une fausse section.

### `retire_featurings(texte, alias_principal)`

Parcourt le texte ligne à ligne en maintenant un drapeau « la section courante
appartient-elle à l'artiste principal ? », et compte les mots gardés et
retirés. Règle : une section n'est conservée que si elle n'est attribuée à
personne, ou au seul artiste principal. **Une section partagée est retirée**,
faute de pouvoir savoir qui en a écrit quoi.

### `texte_selon_option(...)`

Les deux façons d'écarter les featurings, à partir du même repérage :
`"parties"` retire les strophes d'invités et garde le reste du titre ;
`"titres"` écarte le titre en entier. Elles ne diffèrent que par ce qu'elles
font d'un morceau partagé.

---

# 2. Les analyses

## `02_diagnostic_biais_taille.py` — pourquoi l'intuition échoue

**Bloc 1 — le classement naïf.** Chaque artiste est représenté par son corpus
*intégral*, vectorisé en fréquences relatives, comparé à Ziak par cosinus brut.
On corrèle ensuite la distance obtenue à la taille du corpus du candidat.

**Bloc 2 — le même classement à taille contrôlée**, pour voir les rangs se
réorganiser.

**Bloc 3 — l'arbitrage.** Les trois variantes sont soumises au *même* protocole
de vérité-terrain :

```python
# A. naïve : les autres candidats gardent leur corpus intégral
# B. taille contrôlée, sans standardisation
# C. taille contrôlée + Cosine Delta
```

C'est l'argument central : on ne discute pas de quel classement paraît le plus
plausible, **on mesure lequel a raison**. 25,0 % / 73,5 % / 91,5 %.

## `03_validation_protocole.py` — que vaut la méthode ?

**Bloc 1 — le pool.** Un artiste de contrôle doit fournir requête *et* jumeau
sur des titres **disjoints** : sans cela, on retrouverait le même texte des deux
côtés.

**Bloc 2 — les deux conditions**, au cœur du script :

```python
# --- H1 : le jumeau est présent dans le pool ---
# --- H0 : le jumeau est retiré (auteur absent du corpus) ---
```

H₁ mesure la puissance. H₀ reproduit la situation d'un auteur absent du
corpus — **c'est-à-dire exactement l'hypothèse à laquelle Ziak sera confronté**.
Sans H₀, on n'aurait aucun repère pour interpréter son score.

**Bloc 3 — les seuils de décision** : distributions de séparation sous H₁ et
sous H₀. Le critère de décision naît de leur écart.

Sorties : 1 770 essais, 177 artistes, 90 % au premier rang.

## `04_attribution_ziak.py` — le verdict

**Bloc 1 — contrôle positif.** Avant tout, écarter l'explication triviale : le
corpus de Ziak est coupé en deux moitiés disjointes, et l'on cherche la seconde
depuis la première. Elle ressort première dans 100 % des tirages. **Aucun échec
ultérieur ne pourra être imputé à un manque de matière.**

**Bloc 2 — le classement**, 30 rééchantillonnages × 4 combinaisons.

**Bloc 3 — stabilité et consensus** : fréquence d'apparition dans le top-5, rang
moyen toutes méthodes confondues. C'est ici qu'apparaît le désaccord entre
méthodes.

**Bloc 4 — la confrontation.**

```python
# Vraisemblance relative sous chaque hypothèse (densités normales).
```

La séparation de Ziak est évaluée sous les deux distributions établies par
`03_`, et le rapport de vraisemblance tranche. Les quatre analyses désignent
H₀.

## `05_robustesse.py` — quatre objections

Quatre fonctions indépendantes, une par objection :

| Fonction | Répond à |
|---|---|
| `puissance_par_generation` | la méthode est-elle plus faible sur les artistes récents ? |
| `sensibilite_seuil` | le seuil de 12 000 mots exclut-il le bon candidat ? |
| `test_imposteurs(n_iter=200)` | ce candidat précis tient-il face à des inconnus tirés au sort ? |
| `imposteurs_reference` | **et que vaut ce score pour un vrai jumeau ?** |

La quatrième est celle qui donne son sens à la troisième. Un score d'imposteurs
ne se lit pas dans l'absolu : il faut savoir ce qu'atteint un alias authentique
de même taille. Sans ce repère, « Kerchak bat le hasard » ne voudrait rien dire.

## `06_profil_stylistique.py` — le portrait

**`log_odds_marqueurs`** — log-odds à prior informatif (Monroe et al. 2008) :

```python
# Prior informatif : la force totale `alpha0` est répartie sur les mots
# Un marqueur doit reposer sur assez d'occurrences pour être commentable.
```

Le prior est réparti **proportionnellement à la fréquence de chaque mot dans le
corpus**, et non uniformément. Une première version, avec prior scalaire,
produisait des z de 30 pour des ratios de 1,0 — un mot au comportement normal
paraissait un marqueur spectaculaire.

**`controle_elision`** — le garde-fou. Pour chaque paire (`je`/`j'`,
`de`/`d'`…), compare le ratio de chaque forme au ratio cumulé. La moitié des
paires révèle un artefact de transcription plutôt qu'une habitude d'écriture.

**`excentricite`** — deux mesures à ne pas confondre : la distance médiane au
corpus (est-il atypique ?) et la distance au plus proche voisin (a-t-il un
parent ?). Ziak est banal sur la première, isolé sur la seconde.

**`stabilite_temporelle`** — compare ses deux périodes, avec le bon repère :

```python
# Repère : écart entre deux moitiés aléatoires, chez Ziak et chez les autres.
```

Un écart entre deux périodes ne signifie rien si l'on ne sait pas ce que produit
une coupe arbitraire du même corpus.

## `08_` et `09_` — Mikeysem au format LRFAF

`08_ajout_mikeysem.py` produit ses lignes dans le format exact du corpus, via
`lrfaf_pipeline.py` — reconstitution du pipeline de LRFAF, dont le code n'a
jamais été publié. **Rien n'est inventé** : les colonnes irreproductibles
restent vides, les estimations vivent dans un fichier séparé, suffixées `_est`.

`09_controle_reproduction.py` vérifie le tout contre le corpus publié — et c'est
là qu'apparaît une incohérence de LRFAF lui-même : `n_words` ne correspond aux
paroles stockées que dans 10,2 % des lignes.

## `10_test_ziak_mikeysem.py` — l'hypothèse Mikeysem

L'ordre des fonctions est l'argument :

```python
validation(...)      # 1. que vaut la méthode à 3 745 mots ?
rank_mikeysem(...)   # 2. seulement ensuite, son rang
imposteurs(...)      # 3. et son score par paire
controles_complementaires(...)
```

**La puissance est mesurée avant que le résultat ne soit regardé.** Un résultat
négatif obtenu avec une méthode aveugle ne vaudrait rien — et à 3 745 mots, la
question se pose vraiment.

## `13_` et `14_` — la validation sur des liens réels

Ces deux scripts répondent à l'objection la plus sérieuse : les jumeaux de
`03_` sont *fabriqués*, donc faciles à retrouver.

`13_validation_alias_reels.py` teste 14 paires solo/groupe — plus sévère qu'un
alias, puisque dans un trio l'auteur ne signe qu'un tiers du texte. Le dernier
bloc compare ces cas à Ziak et à Mikeysem, **seule comparaison légitime** :
même grandeur, même protocole.

`14_test_alias_temporel.py` découpe trois artistes de part et d'autre de leur
changement de nom :

```python
# --- Contrôles : même découpage, mais sans changement de nom ---
```

Le contrôle est indispensable : découper une carrière en deux dégrade la
détection *par soi-même*. Sans lui, on attribuerait au changement de nom un
effet qui vient de la coupe.

## `15_` et `16_` — l'hypothèse web7

`15_test_ziak_7jaws.py` teste au niveau de l'artiste, en plusieurs variantes,
plus un **test inverse** (Ziak vu depuis web7) — une proximité doit être
symétrique.

`16_test_2025_web7.py` exploite une expérience naturelle. Sur un même album,
8 titres sont crédités à web7 et 15 ne le sont pas.

```python
# --- Expérience 2025 ---
# --- Contrôles de puissance ---
# --- Toutes années confondues (époque non contrôlée) ---
# --- Classement général, invités retirés ---
```

Le test de permutation compare la distance des 8 titres crédités au profil de
web7 à celle de 5 000 tirages de 8 titres parmi les 23 — **la taille du groupe
restant identique, l'effet de taille est neutralisé**.

Les contrôles de puissance sont ce qui rend le résultat lisible : on remplace
les titres crédités par du texte entièrement de web7, puis par un mélange à
parts égales, pour savoir ce que le test *aurait su voir*.

## `18_controle_featurings.py` — les invités faussent-ils le classement ?

Trois mesures, dont la troisième est la plus utile :

`textes_genius` construit **deux versions** de la requête Ziak — avec et sans
invités — depuis la même source. Une troisième variante rejoue la requête
LRFAF. Ce triptyque **sépare l'effet du nettoyage de celui du changement de
source** : sans lui, on ne saurait pas lequel des deux a bougé.

`titres_ou_ziak_invite` retrouve les titres du corpus contenant un couplet de
Ziak — la contamination qui pourrait créer un faux positif.

`simulation_contamination` injecte du texte étranger chez *tous* les candidats,
à taux croissant, et mesure quand la méthode lâche. C'est la seule façon de
borner un défaut qu'on ne peut pas corriger.

## `19_` et `20_` — le corpus nettoyé

`19_collecte_corpus.py` re-télécharge les pages Genius, 4 fils, sortie JSONL en
ajout seul **indexée par URL** : relancer reprend où l'on s'est arrêté. Sur une
collecte de six heures, ce n'est pas un luxe.

`20_corpus_sans_invites.py` reconstruit le corpus. Sa difficulté est de
distinguer un membre de groupe d'un invité :

```python
motif = ("répertoire" if part >= PART_MEMBRE else
         "récurrent" if part >= PART_RECURRENT and k >= TITRES_RECURRENT
         else None)
```

**Cette règle a une limite connue, et elle est documentée** : Freeman figure sur
6,7 % des titres d'IAM dont il est membre, Rohff sur 6,3 % de ceux de 113 dont
il est invité. Aucun seuil ne les sépare. Le réglage penche donc vers la
conservation — garder le couplet d'un invité ne fait que reproduire le défaut
de LRFAF, tandis que retirer celui d'un membre en fabriquerait un nouveau. Le
nettoyage obtenu (12 % des mots) est un **plancher**, pas une valeur exacte.

---

# 3. La restitution

## `07_figures.py`

Une fonction par figure, toutes lisant les CSV de `export/`, et un bloc final
qui les appelle :

```python
FIGURES = [("biais de taille", fig_biais_taille), ...]
```

Ce bloc est en **fin de fichier** pour une raison apprise à la dure : il s'était
retrouvé au milieu, avant la définition des quatre dernières fonctions, et le
script ne produisait silencieusement que cinq figures sur neuf.

## `article_contenu.py`

Le texte de l'article, partagé par les deux formats.

```python
CORPUS = {"brut": RESULT, "o1": RESULT / "sans_feats", "o2": RESULT / "sans_invites"}
```

`chiffres(k)` charge les résultats d'un corpus sous une forme directement
citable, et l'article appelle la fonction trois fois — ce qui lui permet de
comparer les trois versions de l'étude dans le même paragraphe.

**Aucun chiffre n'est écrit à la main dans la prose.** Tout est relu au moment
de la génération : rejouer une analyse met l'article à jour, et une valeur ne
peut pas se désynchroniser du résultat qu'elle cite.

## `12_article_pdf.py` et `17_article_docx.py`

Deux moteurs de rendu implémentant la même interface — `p`, `gap`, `tableau`,
`figure` — l'un vers reportlab, l'autre vers python-docx. Le texte étant
partagé, **les deux formats ne peuvent pas diverger**.

Le Word passe par des **styles nommés** plutôt que du formatage direct : changer
la police du style « Normal » se propage à tout le corps du texte. Les légendes
sont numérotées par des champs `SEQ`, donc renumérotées automatiquement à
l'insertion d'une figure.

---

# 4. Ce qui a été supprimé, et pourquoi

| Fichier | Motif |
|---|---|
| `02_stylometrie_ziak.ipynb` | devenu faux : écrit pour un corpus unique, il ignore les trois variantes, le nettoyage des featurings et l'hypothèse web7 |
| `images/*.png` (racine) | doublons des figures de `images/sans_invites/`, que l'article n'utilise pas |
| `export/15_0_discographie_web7.csv` | plus cité nulle part depuis que l'article a été resserré |
| `collecte_mikeysem.py` | entièrement couvert par `collecte_genius.py` |
| `~$*.docx` | fichier de verrouillage Word |

Tout reste récupérable dans l'historique git.
