#!/usr/bin/env python3
"""Pipeline LRFAF reconstitué, pour produire des lignes compatibles avec le corpus.

Le code de construction du corpus LRFAF n'a jamais été publié : le dépôt
`regicid/genius_french_rap_corpus` ne contient que les listes d'artistes et les
textes bruts, et `article/rap.Rmd` (sur le dépôt Hugging Face) n'est que le code
d'*analyse* de l'article. Ce module reconstitue le pipeline de production à
partir de trois sources :

1. la description en prose de l'article (lexiques et modèles employés) ;
2. le code source de `lyricsgenius`, dont l'article dit qu'il a servi à la
   collecte, et dont on retrouve la signature exacte dans les paroles publiées ;
3. la rétro-ingénierie des relations arithmétiques du corpus publié.

Chaque fonction indique son statut de reproductibilité. Voir `RAPPORT_LRFAF.md`
pour le détail colonne par colonne et les validations chiffrées.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Ordre exact des colonnes du corpus LRFAF (corpus.csv).
LRFAF_COLUMNS = [
    "artist", "title", "year", "lyrics", "pageviews", "contributors", "url",
    "topic", "topic_clean", "ranking", "n_je", "n_profanity", "n_verlan",
    "n_french_words", "means_word_length", "n_unique_words", "n_words",
    "pageviews_corrected", "n_non_french_words", "n_argot", "n_onomatopee",
    "n_negative", "n_positive", "birthdate_artist", "age_artist",
    "pageview_mean", "pageviews_2", "n_sexe", "hate", "sexism", "n_lines",
    "sentiment2", "born_in_france",
]


# --------------------------------------------------------------------------
# 1. Nettoyage des paroles — REPRODUIT EXACTEMENT
# --------------------------------------------------------------------------
# Signature de `lyricsgenius` avec remove_section_headers=True :
#     lyrics = re.sub(r'(\[.*?\])*', '', lyrics)
#     lyrics = re.sub('\n{2}', '\n', lyrics)
# Le second appel remplace *exactement deux* sauts de ligne par un seul (et non
# les séquences de trois ou plus) : c'est ce détail qui reproduit à l'identique
# la ponctuation des paroles publiées. Validé sur 2 455 textes bruts d'archive :
# 97,7 % correspondent exactement ou en préfixe.

def lg_clean(raw: str) -> str:
    """Nettoyage `lyricsgenius`, appliqué au texte scrapé tel quel."""
    t = re.sub(r"(\[.*?\])*", "", raw)
    return re.sub("\n{2}", "\n", t)


def strip_genius_header(text: str) -> str:
    """Retire l'en-tête « N ContributorsTitre Lyrics » du site Genius.

    Cet en-tête fait partie du bloc scrapé mais pas des paroles publiées dans
    LRFAF. Point important : les compteurs du corpus ont été calculés *avant*
    ce retrait (l'en-tête pèse 3 à 4 tokens, ce qui explique l'écart médian
    observé entre `n_words` et les paroles stockées).
    """
    return re.sub(r"^.*?Lyrics", "", text, count=1, flags=re.S)


# --------------------------------------------------------------------------
# 2. Tokenisation — REPRODUITE
# --------------------------------------------------------------------------
# L'article décrit la fonction R `get_complexity`, qui remplace les apostrophes
# par des espaces avant d'appeler `tokenizers::tokenize_words` :
#     words = tokenize_words(str_replace_all(text, "'|’", " "))
# Équivalent Python retenu : minuscules, apostrophes -> espace, puis \w+.
# Validation : sur les textes où `n_words` est reproduit exactement,
# `n_unique_words` l'est aussi dans 99,9 % des cas — les deux colonnes reposent
# donc bien sur cette tokenisation.

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", re.sub(r"['’]", " ", text).lower())


def n_words(text: str) -> int:
    return len(tokenize(text))


def n_unique_words(text: str) -> int:
    return len(set(tokenize(text)))


def means_word_length(text: str) -> float:
    """Longueur moyenne des mots (`get_complexity`).

    Dans le code R, le terme `sum(str_count(words, "'|’"))` est toujours nul :
    les apostrophes ont déjà été remplacées par des espaces avant tokenisation.
    La formule se réduit donc à la moyenne des longueurs.
    """
    w = tokenize(text)
    return float(sum(len(x) for x in w) / len(w)) if w else np.nan


def n_lines(text: str) -> int:
    """Nombre de lignes — APPROXIMATION (86,5 % d'exactitude).

    La définition exacte n'est pas documentée. Parmi les variantes testées, le
    nombre de lignes contenant au moins deux mots est la plus proche du corpus
    publié ; les lignes non vides seules ne donnent que 68 %.
    """
    return sum(1 for line in text.split("\n") if len(re.findall(r"\w+", line)) >= 2)


def n_je(text: str) -> int:
    """Occurrences de la première personne : « je » et son élision « j' ».

    L'apostrophe étant convertie en espace, « j'suis » produit le token « j ».
    Reproduit exactement dans 79,9 % des cas (le reste s'explique par les
    paroles modifiées sur Genius depuis la collecte).
    """
    t = tokenize(text)
    return t.count("je") + t.count("j")


# --------------------------------------------------------------------------
# 3. Comptages lexicaux
# --------------------------------------------------------------------------

def count_lexicon(text: str, lexicon: set[str]) -> int:
    """Nombre de tokens appartenant au lexique."""
    return sum(1 for w in tokenize(text) if w in lexicon)


def load_lexicons(lex_dir: Path) -> dict[str, set[str]]:
    """Charge les lexiques disponibles.

    - `french` : Morphalou 3.1 (ATILF), le lexique nommé par l'article.
      Corrélation 0,988 avec `n_french_words` : c'est bien la bonne ressource,
      mais la version employée en 2024 diffère légèrement de celle-ci.
    - `verlan`, `argot`, `sexe` : catégories du Wiktionnaire citées en note de
      l'article. Leur contenu a évolué depuis : APPROXIMATIONS.
    - `profanity` : liste `MauriceButler/badwords` citée par l'article, plus
      « fuck », « nigga » et « bitch ». Cette liste est **anglaise** (issue du
      projet Google « what do you love »), alors que l'article la présente comme
      fondée sur le Wiktionnaire : APPROXIMATION.
    """
    lex: dict[str, set[str]] = {}
    files = {
        "french": "morphalou_forms.json",
        "verlan": "verlan.json",
        "argot": "argot.json",
        "sexe": "sexualite.json",
        "profanity": "profanity_en.json",
    }
    for key, fname in files.items():
        path = lex_dir / fname
        if not path.exists():
            continue
        words = json.load(open(path, encoding="utf-8"))
        norm = {re.sub(r"['’]", " ", w.lower()).strip() for w in words}
        # Seules les entrées d'un seul token peuvent être comptées mot à mot.
        lex[key] = {w for w in norm if w and " " not in w}
    if "profanity" in lex:
        lex["profanity"] |= {"fuck", "nigga", "bitch"}
    return lex


# --------------------------------------------------------------------------
# 4. Popularité — REPRODUITE EXACTEMENT
# --------------------------------------------------------------------------
# Relations retrouvées par rétro-ingénierie sur le corpus publié :
#     pageviews_2         = log(pageviews / pageview_mean[year])
#     pageviews_corrected = log(pageviews + 10) + f(year)
# La seconde est exacte au bruit machine près (écart-type intra-année de 1e-14).
# `pageview_mean` et `f(year)` sont des constantes annuelles estimées sur
# l'ensemble du corpus : on les relit directement dans LRFAF plutôt que de
# tenter de réestimer le lissage, dont la forme n'est pas documentée.

def year_constants(corpus: pd.DataFrame) -> pd.DataFrame:
    """Extrait `pageview_mean` et le décalage annuel `f(year)` du corpus."""
    d = corpus.dropna(subset=["year", "pageviews_corrected"]).copy()
    d["f_year"] = d["pageviews_corrected"] - np.log(d["pageviews"] + 10)
    out = d.groupby("year").agg(
        pageview_mean=("pageview_mean", "first"),
        f_year=("f_year", "median"),
        n=("f_year", "size"),
    )
    return out


def fallback_pageview_mean(corpus: pd.DataFrame) -> float:
    """Valeur de `pageview_mean` employée quand l'année est inconnue.

    Les 3 611 titres sans année du corpus partagent tous la même valeur
    (0.80089547) : une moyenne de repli, distincte des moyennes annuelles.
    """
    vals = corpus.loc[corpus["year"].isna(), "pageview_mean"].dropna().unique()
    return float(vals[0]) if len(vals) else np.nan


def pageviews_corrected(pageviews: float, year, consts: pd.DataFrame) -> float:
    """Vaut NA lorsque l'année est inconnue, comme dans le corpus original."""
    if pd.isna(year) or year not in consts.index:
        return np.nan
    return float(np.log(pageviews + 10) + consts.loc[year, "f_year"])


def pageviews_2(pageviews: float, year, consts: pd.DataFrame,
                fallback: float = np.nan) -> float:
    """log(vues / moyenne de l'année), et 0 quand les vues sont nulles.

    Genius ne publie le compteur qu'au-delà de 5 000 vues : LRFAF code cette
    censure par `pageviews = 0`, et pose alors `pageviews_2 = 0` plutôt qu'une
    valeur manquante (26 754 titres du corpus). Quand l'année est inconnue, la
    moyenne de repli est utilisée et la colonne reste calculée.
    """
    if pageviews <= 0:
        return 0.0
    pvm = (consts.loc[year, "pageview_mean"]
           if (not pd.isna(year) and year in consts.index) else fallback)
    if not pvm or pd.isna(pvm):
        return np.nan
    return float(np.log(pageviews / pvm))


# --------------------------------------------------------------------------
# 5. Modèles de discours — REPRODUCTIBLES (modèles publics identifiés)
# --------------------------------------------------------------------------
# L'article nomme les deux modèles et précise qu'ils ont été appliqués ligne à
# ligne, chacun renvoyant une probabilité par ligne :
#   hate   : Hate-speech-CNERG/dehatebert-mono-french
#   sexism : annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal
# La version des poids employée en 2024 n'est pas épinglée, et l'article ne
# précise pas comment les probabilités par ligne sont agrégées au niveau du
# titre : la moyenne simple est l'hypothèse retenue ici (l'agrégation par
# artiste, elle, est documentée comme une moyenne pondérée par `n_lines`).

HATE_MODEL = "Hate-speech-CNERG/dehatebert-mono-french"
SEXISM_MODEL = "annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal"


def score_lines(texts: list[str], model_name: str, positive_index: int = 1
                ) -> list[float]:
    """Probabilité de la classe positive pour chaque ligne."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            enc = tok(batch, return_tensors="pt", truncation=True,
                      max_length=128, padding=True)
            probs = torch.softmax(model(**enc).logits, dim=-1)
            out += probs[:, positive_index].tolist()
    return out


def song_lines(text: str) -> list[str]:
    """Lignes soumises aux modèles (mêmes lignes que celles comptées)."""
    return [l for l in text.split("\n") if len(re.findall(r"\w+", l)) >= 2]
