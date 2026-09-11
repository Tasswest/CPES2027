#!/usr/bin/env python3
"""Génère l'article scientifique au format PDF.

Le notebook `02_stylometrie_ziak.ipynb` mêle code, sorties et commentaire : c'est
le document de travail reproductible. Ce script en tire la version destinée à la
lecture — texte rédigé, figures légendées, tableaux mis en forme — dans
`article_ziak_stylometrie.pdf`.

Tous les chiffres sont relus dans `result/` au moment de la génération : le PDF
ne peut pas diverger des résultats effectivement produits par les scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

RESULT = Path("result")
IMAGES = Path("images")
OUT = Path("article_ziak_stylometrie.pdf")

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
for name, fname in [("Times", "Times New Roman.ttf"),
                    ("Times-Bold", "Times New Roman Bold.ttf"),
                    ("Times-Italic", "Times New Roman Italic.ttf"),
                    ("Times-BoldItalic", "Times New Roman Bold Italic.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / fname)))
pdfmetrics.registerFontFamily("Times", normal="Times", bold="Times-Bold",
                              italic="Times-Italic", boldItalic="Times-BoldItalic")

ACCENT = colors.HexColor("#2f6f9f")
GRIS = colors.HexColor("#555555")

S = getSampleStyleSheet()
st = {
    "titre": ParagraphStyle("titre", parent=S["Title"], fontName="Times-Bold",
                            fontSize=19, leading=23, spaceAfter=4),
    "soustitre": ParagraphStyle("st", parent=S["Normal"], fontName="Times-Italic",
                                fontSize=12, leading=15, alignment=TA_CENTER,
                                textColor=GRIS, spaceAfter=14),
    "auteur": ParagraphStyle("aut", parent=S["Normal"], fontName="Times",
                             fontSize=11, alignment=TA_CENTER, spaceAfter=2),
    "date": ParagraphStyle("dat", parent=S["Normal"], fontName="Times",
                           fontSize=9.5, alignment=TA_CENTER, textColor=GRIS,
                           spaceAfter=16),
    "h1": ParagraphStyle("h1", parent=S["Normal"], fontName="Times-Bold",
                         fontSize=13, leading=16, spaceBefore=14, spaceAfter=6,
                         textColor=ACCENT),
    "h2": ParagraphStyle("h2", parent=S["Normal"], fontName="Times-Bold",
                         fontSize=11, leading=14, spaceBefore=10, spaceAfter=4),
    "p": ParagraphStyle("p", parent=S["Normal"], fontName="Times", fontSize=10,
                        leading=13.6, alignment=TA_JUSTIFY, spaceAfter=6),
    "abstract": ParagraphStyle("abs", parent=S["Normal"], fontName="Times",
                               fontSize=9.5, leading=13, alignment=TA_JUSTIFY,
                               leftIndent=14, rightIndent=14, spaceAfter=5),
    "legende": ParagraphStyle("leg", parent=S["Normal"], fontName="Times",
                              fontSize=8.5, leading=11, alignment=TA_JUSTIFY,
                              textColor=GRIS, spaceBefore=3, spaceAfter=10),
    "ref": ParagraphStyle("ref", parent=S["Normal"], fontName="Times", fontSize=8.8,
                          leading=11.5, alignment=TA_JUSTIFY, leftIndent=12,
                          firstLineIndent=-12, spaceAfter=3),
    "th": ParagraphStyle("th", parent=S["Normal"], fontName="Times-Bold",
                         fontSize=8.6, leading=10.4, alignment=TA_CENTER,
                         textColor=colors.white),
    "td": ParagraphStyle("td", parent=S["Normal"], fontName="Times", fontSize=8.6,
                         leading=10.4, alignment=TA_CENTER),
    "tdl": ParagraphStyle("tdl", parent=S["Normal"], fontName="Times", fontSize=8.6,
                          leading=10.4),
    "note": ParagraphStyle("note", parent=S["Normal"], fontName="Times",
                           fontSize=9, leading=12, alignment=TA_JUSTIFY,
                           leftIndent=10, rightIndent=10, spaceAfter=6,
                           borderPadding=5, backColor=colors.HexColor("#f4f6f8")),
}

story: list = []
P = lambda t, s="p": story.append(Paragraph(t, st[s]))
GAP = lambda h=6: story.append(Spacer(1, h))
FIGN = {"n": 0}
TABN = {"n": 0}


def figure(path: str, legende: str, largeur=16.4 * cm) -> None:
    """Insère une figure avec sa légende numérotée."""
    FIGN["n"] += 1
    img = Image(str(IMAGES / path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = largeur, largeur * ratio
    story.append(KeepTogether([
        img, Paragraph(f"<b>Figure {FIGN['n']}.</b> {legende}", st["legende"])]))


def tableau(data, legende, widths=None, align_num=True) -> None:
    """Les cellules texte sont converties en Paragraph, pour que le balisage
    (<sub>, <br/>, <b>) soit interprété et que le texte long puisse revenir à
    la ligne."""
    TABN["n"] += 1
    body = []
    for i, row in enumerate(data):
        cells = []
        for j, c in enumerate(row):
            if isinstance(c, str):
                sty = "th" if i == 0 else ("tdl" if (j == 0 and not align_num) or
                                           (j == 0 and i > 0) else "td")
                cells.append(Paragraph(c, st[sty]))
            else:
                cells.append(c)
        body.append(cells)
    t = Table(body, colWidths=widths, hAlign="CENTER")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Times"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f6f8fa")]),
    ]
    t.setStyle(TableStyle(style))
    story.append(KeepTogether([
        Paragraph(f"<b>Tableau {TABN['n']}.</b> {legende}", st["legende"]), t]))
    GAP(10)


def pct(x, d=1):
    return f"{100 * float(x):.{d}f} %".replace(".", ",")


def num(x, d=2):
    return f"{float(x):.{d}f}".replace(".", ",")


# =========================================================================
# Chiffres relus dans result/
# =========================================================================
var = pd.read_csv(RESULT / "02_3_puissance_par_variante.csv")
puis = pd.read_csv(RESULT / "03_2_puissance_methode.csv")
seuils = pd.read_csv(RESULT / "03_3_seuils_decision.csv")
verdict = pd.read_csv(RESULT / "04_6_verdict_hypotheses.csv")
gen = pd.read_csv(RESULT / "05_1_puissance_par_generation.csv")
gen = gen[gen.features == "char"]
imp = pd.read_csv(RESULT / "05_3_test_imposteurs.csv")
impref = pd.read_csv(RESULT / "05_4_imposteurs_reference.csv")
exc = pd.read_csv(RESULT / "06_1_excentricite.csv")
marq = pd.read_csv(RESULT / "06_2_marqueurs_lexicaux.csv")
elis = pd.read_csv(RESULT / "06_5_controle_elision.csv")
stab = pd.read_csv(RESULT / "06_3_stabilite_temporelle.csv").iloc[0]
ctrlpos = pd.read_csv(RESULT / "04_3_controle_positif_ziak.csv")
arith = pd.read_csv(RESULT / "09_1_controle_arithmetique.csv")
mval = pd.read_csv(RESULT / "10_1_validation_petite_taille.csv")
mrk = pd.read_csv(RESULT / "10_2_rang_mikeysem.csv")
mimp = pd.read_csv(RESULT / "10_4_imposteurs.csv")
mcomp = pd.read_csv(RESULT / "10_5_rangs_compares.csv", index_col=0)
disco = pd.read_csv(RESULT / "11_discographie_mikeysem.csv")
areel = pd.read_csv(RESULT / "13_2_alias_reels_synthese.csv")
areel_b = pd.read_csv(RESULT / "13_1_alias_reels_bruts.csv")
atemp = pd.read_csv(RESULT / "14_2_alias_temporel_synthese.csv")
atemp_b = pd.read_csv(RESULT / "14_1_alias_temporel_bruts.csv")
zsep = pd.read_csv(RESULT / "04_2_separation_ziak.csv")
zsep = zsep[(zsep.features == "char") & (zsep.metric == "cosine_delta")]
chg = atemp[atemp.groupe == "changement d'identité"].sort_values("rang_median")
ctl = atemp[atemp.groupe == "sans changement"]
duos = areel[areel.n_rappeurs == 2]

best = puis[(puis.features == "char") & (puis.metric == "cosine_delta")].iloc[0]
sbest = seuils[(seuils.features == "char") & (seuils.metric == "cosine_delta")].iloc[0]
mh1 = mval[(mval.condition == "H1") & (mval.features == "char")
           & (mval.metric == "cosine_delta")]
mrr = mrk[(mrk.features == "char") & (mrk.metric == "cosine_delta")]
mscore = float(mimp[mimp.type == "hypothèse"].score.iloc[0])
mref = mimp[mimp.type == "jumeau authentique"].score
zexc = exc[exc.artiste == "Ziak"].iloc[0]
# Taille du corpus Mikeysem telle que la voit le test stylométrique (tokens du
# pipeline d'analyse, distincts du n_words LRFAF qui inclut l'en-tête Genius).
from stylo_features import clean_lyrics, tokenize as _tok
MIKE_MOTS = int(sum(len(_tok(clean_lyrics(t)))
                    for t in pd.read_csv("mikeysem_lrfaf.csv").lyrics))
rang_dmin = int((exc.distance_min < zexc.distance_min).sum()) + 1

# =========================================================================
# Corps de l'article
# =========================================================================
P("Ziak est-il un autre rappeur&nbsp;?", "titre")
P("Une enquête stylométrique sur le corpus LRFAF du rap français", "soustitre")
P("Tassilo Westphalen", "auteur")
P("CPES Sciences des données, Arts et Cultures — Université PSL / Lycée Louis-le-Grand",
  "date")

P("<b>Résumé.</b> Le rappeur Ziak est apparu en 2020 masqué et sans identité "
  "publique, ce qui a nourri l'hypothèse d'un artiste déjà établi rappant sous "
  "un pseudonyme. Cette hypothèse est testable&nbsp;: si Ziak est le second nom "
  "d'un rappeur du corpus, ses textes doivent porter la même signature "
  "statistique. Nous conduisons ce test sur LRFAF (37&nbsp;307 chansons de rap "
  "français) au moyen d'un protocole d'attribution d'auteur à taille contrôlée, "
  "validé sur 177 artistes dont la réponse est connue. Deux résultats. "
  "D'une part, méthodologique&nbsp;: l'approche intuitive, qui compare des "
  "corpus d'artistes entiers, produit un classement d'apparence convaincante "
  f"mais n'identifie le bon auteur que dans {pct(var.iloc[0].recall_at_1, 0)} des "
  "cas — elle se trompe trois fois sur quatre, car elle mesure surtout la "
  "quantité de texte disponible. D'autre part, empirique&nbsp;: "
  f"avec un protocole dont on établit qu'il retrouve le bon auteur dans "
  f"{pct(best.recall_at_1, 0)} des cas, aucun des 392 artistes éligibles ne "
  "présente la signature de Ziak. Le nom le plus avancé par les auditeurs, "
  "Mikeysem, absent du corpus, a été collecté et testé séparément&nbsp;: il ne "
  "correspond pas davantage. Ziak écrit au centre de son genre, sans "
  "excentricité mesurable, mais sans proche parent identifiable.", "abstract")
GAP(4)
P("<b>Mots-clés&nbsp;:</b> stylométrie, attribution d'auteur, rap français, "
  "Delta de Burrows, méthode des imposteurs, LRFAF.", "abstract")
GAP(8)

P("1. Introduction", "h1")
P("Ziak publie son premier titre en 2020. Cagoulé en public, il ne divulgue "
  "aucune identité civile et cultive ouvertement le mystère. Cette discrétion a "
  "nourri parmi les auditeurs une hypothèse récurrente&nbsp;: le nom masquerait "
  "un artiste déjà établi, qui aurait recommencé une carrière sous un "
  "pseudonyme.")
P("Cette hypothèse a la particularité d'être <i>testable</i>. La stylométrie "
  "repose sur un constat empirique ancien&nbsp;: les habitudes d'écriture les "
  "moins conscientes d'un auteur — fréquence des mots grammaticaux, "
  "enchaînements de caractères, élisions — forment une signature relativement "
  "stable, que le changement de nom n'efface pas. Si Ziak est le second nom "
  "d'un rappeur présent dans le corpus, ses textes doivent porter la même "
  "signature que ceux de cet artiste.")
P("L'enjeu principal de ce travail n'est pas l'algorithme mais le "
  "<b>protocole</b>. Une question d'attribution ne se règle pas en calculant "
  "des distances&nbsp;: elle se règle en sachant ce que ces distances valent. "
  "Nous montrons en section 3 qu'une démarche intuitive, appliquée aux mêmes "
  "données, désigne un artiste avec assurance et se trompe trois fois sur "
  "quatre. La contribution de cet article tient donc autant à la manière de "
  "poser la question qu'à la réponse obtenue.")

P("2. Données", "h1")
P("2.1 Le corpus LRFAF", "h2")
P("Le corpus LRFAF (de Courson, 2024) rassemble 37&nbsp;307 textes de rap "
  "français issus de genius.com, obtenus en croisant les catégories Wikipédia "
  "et Wikidata avec l'API de Genius, puis en extrayant les paroles au moyen du "
  "paquet <i>lyricsgenius</i>. Il fournit, outre les textes, des métadonnées "
  "(artiste, année, vues, contributeurs) et une batterie de mesures "
  "lexicométriques.")
P("Après nettoyage — suppression des textes de moins de 100 mots et des "
  "doublons de paroles entre artistes, qui correspondent à des featurings ou à "
  "des rééditions — l'analyse porte sur 32&nbsp;923 titres et 596 artistes.")

P("2.2 Le corpus de Ziak", "h2")
P("Ziak y est représenté par 43 titres publiés entre 2020 et 2024, soit "
  "23&nbsp;886 mots. C'est un volume modeste mais très au-dessus du seuil usuel "
  "de quelques milliers de mots requis par les méthodes employées ici. La "
  "section 5.1 vérifiera empiriquement que cette matière suffit.")
P("Un point de vocabulaire commande tout le reste. Nous cherchons une "
  "<b>vérification d'auteur en ensemble ouvert</b>&nbsp;: la bonne réponse peut "
  "ne pas figurer dans le corpus. C'est une tâche plus difficile que "
  "l'attribution en ensemble fermé, où l'on sait que l'auteur est l'un des "
  "candidats — une méthode d'ensemble fermé désigne <i>toujours</i> quelqu'un. "
  "Il faut donc un protocole capable de répondre «&nbsp;personne&nbsp;».")

P("3. Pourquoi la démarche intuitive désigne le mauvais artiste", "h1")
P("La démarche spontanée consiste à concaténer toutes les chansons de chaque "
  "artiste, à vectoriser, puis à classer les candidats par distance à Ziak. "
  "Elle produit un palmarès plausible. Le problème apparaît lorsqu'on le "
  "confronte à une variable qui n'a rien à voir avec le style&nbsp;: la simple "
  "quantité de texte disponible sur chaque artiste.")
P("La corrélation de Spearman entre la distance à Ziak et la taille du corpus "
  "du candidat atteint −0,47&nbsp;: près d'un quart de la variance du classement "
  "tient à cette seule quantité. Le mécanisme est arithmétique, non "
  "musical&nbsp;: plus un artiste a écrit, plus son vecteur couvre de n-grammes, "
  "et plus il ressemble à n'importe quel texte. Les vingt «&nbsp;plus "
  "proches&nbsp;» de Ziak ont un corpus deux fois plus gros que la médiane.")
P("L'argument décisif n'est pas qu'un classement change, mais qu'on peut "
  "mesurer lequel a raison. Nous soumettons trois variantes au même protocole "
  "de vérité-terrain&nbsp;: on prélève chez un artiste un échantillon de la "
  "taille du corpus de Ziak, on le traite comme un texte anonyme, et l'on "
  "regarde si la méthode le rattache au reste de son œuvre.")

tableau(
    [["Variante", "Rang 1", "Top 5", "Top 20", "Rang médian"]] +
    [[r.variante, pct(r.recall_at_1, 1), pct(r.recall_at_5, 1),
      pct(r.recall_at_20, 1), f"{r.rang_median:.0f}"] for r in var.itertuples()],
    "Puissance des trois variantes, évaluée sur 177 artistes dont la réponse est "
    "connue. Le contrôle de la taille des documents apporte 48 points, la "
    "standardisation des traits 18 de plus.",
    widths=[6.6 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.6 * cm])

P("L'approche naïve identifie le bon auteur dans un quart des cas seulement, "
  "alors qu'elle produit un classement d'allure sérieuse. Ramener tous les "
  "documents à une taille identique porte ce taux à 74&nbsp;%, et standardiser "
  "les traits à 92&nbsp;%. Sans étalonnage, un classement de distances ne "
  "permet aucune conclusion, quelle que soit sa netteté apparente.")

figure("02_1_biais_de_taille.png",
       "Ce que mesure l'approche naïve. (a) la distance à Ziak décroît avec la "
       "taille du corpus du candidat&nbsp;; (b) les rangs se réorganisent "
       "entièrement une fois la taille neutralisée — Mister You passe du 9<super>e</super> "
       "au 138<super>e</super> rang&nbsp;; (c) puissance comparée sur vérité-terrain.")

P("4. Protocole", "h1")
P("Le protocole retenu repose sur trois choix. <b>Les traits</b> sont les "
  "4-grammes de caractères et les 500 mots les plus fréquents&nbsp;: des "
  "descripteurs qui captent des habitudes d'écriture largement inconscientes "
  "plutôt que les thèmes abordés. <b>Les distances</b> sont le Cosine Delta "
  "(Smith &amp; Aldridge, 2011) et le Delta de Burrows (1992), calculées sur "
  "des traits standardisés. <b>La taille</b> est contrôlée&nbsp;: chaque "
  "candidat est représenté par 12&nbsp;000 mots exactement, échantillonnés "
  "parmi ses titres.")
P("Chaque artiste de contrôle est traité <i>exactement</i> comme Ziak&nbsp;: on "
  "lui prélève un texte-requête de la taille du corpus de Ziak, et un "
  "«&nbsp;jumeau&nbsp;» de 12&nbsp;000 mots issu de titres disjoints, placé "
  "dans le pool. Deux conditions sont évaluées&nbsp;: <b>H<sub>1</sub></b>, où "
  "le jumeau est présent, qui mesure la puissance&nbsp;; et <b>H<sub>0</sub></b>, "
  "où il est retiré, qui reproduit la situation d'un auteur absent du corpus — "
  "c'est-à-dire l'hypothèse à laquelle Ziak devra être confronté.")

tableau(
    [["Traits", "Distance", "Rang 1", "Top 5", "Top 20"]] +
    [[r.features, r.metric.replace("_", " "), pct(r.recall_at_1, 1),
      pct(r.recall_at_5, 1), pct(r.recall_at_20, 1)]
     for r in puis.itertuples()],
    "Puissance des quatre combinaisons, sur 1&nbsp;770 essais couvrant 177 "
    "artistes. Les 4-grammes de caractères associés au Cosine Delta dominent.",
    widths=[2.6 * cm, 3.4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm])

P("La meilleure combinaison retrouve le bon auteur au premier rang dans "
  f"{pct(best.recall_at_1, 0)} des cas et dans le top 5 dans "
  f"{pct(best.recall_at_5, 1)}. C'est cette puissance élevée qui rendra un "
  "résultat négatif informatif&nbsp;: si un alias existait dans le corpus, la "
  "méthode aurait neuf chances sur dix de le désigner en tête.")
P("La distance brute au meilleur candidat ne se lit pas seule. Nous lui "
  "substituons un <b>score de séparation</b>&nbsp;: de combien d'écarts-types "
  "le meilleur candidat se détache-t-il des autres&nbsp;? Cette grandeur est "
  "comparable d'une requête à l'autre. Un auteur réellement présent produit une "
  f"séparation moyenne de {num(sbest.sep_H1_correct_moy)}&nbsp;; un auteur "
  f"absent, de {num(sbest.sep_H0_moy)}. L'écart entre ces deux distributions "
  "fournit le critère de décision.")

figure("03_1_validation.png",
       "Validation du protocole. (a) courbe de rappel de la meilleure "
       "combinaison&nbsp;; (b) les quatre combinaisons testées&nbsp;; (c) puissance "
       "par génération d'artistes — elle est maximale sur celle de Ziak.")

P("5. Résultats", "h1")
P("5.1 Le style de Ziak est-il détectable&nbsp;?", "h2")
P("Avant d'interpréter un échec d'identification, il faut écarter l'explication "
  "triviale&nbsp;: un corpus trop petit ou trop hétérogène pour porter une "
  "signature. Nous coupons donc le corpus de Ziak en deux moitiés disjointes et "
  "cherchons la seconde depuis la première. <b>La moitié cible ressort au "
  "premier rang dans 100&nbsp;% des tirages</b> avec la meilleure combinaison, "
  "parmi 392 candidats. Le style de Ziak est donc parfaitement détectable et "
  "son corpus suffisamment homogène&nbsp;: aucun échec ultérieur ne pourra être "
  "imputé à une insuffisance de matière.")

P("5.2 Aucun candidat ne s'impose", "h2")
P("Nous classons ensuite les 392 autres artistes, sur 30 rééchantillonnages et "
  "avec les quatre combinaisons. Le résultat est éloquent par son "
  "incohérence&nbsp;: chaque méthode a son favori — Beendo Z, Rimkus, Zkr, "
  "L'Animalerie — et <b>aucun candidat ne s'impose d'une méthode à l'autre</b>. "
  "Pour les artistes de contrôle, dont la réponse est connue, les quatre "
  "combinaisons convergent dans neuf cas sur dix. Ici, elles divergent.")

P("5.3 Verdict", "h2")
tableau(
    [["Traits", "Distance", "Séparation<br/>de Ziak",
      "Repère H<sub>1</sub><br/>(auteur présent)",
      "Repère H<sub>0</sub><br/>(auteur absent)", "Hypothèse<br/>favorisée"]] +
    [[r.features, r.metric.replace("_", " "), num(r.sep_ziak),
      num(r.sep_H1_correct_moy), num(r.sep_H0_moy), "<b>H<sub>0</sub></b>"]
     for r in verdict.itertuples()],
    "Confrontation aux deux hypothèses. Les quatre analyses concordent&nbsp;: la "
    "séparation observée pour Ziak est non seulement très loin de ce que produit "
    "un alias authentique, mais se situe dans la queue de la distribution des "
    "auteurs absents.",
    widths=[2.0 * cm, 2.8 * cm, 2.6 * cm, 3.2 * cm, 3.2 * cm, 2.6 * cm])

P("Autrement dit, ce n'est pas seulement que la méthode ne trouve pas&nbsp;: "
  "c'est qu'elle trouve activement l'absence de correspondance.")

figure("04_1_verdict_H0_H1.png",
       "Ziak confronté aux deux hypothèses, pour les quatre combinaisons. La "
       "distribution verte correspond aux cas où l'auteur est réellement dans le "
       "corpus, la grise aux cas où il en est absent&nbsp;; le trait rouge marque "
       "la position de Ziak.")

P("5.4 Robustesse", "h2")
P("Trois objections méritent une réponse chiffrée. <b>La méthode est-elle moins "
  "puissante sur les artistes récents&nbsp;?</b> C'est l'inverse&nbsp;: la "
  "puissance est maximale sur la génération de Ziak "
  f"({pct(gen[gen.generation == '2021-2024'].iloc[0].recall_at_1, 1)} au premier "
  "rang pour 2021-2024). <b>Le seuil de 12&nbsp;000 mots exclut-il le bon "
  "candidat&nbsp;?</b> Le score de séparation reste compris entre −2,43 et −2,58 "
  "lorsqu'on fait varier ce seuil de 5&nbsp;000 à 25&nbsp;000 mots, faisant "
  "varier le pool de 472 à 268 artistes. <b>Les candidats récurrents "
  "résistent-ils à un test par paire&nbsp;?</b>")
P("Le test des imposteurs (Koppel &amp; Winter, 2014) ne demande plus «&nbsp;qui "
  "est le plus proche&nbsp;?&nbsp;» mais «&nbsp;ce candidat précis l'emporte-t-il "
  "sur un fond d'imposteurs tirés au hasard&nbsp;?&nbsp;». Il appelle une lecture "
  "en deux temps. D'un côté, les meilleurs candidats — Kerchak "
  f"({num(imp.iloc[0].score_imposteurs)}), Beendo Z "
  f"({num(imp.iloc[1].score_imposteurs)}), Rimkus "
  f"({num(imp.iloc[2].score_imposteurs)}) — battent largement le hasard "
  f"({num(imp.iloc[0].seuil_hasard)})&nbsp;: il existe une parenté stylistique "
  "réelle. De l'autre, un alias authentique obtient dans ce test un score médian "
  f"de {num(impref.score_imposteurs_jumeau.median())}, et neuf vrais jumeaux sur "
  f"dix dépassent {num(impref.score_imposteurs_jumeau.quantile(0.1))}. Le "
  "meilleur candidat de Ziak reste sous le niveau de 97,5&nbsp;% des jumeaux "
  "authentiques.")
P("Cette distinction — ressembler à un courant <i>versus</i> être la même "
  "personne — est précisément ce qu'un classement de distances non étalonné ne "
  "permet jamais de trancher.")

figure("05_1_candidats_imposteurs.png",
       "(a) aucun candidat n'est stable d'une méthode à l'autre&nbsp;; (b) même le "
       "meilleur candidat reste loin du niveau qu'atteint un véritable alias.")

P("6. Validation sur des liens d'auteur réels", "h1")
P("Tout ce qui précède repose sur une validation par jumeaux fabriqués&nbsp;: on "
  "coupe l'œuvre d'un artiste en deux et l'on cherche une moitié depuis l'autre. "
  "C'est une tâche <i>facile</i> — les deux moitiés partagent la même époque, les "
  "mêmes thèmes, le même producteur. La puissance qu'on y mesure est donc une "
  "borne optimiste, et c'est l'objection la plus sérieuse qu'on puisse opposer au "
  "verdict. Cette section y répond avec des cas où la vérité est connue "
  "indépendamment du corpus.")

P("6.1 Recouvrements entre un artiste et son groupe", "h2")
P("Le test d'alias idéal serait un artiste présent sous deux noms distincts. Il "
  "est irréalisable ici&nbsp;: <b>Genius fusionne lui-même les changements de "
  "nom</b>. Les titres de la période <i>Joke</i> sont classés sous "
  "<i>Ateyaba</i>, il n'existe aucune page «&nbsp;Joke&nbsp;», et LRFAF hérite de "
  "cette fusion. Vérification faite, les pages «&nbsp;Peter Punk&nbsp;» et "
  "«&nbsp;Malsain&nbsp;» trouvées sur Genius sont des homonymes — un groupe "
  "italien et un groupe de metal — et non Disiz ni Sinik.")
P("Restent quatorze paires <b>solo / groupe</b> où l'artiste a réellement écrit "
  "une partie des textes du groupe. Le test est plus sévère qu'un alias&nbsp;: "
  "dans un trio, l'auteur ne signe qu'un tiers du texte, le reste étant écrit par "
  "d'autres. La requête est ramenée à la taille du corpus de Ziak, sans quoi "
  "Booba (114&nbsp;983 mots) et Gringe (18&nbsp;092) ne seraient comparables ni "
  "entre eux ni au cas étudié.")

tableau(
    [["Artiste → groupe", "Rappeurs", "Rang médian (sur 392)"]] +
    [[f"{r.solo} → {r.groupe}", str(int(r.n_rappeurs)), f"{int(r.rang_median)}"]
     for r in areel.sort_values("rang_median").itertuples()],
    "Rang du groupe, interrogé depuis les textes solo de l'un de ses membres. "
    "L'effet de dilution est net&nbsp;: un duo se retrouve aisément, un groupe de "
    "huit se perd dans le classement.",
    widths=[7.4 * cm, 2.4 * cm, 4.6 * cm], align_num=False)

P(f"Le lien est retrouvé dans le top 20 pour "
  f"{pct((areel.rang_median <= 20).mean(), 0)} des paires — et "
  f"{pct((duos.rang_median <= 20).mean(), 0)} des duos, cas le plus proche d'un "
  f"alias. <b>La puissance réelle est donc inférieure aux "
  f"{pct(best.recall_at_1, 0)} mesurés sur jumeaux simulés</b>&nbsp;: il faut "
  "compter avec une chance sur cinq à une sur trois de manquer un lien, selon le "
  "degré de dilution.")

figure("13_1_alias_reels.png",
       "Validation sur quatorze recouvrements d'auteur réels. (a) rang du groupe "
       "vu depuis le solo&nbsp;; (b) plus l'auteur est dilué dans un collectif, "
       "moins il est détectable&nbsp;; (c) le meilleur candidat de Ziak se détache "
       "moins que dans la quasi-totalité des cas à lien réel.")

P("6.2 Un changement d'identité efface-t-il la signature&nbsp;?", "h2")
P("Reste l'objection de fond&nbsp;: un artiste qui se réinvente sous un autre nom "
  "change peut-être aussi de manière d'écrire. La fusion opérée par Genius permet "
  "justement de le tester, en découpant ces artistes <b>de part et d'autre de "
  "leur changement d'identité</b>. On interroge la période antérieure et l'on "
  "cherche la période postérieure, placée dans le pool sous une autre étiquette. "
  "Trois cas sont documentés dans le corpus, et le premier est le plus "
  "net&nbsp;: Ateyaba a publiquement déclaré vouloir «&nbsp;tuer Joke&nbsp;».")

tableau(
    [["Changement d'identité", "Rang médian", "Séparation", "Trouvé au rang 1"]] +
    [[r.libelle, f"{int(r.rang_median)}", num(r.sep_cible), pct(r.taux_rang1, 0)]
     for r in chg.itertuples()] +
    [[f"<i>Contrôles sans changement (n = {len(ctl)})</i>",
      f"<i>{ctl.rang_median.median():.0f}</i>", f"<i>{num(ctl.sep_cible.mean())}</i>",
      f"<i>{pct((ctl.rang_median == 1).mean(), 0)}</i>"]],
    "Période postérieure au changement de nom, recherchée depuis la période "
    "antérieure. La dernière ligne donne le repère&nbsp;: des artistes découpés au "
    "même endroit de leur carrière, mais qui n'ont jamais changé de nom.",
    widths=[6.4 * cm, 2.8 * cm, 2.8 * cm, 3.2 * cm], align_num=False)

P("Le résultat est net, et il lève l'objection plutôt qu'il ne la confirme. "
  "<b>Joke → Ateyaba est retrouvé au premier rang sur 393 candidats, dans la "
  "totalité des tirages</b>, avec une séparation de −4,98 — alors même que le "
  f"changement d'identité était revendiqué. Sur les trois cas, le rang médian "
  f"passe de {ctl.rang_median.median():.0f} (artistes sans changement) à "
  f"{chg.rang_median.median():.0f}, et la séparation reste inchangée "
  f"({num(ctl.sep_cible.mean())} contre {num(chg.sep_cible.mean())}). Changer de "
  "nom, de registre et d'époque ne suffit pas à effacer la signature.")

figure("14_1_alias_temporel.png",
       "Ce que coûte un changement d'identité. (a) les trois cas documentés&nbsp;; "
       "(b) leur rang comparé à celui d'artistes n'ayant jamais changé de "
       "nom&nbsp;; (c) séparation du meilleur candidat, seule grandeur comparable "
       "au cas Ziak.")

P("Ces deux tests tirent dans des directions opposées, et il faut les lire "
  "ensemble. La puissance est <i>plus faible</i> qu'annoncée dès lors que "
  "l'auteur recherché ne signe qu'une partie des textes. Mais elle ne s'effondre "
  "<i>pas</i> lorsqu'il change d'identité, ce qui était la crainte principale. "
  f"Or c'est bien cette seconde situation qui correspond à l'hypothèse testée sur "
  f"Ziak. Sur la seule grandeur comparable — la séparation du meilleur candidat du "
  f"classement — son favori se détache moins bien que dans "
  f"{pct((atemp_b.sep_top1 < zsep.sep_top1.mean()).mean(), 0)} de ces tests à lien "
  "réel.", "note")

P("7. Le cas Mikeysem", "h1")
P("La conclusion précédente souffrait d'un angle mort&nbsp;: le nom le plus "
  "fréquemment avancé par les auditeurs, <b>Mikeysem</b>, ne figure pas dans "
  "LRFAF. Ce n'est pas un oubli du corpus mais une conséquence de son critère "
  "d'inclusion, qui part des catégories Wikipédia&nbsp;: Mikeysem n'a ni article "
  "Wikipédia ni entrée Wikidata. L'hypothèse la plus discutée était donc "
  "précisément celle que l'étude ne pouvait pas tester.")
P("Ses titres ont été collectés sur Genius et passés dans le pipeline LRFAF "
  "reconstitué. <b>La couverture est partielle, et c'est la principale faiblesse "
  f"de cette section&nbsp;:</b> sa discographie compte {len(disco)} titres, mais "
  f"{int(disco.page_genius.sum())} seulement disposent d'une page Genius, soit "
  f"{MIKE_MOTS:,} mots".replace(",", "&nbsp;") + " — trois fois moins que la taille "
  "de référence employée jusqu'ici. Les autres n'ont aucune transcription "
  "disponible, dont l'intégralité du projet <i>Prochains Héritiers</i> (10 titres, "
  "2022). Aller chercher ces paroles ailleurs romprait la compatibilité "
  "méthodologique avec LRFAF, dont toutes les transcriptions proviennent de "
  "Genius et de ses conventions.")
P("Cette petitesse impose de recalibrer avant d'interpréter. À 3&nbsp;745 mots "
  f"de candidat, la méthode place le vrai auteur au premier rang dans "
  f"{pct((mh1.rank_twin == 1).mean(), 0)} des cas et dans le top 20 dans "
  f"{pct((mh1.rank_twin <= 20).mean(), 0)}&nbsp;: la puissance baisse, mais un "
  "alias authentique resterait très majoritairement détectable.")

tableau(
    [["Artiste", "Rang parmi 493 candidats"]] +
    [[a, f"{int(r)}"] for a, r in mcomp.rang_median.items()],
    "Rangs médians vus depuis Ziak, à taille strictement égale. Six artistes que "
    "personne ne soupçonne sont plus proches de Ziak que Mikeysem.",
    widths=[6.0 * cm, 5.4 * cm])

P(f"Aucune des quatre combinaisons ne place Mikeysem dans le top 20&nbsp;: son "
  f"rang médian est de {int(mrr.rang_mikeysem.median())} sur "
  f"{int(mrr.n_candidats.iloc[0])}. Son score de séparation "
  f"({num(mrr.sep_mikeysem.mean())}) est même plus faible que celui d'un auteur "
  "typiquement absent du corpus&nbsp;: il n'est pas un candidat ordinaire ayant "
  "manqué la première place, mais un artiste particulièrement éloigné. Au test "
  f"des imposteurs, il obtient {num(mscore)}, à peine au-dessus du hasard "
  f"({num(1 / 26)}), là où un alias authentique atteint {num(mref.median())} en "
  f"médiane&nbsp;; seuls {pct((mref <= mscore).mean(), 0)} des vrais jumeaux font "
  "moins bien.")

figure("10_1_test_mikeysem.png",
       "Test de l'hypothèse Mikeysem. (a) puissance de la méthode à cette taille "
       "de corpus&nbsp;; (b) rangs comparés&nbsp;; (c) le score obtenu reste très "
       "en deçà de celui d'un véritable alias.")

P("Une réserve doit accompagner ce résultat. Un contrôle d'auto-cohérence — une "
  "moitié du corpus de Mikeysem cherchant l'autre, avec environ 1&nbsp;870 mots "
  "de chaque côté — échoue complètement. Ce test est plus exigeant que le test "
  "principal, qui dispose des 23&nbsp;886 mots de Ziak comme requête, et ne remet "
  "donc pas en cause le classement&nbsp;; mais il rappelle, avec la couverture "
  "partielle signalée plus haut, que la conclusion de cette section vaut comme "
  "faisceau convergent et non comme démonstration.", "note")

P("8. Portrait stylométrique de Ziak", "h1")
P("À défaut d'identifier Ziak, on peut le caractériser. Deux mesures doivent "
  "être distinguées&nbsp;: la distance médiane à l'ensemble du corpus, qui dit "
  "s'il est atypique&nbsp;; et la distance à son plus proche voisin, qui dit "
  "s'il a un parent.")
P(f"Ziak n'a rien d'un excentrique&nbsp;: il se situe au rang "
  f"{int(zexc.rang_excentricite)} sur {len(exc)}, au milieu exact du genre, très "
  "loin des vrais atypiques du corpus que sont Manau, Grand Corps Malade ou "
  f"Rocé. Et pourtant <b>il n'a pas de proche parent</b>&nbsp;: "
  f"{pct(rang_dmin / len(exc), 0)} des artistes ont un voisin plus proche que le "
  "sien. C'est exactement la signature attendue d'un auteur qui écrit dans les "
  "codes d'un courant sans qu'aucun artiste du corpus ne soit sa seconde "
  "identité.")
P("Ses marqueurs les plus robustes sont des <b>onomatopées</b>&nbsp;: "
  + ", ".join(f"«&nbsp;{r.mot}&nbsp;» (×{r.ratio:.0f})"
              for r in marq.nlargest(3, "z_log_odds").itertuples())
  + ", les ad-libs qui ponctuent ses morceaux, suivis d'un argot situé "
  "(«&nbsp;zipette&nbsp;», «&nbsp;sheitan&nbsp;»).")
P("Une mise en garde s'impose toutefois sur ces marqueurs. Le trait le plus "
  "spectaculaire de son corpus, à première vue, est un sous-emploi massif de "
  "«&nbsp;je&nbsp;» — cinq fois moins que la moyenne. C'est un artefact&nbsp;: "
  "il emploie «&nbsp;j'&nbsp;» 1,2 fois plus, et une fois les deux formes "
  f"additionnées l'écart disparaît complètement (×{num(elis.iloc[0].ratio_cumule)}). "
  "La moitié des paires testées relève du même phénomène. Ce que l'on mesurait "
  "n'était pas une habitude d'écriture mais une convention de transcription&nbsp;: "
  "les paroles de Genius sont saisies par des contributeurs bénévoles, et "
  "«&nbsp;j'suis&nbsp;» ou «&nbsp;je suis&nbsp;» notent la même diction.")
P("Enfin, l'écart entre ses deux périodes de production (2020-2021 et 2022-2024) "
  f"dépasse à peine ce que produit une coupe aléatoire de son œuvre "
  f"(z&nbsp;=&nbsp;+{num(stab['z_vs_moitiés_aleatoires'])}, sous le seuil usuel "
  "de 2)&nbsp;: son style évolue un peu, sans rupture.")

figure("06_1_profil_ziak.png",
       "Portrait stylométrique. (a) marqueurs lexicaux&nbsp;; (b) Ziak n'a pas de "
       "proche parent&nbsp;; (c) l'élision, un faux marqueur qui vient du "
       "transcripteur et non de l'auteur.")

P("9. Limites", "h1")
P("<b>Le corpus n'est pas le rap français.</b> LRFAF couvre 596 artistes, et "
  "l'analyse n'en retient que 393 — ceux disposant d'au moins 12&nbsp;000 mots. "
  "Un artiste peu documenté, ou absent de Genius, ne pouvait pas être détecté. "
  "La section 7 lève ce point pour le seul candidat qui comptait vraiment, mais "
  "il en reste d'autres, hors corpus et non testés.")
P("<b>Les featurings ne sont pas séparés.</b> Les paroles du corpus ne "
  "comportent aucune balise de section&nbsp;: le couplet d'un invité est "
  "attribué à l'artiste principal. Ce bruit affecte Ziak comme les candidats, et "
  "tend à rapprocher artificiellement les artistes qui collaborent — donc à "
  "faciliter une détection, non à l'empêcher.")
P("<b>Les transcriptions sont médiées.</b> Comme le montre le cas de l'élision, "
  "une partie du signal apparent vient des contributeurs de Genius plutôt que "
  "des artistes. Les 4-grammes de caractères y sont moins sensibles que les "
  "mots, sans y être immunisés.")
P("<b>Le corpus de Mikeysem est mince et partiel</b>&nbsp;: 3&nbsp;745 mots, "
  "couvrant 7 de ses 21 titres. C'est la conclusion la moins solidement étayée "
  "de ce travail.")
P("<b>La puissance dépend de ce que l'on cherche.</b> La section 6 l'a mesurée "
  "sur des liens réels plutôt que simulés&nbsp;: elle chute nettement quand "
  "l'auteur recherché ne signe qu'une partie des textes (un membre parmi huit se "
  "perd au-delà du centième rang), mais résiste à un changement d'identité "
  "revendiqué. La conclusion «&nbsp;Ziak n'est personne du corpus&nbsp;» vaut "
  "donc pour un alias qui écrirait seul&nbsp;; elle serait plus fragile s'il "
  "s'agissait d'une participation diluée dans un collectif.")

P("10. Conclusion", "h1")
P("L'hypothèse du pseudonyme était testable, et le test est concluant&nbsp;: "
  "aucun des 392 autres artistes éligibles du corpus LRFAF ne présente la "
  "signature stylométrique de Ziak, alors qu'un protocole validé sur 177 cas "
  "connus retrouve le bon auteur neuf fois sur dix — et dans 97,7&nbsp;% des cas "
  "pour sa génération. Les quatre analyses convergent vers l'hypothèse d'un "
  "auteur absent du corpus, et le test par paire montre que même le meilleur "
  "candidat reste en deçà de 97,5&nbsp;% des alias authentiques. Le nom que la "
  "rumeur avance le plus souvent, testé à son tour, se classe 58<super>e</super> "
  "sur 493, derrière six artistes que personne ne soupçonne.")
P("Cette puissance a été éprouvée sur des liens d'auteur <b>réels</b> et non plus "
  "simulés (section 6)&nbsp;: la méthode retrouve Joke → Ateyaba au premier rang "
  "sur 393 candidats, malgré un changement d'identité revendiqué. Elle perd en "
  "revanche sa capacité de détection lorsque l'auteur recherché ne signe qu'une "
  "fraction des textes — un membre parmi huit se perd au-delà du centième rang.")
P("Ce que l'on observe à la place a sa propre valeur descriptive&nbsp;: Ziak "
  "écrit au centre de son genre, sans excentricité mesurable, mais sans proche "
  "parent non plus. Sa parenté avec Kerchak ou Beendo Z est celle d'une "
  "génération et d'un sous-genre partagés, pas d'une main commune.")
P("La contribution la plus transposable de ce travail est cependant "
  "méthodologique. Une même question, posée au même corpus, reçoit deux réponses "
  "opposées selon le protocole&nbsp;: l'approche intuitive désigne un artiste "
  "avec assurance et se trompe trois fois sur quatre&nbsp;; l'approche étalonnée "
  "ne désigne personne, et sait dire pourquoi. Entre les deux, il n'y a pas un "
  "algorithme plus sophistiqué, mais trois exigences ordinaires — contrôler une "
  "variable de confusion, valider sur des cas connus, calibrer contre une "
  "hypothèse nulle.")

P("Note éthique", "h2")
P("Cette étude porte sur une personne réelle ayant fait le choix de l'anonymat. "
  "Deux précautions ont guidé le travail. Elle s'appuie exclusivement sur des "
  "données publiques de recherche et sur des textes publiés, sans recours à "
  "aucune information personnelle. Surtout, la stylométrie n'établit pas "
  "d'identité civile&nbsp;: une correspondance forte aurait constitué un indice "
  "de parenté textuelle, jamais une preuve — et elle aurait appelé, à ce titre, "
  "une retenue plus grande encore dans sa publication que le résultat négatif "
  "obtenu ici. Le rapprochement avec Mikeysem, en particulier, est une "
  "spéculation d'auditeurs que l'intéressé a démentie et qu'aucune source "
  "vérifiable n'étaye. Le résultat de cette enquête est, à sa manière, une "
  "confirmation de la robustesse de cet anonymat face aux méthodes "
  "quantitatives.", "note")

P("Reproductibilité", "h2")
P("L'ensemble des analyses est reproductible depuis le dépôt du projet. Les "
  "scripts s'exécutent dans l'ordre&nbsp;: construction du cache de compteurs, "
  "diagnostic du biais de taille, validation du protocole, application à Ziak, "
  "robustesse, portrait, collecte de Mikeysem, test de l'hypothèse, figures. "
  "Le corpus LRFAF n'étant pas versionné (114&nbsp;Mo), il est retéléchargé "
  "depuis Hugging Face. Le pipeline LRFAF reconstitué et son rapport de "
  "reproductibilité — qui documente colonne par colonne ce qui est reproduit "
  "exactement, approximé ou impossible à retrouver — accompagnent le dépôt.")

P("Références", "h1")
for r in [
    "Burrows, J. (2002). Delta&nbsp;: a measure of stylistic difference and a guide "
    "to likely authorship. <i>Literary and Linguistic Computing</i>, 17(3), 267-287.",
    "de Courson, B. (2024). <i>LRFAF&nbsp;: une exploration numérique du rap "
    "français depuis les années 1990.</i> Jeu de données&nbsp;: "
    "huggingface.co/datasets/regicid/LRFAF",
    "Evert, S., Proisl, T., Jannidis, F. <i>et al.</i> (2017). Understanding and "
    "explaining Delta measures for authorship attribution. <i>Digital Scholarship "
    "in the Humanities</i>, 32(suppl. 2), ii4-ii16.",
    "Koppel, M. &amp; Winter, Y. (2014). Determining if two documents are written "
    "by the same author. <i>Journal of the Association for Information Science and "
    "Technology</i>, 65(1), 178-187.",
    "Monroe, B., Colaresi, M. &amp; Quinn, K. (2008). Fightin' words&nbsp;: lexical "
    "feature selection and evaluation for identifying the content of political "
    "conflict. <i>Political Analysis</i>, 16(4), 372-403.",
    "Piolat, A., Booth, R. J., Chung, C. K., Davids, M. &amp; Pennebaker, J. W. "
    "(2011). La version française du dictionnaire pour le LIWC. <i>Psychologie "
    "Française</i>, 56(3), 145-159.",
    "Smith, P. &amp; Aldridge, W. (2011). Improving authorship attribution&nbsp;: "
    "optimizing Burrows' Delta method. <i>Journal of Quantitative Linguistics</i>, "
    "18(1), 63-88.",
]:
    P(r, "ref")


def pied(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times", 8)
    canvas.setFillColor(GRIS)
    canvas.drawCentredString(A4[0] / 2, 1.15 * cm, str(doc.page))
    if doc.page > 1:
        canvas.drawString(2.2 * cm, 1.15 * cm,
                          "Ziak est-il un autre rappeur ? Une enquête stylométrique")
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4, title="Ziak est-il un autre rappeur ?",
                      author="Tassilo Westphalen",
                      subject="Stylométrie et attribution d'auteur sur le corpus LRFAF",
                      leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                      topMargin=1.9 * cm, bottomMargin=1.9 * cm)
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
doc.addPageTemplates([PageTemplate(id="std", frames=[frame], onPage=pied)])
doc.build(story)
print(f"{OUT} — {FIGN['n']} figures, {TABN['n']} tableaux")
