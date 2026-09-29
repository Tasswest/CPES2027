#!/usr/bin/env python3
"""Schéma de la chaîne de traitement, destiné à l'impression.

Produit `images/21_1_pipeline.png`, puis `pipeline_traitement.pdf` : une page
A4 qui tient seule, à remettre sur papier.

Le schéma est dessiné plutôt que rendu depuis Mermaid, pour trois raisons :
il s'imprime en vectoriel-équivalent (300 dpi), il reste lisible en noir et
blanc — un document remis sur papier est souvent photocopié — et il porte les
chiffres réels du corpus, relus dans `export/`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

IMAGES_DIR = Path("images")
RESULT_DIR = Path("export")
IMAGES_DIR.mkdir(exist_ok=True)

mpl.rcParams.update({"font.size": 8.6, "font.family": "serif",
                     "savefig.dpi": 300, "figure.dpi": 300})

BLEU = "#2f6f9f"
GRIS_T = "#44506b"       # texte
GRIS_C = "#9aa4b2"       # contours secondaires
CREME = "#fdf6e7"
BLEU_P = "#eef3f8"
VERT_P = "#eef5f1"
VERT = "#3d8b6d"


def chiffres() -> dict:
    """Les nombres cités par le schéma, relus dans les résultats."""
    net = pd.read_csv(RESULT_DIR / "20_1_nettoyage_resume.csv").iloc[0]
    sens = pd.read_csv(RESULT_DIR / "05_2_sensibilite_seuil.csv")
    var = pd.read_csv(RESULT_DIR / "02_3_puissance_par_variante.csv")
    puis = pd.read_csv(RESULT_DIR / "03_2_puissance_methode.csv")
    best = puis[(puis.features == "char") & (puis.metric == "cosine_delta")].iloc[0]
    return {
        "titres": int(net.titres_corpus),
        "recol": int(net.titres_recollectes),
        "part_retiree": float(net.part_retiree),
        "ecartes": int(net.option1_titres_ecartes),
        "candidats": int(sens[sens.iloc[:, 0] == 12_000].iloc[0, 1]),
        "naive": float(var.recall_at_1.iloc[0]),
        "rappel": float(best.recall_at_1),
    }


# Repères de mise en page. L'axe va de 0 à 1 sur 9,9 pouces : une ligne de
# corps (7,7 pt, interligne 1,45) occupe environ 0,016 unité, un titre 0,014.
# Les hauteurs de boîte sont calculées là-dessus, sans quoi le texte déborde.
LIGNE, TITRE, MARGE = 0.0165, 0.014, 0.011


def hauteur(n_lignes: int) -> float:
    return MARGE + TITRE + 0.006 + n_lignes * LIGNE + MARGE


def boite(ax, x, y, w, corps, titre, fond, bord):
    h = hauteur(len(corps.split("\n")) if corps else 0)
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.0, edgecolor=bord, facecolor=fond, zorder=2))
    ax.text(x + w / 2, y + h - MARGE, titre, ha="center", va="top",
            fontsize=8.8, color=GRIS_T, fontweight="bold", zorder=3)
    if corps:
        ax.text(x + w / 2, y + h - MARGE - TITRE - 0.006, corps,
                ha="center", va="top", fontsize=7.5, color=GRIS_T,
                linespacing=1.45, zorder=3)
    return h


def fleche(ax, p1, p2, couleur=None, trait="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=10, linewidth=0.9,
        linestyle=trait, color=couleur or GRIS_C, zorder=1,
        shrinkA=0, shrinkB=0))


def dessine(c: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 9.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    G, L, D = 0.030, 0.440, 0.530        # colonne gauche, largeur, colonne droite
    PLEIN = 0.940                        # largeur pleine
    MIL = 0.500                          # couloir central, laissé libre


    # --- sources ---------------------------------------------------------
    y = 0.890
    hs = boite(ax, G, y, L, "corpus publié, paroles sans balises",
               "Hugging Face — jeu LRFAF", BLEU_P, BLEU)
    boite(ax, D, y, L, "paroles balisées, métadonnées",
          "genius.com — pages publiques", BLEU_P, BLEU)
    bas_src = y

    # --- 1 et 2 : récolte -------------------------------------------------
    y = 0.790
    h1 = boite(ax, G, y, L,
               f"corpus.csv — {c['titres']:,} titres, 596 artistes".replace(",", " "),
               "1.  Corpus", "white", GRIS_C)
    boite(ax, D, y, L, "Mikeysem, web7, Ziak\ncollecte_genius.py",
          "2.  Artistes hors corpus", "white", GRIS_C)
    fleche(ax, (G + L / 2, bas_src), (G + L / 2, y + h1))
    fleche(ax, (D + L / 2, bas_src), (D + L / 2, y + h1))
    bas_12 = y

    # --- 3 : prétraitement ------------------------------------------------
    y = 0.672
    h3 = boite(ax, G, y, PLEIN,
               "nettoyage (accents et apostrophes conservés)  →  tokenisation "
               "(j'ai → j' + ai)\n"
               "filtres (≥ 100 mots, doublons retirés)  →  comptes par chanson\n"
               f"{c['candidats']} candidats éligibles (≥ 12 000 mots)",
               "3.  Prétraitement  ·  stylo_features.py", BLEU_P, BLEU)
    fleche(ax, (G + L / 2, bas_12), (G + L / 2, y + h3))
    fleche(ax, (D + L / 2, bas_12), (D + L / 2, y + h3))
    bas_3 = y

    # --- 4 et 5 -----------------------------------------------------------
    y = 0.542
    h4 = boite(ax, G, y, L,
               "13 scripts sur le corpus publié\nvalider → tester → décrire\n"
               "résultats en CSV",
               "4.  Analyses", "white", GRIS_C)
    boite(ax, D, y, L,
          f"{c['recol']:,} pages re-téléchargées".replace(",", " ")
          + "\n19_collecte_corpus.py\n≈ 6 h, reprise automatique",
          "5.  Re-collecte des balises", CREME, "#d9a441")
    fleche(ax, (G + L / 2, bas_3), (G + L / 2, y + h4))
    fleche(ax, (D + L / 2, bas_3), (D + L / 2, y + h4))
    bas_45 = y

    # --- 6 : corpus propres ----------------------------------------------
    y = 0.412
    h6 = boite(ax, D, y, L,
               f"sans_invites — strophes retirées ({100*c['part_retiree']:.0f} % des mots)\n"
               f"sans_feats — {c['ecartes']:,} titres écartés".replace(",", " ")
               + "\n20_corpus_sans_invites.py",
               "6.  Deux corpus propres", VERT_P, VERT)
    fleche(ax, (D + L / 2, bas_45), (D + L / 2, y + h6))
    bas_6 = y

    # --- 7 : étude rejouée ------------------------------------------------
    y = 0.292
    h7 = boite(ax, D, y, L,
               "les mêmes scripts, sur chaque variante\ncache et résultats séparés",
               "7.  Étude rejouée", VERT_P, VERT)
    fleche(ax, (D + L / 2, bas_6), (D + L / 2, y + h7))

    # Retour par le couloir central : chaque corpus repasse par le moteur.
    milieu = y + h7 / 2
    fleche(ax, (D, milieu), (MIL, milieu), couleur=VERT, trait=(0, (3, 2)))
    fleche(ax, (MIL, milieu), (MIL, bas_3), couleur=VERT, trait=(0, (3, 2)))
    ax.text(MIL - 0.012, (milieu + bas_3) / 2,
            "chaque corpus repasse\npar le même moteur", ha="right", va="center",
            fontsize=6.9, color=VERT, style="italic", linespacing=1.4)
    bas_7 = y

    # --- 8 : restitution --------------------------------------------------
    y = 0.170
    h8 = boite(ax, G, y, PLEIN,
               "07_figures.py → figures     ·     article_contenu.py → texte partagé, "
               "chiffres relus dans export/\n"
               "12_article_pdf.py → PDF     ·     17_article_docx.py → Word",
               "8.  Restitution", BLEU_P, BLEU)
    fleche(ax, (G + L / 2, bas_45), (G + L / 2, y + h8))
    fleche(ax, (D + L / 2, bas_7), (D + L / 2, y + h8))

    # --- bandeau ----------------------------------------------------------
    yb, hb = 0.038, 0.098
    ax.add_patch(FancyBboxPatch(
        (G, yb), PLEIN, hb, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0, facecolor="#f4f6f8", zorder=2))
    ax.text(0.5, yb + hb - 0.014, "CE QUE LA CHAÎNE ÉTABLIT", ha="center",
            va="top", fontsize=7.4, color=BLEU, fontweight="bold", zorder=3)
    ax.text(0.5, yb + hb - 0.036,
            f"L'approche intuitive identifie le bon auteur dans {100*c['naive']:.0f} %"
            f" des cas ; le protocole contrôlé, dans {100*c['rappel']:.0f} %.\n"
            "Les trois corpus traversent le même moteur et les mêmes analyses :\n"
            "si le verdict ne change pas, les featurings n'ont rien faussé.",
            ha="center", va="top", fontsize=7.4, color=GRIS_T, linespacing=1.5,
            zorder=3)

    sortie = IMAGES_DIR / "21_1_pipeline.png"
    fig.savefig(sortie, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    return sortie


# ---------------------------------------------------------------------------
# La page à imprimer
# ---------------------------------------------------------------------------
ETAPES_TEXTE = [
    ("1.  Récupérer le corpus",
     "Le corpus LRFAF est téléchargé depuis Hugging Face : {titres} chansons de rap "
     "français, 596 artistes, avec pour chacune les paroles, l'année et l'URL Genius "
     "d'origine. Ces paroles ont déjà été nettoyées par le constructeur du corpus — et "
     "privées de leurs balises de section, ce qui posera problème à l'étape 5."),
    ("2.  Collecter les artistes absents",
     "Mikeysem et web7 ne figurent pas dans le corpus : sans page Wikipédia, ils tombent "
     "hors de son critère d'inclusion. Ils sont collectés sur Genius, ainsi que Ziak, pour "
     "disposer de ses paroles avec leurs balises. Chaque identité est vérifiée à la main, "
     "un homonyme ruinant le test."),
    ("3.  Préparer le texte",
     "Nettoyage (les accents et l'apostrophe sont conservés : les premiers portent du "
     "signal en français, la seconde porte l'élision), puis découpe en mots et en suites "
     "de quatre caractères, puis filtrage — les textes de moins de 100 mots et les "
     "doublons entre artistes sont écartés. Les comptes sont mis en cache <b>par "
     "chanson</b> : un document d'artiste de n'importe quelle taille se recompose ensuite "
     "par simple addition, ce qui rend possibles les centaines de rééchantillonnages dont "
     "dépend la validation."),
    ("4.  Analyser le corpus publié",
     "Treize scripts, dans un ordre qui n'est pas décoratif. D'abord <i>valider</i> : "
     "établir que la méthode retrouve le bon auteur quand on connaît déjà la réponse. "
     "Ensuite <i>tester</i> les trois hypothèses. Enfin <i>décrire</i>. Un résultat "
     "négatif ne vaut rien tant qu'on n'a pas mesuré ce que la méthode sait trouver."),
    ("5.  Récupérer les balises de section",
     "Le corpus publié a perdu ses balises : impossible d'y distinguer le couplet de "
     "l'artiste de celui de son invité. Or laisser le couplet d'un invité dans le corpus "
     "de l'artiste principal rapproche mécaniquement les deux — et l'hypothèse testée "
     "porte justement sur une proximité. L'information n'est pas perdue : le corpus garde "
     "l'URL de chaque morceau, et la page Genius porte encore ses balises. {recol} pages "
     "sont re-téléchargées, en six heures environ."),
    ("6.  Reconstruire deux corpus propres",
     "La difficulté est de distinguer un membre de groupe d'un invité : la règle naïve "
     "viderait les groupes de leur contenu, les balises y nommant les membres, dont les "
     "couplets <i>sont</i> le texte du groupe. Un intervenant récurrent est donc tenu "
     "pour interne. Deux corpus en sortent : l'un retire les strophes d'invités "
     "({part} % des mots), l'autre écarte en entier tout titre qui en comporte "
     "({ecartes} titres)."),
    ("7.  Rejouer l'étude sur chaque corpus",
     "Les mêmes scripts sont relancés sur chaque variante, avec un cache et un dossier de "
     "résultats séparés : les chiffres du corpus publié restent intacts. C'est le test "
     "décisif — <b>si le verdict est le même sur les trois, les featurings n'ont jamais "
     "faussé l'analyse</b>, et on le sait au lieu de le supposer."),
    ("8.  Produire l'article",
     "Les figures sont fabriquées depuis les CSV, et le texte de l'article relit tous les "
     "chiffres dans les résultats au moment de la génération : aucune valeur n'est écrite "
     "à la main, donc aucune ne peut se désynchroniser de l'analyse qu'elle cite. Deux "
     "moteurs partagent ce texte, l'un vers le PDF, l'autre vers le Word."),
]


def page_imprimable(c: dict, schema: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer)

    police = Path("/System/Library/Fonts/Supplemental")
    for nom, f in [("T", "Times New Roman.ttf"), ("T-B", "Times New Roman Bold.ttf"),
                   ("T-I", "Times New Roman Italic.ttf"),
                   ("T-BI", "Times New Roman Bold Italic.ttf")]:
        pdfmetrics.registerFont(TTFont(nom, str(police / f)))
    pdfmetrics.registerFontFamily("T", normal="T", bold="T-B", italic="T-I",
                                  boldItalic="T-BI")
    S = getSampleStyleSheet()
    st = {
        "titre": ParagraphStyle("t", parent=S["Title"], fontName="T-B", fontSize=16,
                                leading=19, spaceAfter=3, textColor=colors.HexColor(BLEU)),
        "sous": ParagraphStyle("s", parent=S["Normal"], fontName="T-I", fontSize=10,
                               alignment=TA_CENTER, textColor=colors.HexColor("#555"),
                               spaceAfter=3),
        "aut": ParagraphStyle("a", parent=S["Normal"], fontName="T", fontSize=9,
                              alignment=TA_CENTER, textColor=colors.HexColor("#555"),
                              spaceAfter=10),
        "h": ParagraphStyle("h", parent=S["Normal"], fontName="T-B", fontSize=10.5,
                            leading=13, spaceBefore=9, spaceAfter=3,
                            textColor=colors.HexColor(BLEU)),
        "p": ParagraphStyle("p", parent=S["Normal"], fontName="T", fontSize=9.4,
                            leading=12.6, alignment=TA_JUSTIFY, spaceAfter=2),
        "leg": ParagraphStyle("l", parent=S["Normal"], fontName="T", fontSize=8.2,
                              leading=10.5, alignment=TA_CENTER,
                              textColor=colors.HexColor("#555"), spaceBefore=6),
        "code": ParagraphStyle("c", parent=S["Normal"], fontName="Courier", fontSize=8.6,
                               leading=12, leftIndent=10, spaceAfter=1,
                               textColor=colors.HexColor("#333")),
    }

    sortie = Path("pipeline_traitement.pdf")
    doc = SimpleDocTemplate(str(sortie), pagesize=A4, title="Chaîne de traitement des données",
                            author="Tassilo Westphalen",
                            leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                            topMargin=1.7 * cm, bottomMargin=1.7 * cm)
    histoire = [
        Paragraph("Chaîne de traitement des données", st["titre"]),
        Paragraph("Enquête stylométrique sur l'identité de Ziak — corpus LRFAF", st["sous"]),
        Paragraph("Tassilo Westphalen · CPES Sciences des données, Arts et Cultures "
                  "— PSL / Louis-le-Grand", st["aut"]),
    ]
    larg = 16.8 * cm
    img = Image(str(schema))
    img.drawWidth, img.drawHeight = larg, larg * img.imageHeight / img.imageWidth
    histoire += [img, PageBreak(),
                 Paragraph("Le détail, étape par étape", st["titre"]), Spacer(1, 6)]
    valeurs = {"titres": f"{c['titres']:,}".replace(",", "\u00a0"),
               "recol": f"{c['recol']:,}".replace(",", "\u00a0"),
               "part": f"{100 * c['part_retiree']:.0f}",
               "ecartes": f"{c['ecartes']:,}".replace(",", "\u00a0")}
    for titre, corps in ETAPES_TEXTE:
        histoire.append(Paragraph(titre, st["h"]))
        histoire.append(Paragraph(corps.format(**valeurs), st["p"]))
    histoire += [
        Paragraph("Reproduire", st["h"]),
        Paragraph("La chaîne s'exécute d'une commande. Chaque étape est idempotente : "
                  "ce qui existe n'est pas refait, et une relance reprend où elle s'est "
                  "arrêtée. Une exécution complète à partir de rien demande un peu plus "
                  "de sept heures, dont six pour la seule étape 5 ; les suivantes "
                  "tiennent en quarante minutes.", st["p"]),
        Spacer(1, 3),
        Paragraph("./pipeline.sh", st["code"]),
        Paragraph("./pipeline.sh --depuis variantes", st["code"]),
        Paragraph("./pipeline.sh balises --oui", st["code"]),
    ]
    doc.build(histoire)
    return sortie


if __name__ == "__main__":
    c = chiffres()
    schema = dessine(c)
    print("schéma écrit :", schema)
    print("page à imprimer :", page_imprimable(c, schema))
