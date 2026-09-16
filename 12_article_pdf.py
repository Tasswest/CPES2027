#!/usr/bin/env python3
"""Génère l'article scientifique au format PDF.

Le notebook `02_stylometrie_ziak.ipynb` mêle code, sorties et commentaire : c'est
le document de travail reproductible. Ce script en tire la version destinée à la
lecture — texte rédigé, figures légendées, tableaux mis en forme — dans
`article_ziak_stylometrie.pdf`.

Le texte vient de `article_contenu.py`, partagé avec la version Word
(`17_article_docx.py`) : les deux documents ne peuvent pas diverger. Tous les
chiffres sont relus dans `result/` au moment de la génération.
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
from article_contenu import construire
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



class RenduPDF:
    """Adaptateur entre le texte de l'article et la mise en page reportlab."""

    def p(self, texte, style="p"):
        P(texte, style)

    def gap(self, h=6):
        GAP(h)

    def tableau(self, lignes, legende, widths=None, align_num=True):
        tableau(lignes, legende, [w * cm for w in widths] if widths else None, align_num)

    def figure(self, image, legende):
        figure(image, legende)


construire(RenduPDF())

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
