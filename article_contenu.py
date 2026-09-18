#!/usr/bin/env python3
"""Texte de l'article, partagé par la version PDF et la version Word.

`construire(doc)` écrit l'article dans un objet de rendu qui fournit :

- `doc.p(texte, style)` — styles : titre, soustitre, auteur, date, abstract,
  h1, h2, p, note, clair, ref ;
- `doc.gap(points)` ;
- `doc.tableau(lignes, legende, widths, gauche)` — largeurs en centimètres,
  `gauche` donnant les colonnes alignées à gauche ;
- `doc.figure(image, legende)`.

L'article se lit à deux niveaux : la section 1 répond à la question sans
prérequis, et chaque passage technique est suivi d'un encadré « En clair »
(style `clair`) qui en donne le sens en langage ordinaire. Le style `note`
reste réservé aux réserves et aux nuances.

Le texte emploie un balisage minimal compris par les deux rendus : <b>, <i>,
<sub>, <super>, <br/>, &nbsp; et &amp;. Tous les chiffres sont relus dans
`export/` au moment de la génération.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULT = Path("export")


def pct(x, d=1):
    return f"{100 * float(x):.{d}f} %".replace(".", ",")


def num(x, d=2):
    # Signe moins typographique (U+2212), comme dans le texte rédigé.
    return f"{float(x):.{d}f}".replace(".", ",").replace("-", "−")


def construire(doc) -> None:
    P = doc.p
    # =========================================================================
    # Chiffres relus dans export/
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
    mval = pd.read_csv(RESULT / "10_1_validation_petite_taille.csv")
    mrk = pd.read_csv(RESULT / "10_2_rang_mikeysem.csv")
    mimp = pd.read_csv(RESULT / "10_4_imposteurs.csv")
    mcomp = pd.read_csv(RESULT / "10_5_rangs_compares.csv", index_col=0)
    disco = pd.read_csv(RESULT / "11_discographie_mikeysem.csv")
    areel = pd.read_csv(RESULT / "13_2_alias_reels_synthese.csv")
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
    mrang = int(mrr.rang_mikeysem.median())
    mcand = int(mrr.n_candidats.iloc[0])
    mscore = float(mimp[mimp.type == "hypothèse"].score.iloc[0])
    mref = mimp[mimp.type == "jumeau authentique"].score
    zexc = exc[exc.artiste == "Ziak"].iloc[0]
    # Taille du corpus Mikeysem telle que la voit le test stylométrique (tokens du
    # pipeline d'analyse, distincts du n_words LRFAF qui inclut l'en-tête Genius).
    from stylo_features import clean_lyrics, tokenize as _tok
    MIKE_MOTS = int(sum(len(_tok(clean_lyrics(t)))
                        for t in pd.read_csv("mikeysem_lrfaf.csv").lyrics))
    rang_dmin = int((exc.distance_min < zexc.distance_min).sum()) + 1

    # --- Hypothèse web7 (scripts 15 et 16) ---
    w15 = pd.read_csv(RESULT / "15_2_web7_synthese.csv")
    credits = pd.read_csv(RESULT / "15_3_credits_auteurs_ziak.csv")
    credits["year"] = pd.to_numeric(credits["year"], errors="coerce")
    t16 = pd.read_csv(RESULT / "16_1_test_2025_web7.csv")
    V1, V2 = "sans featurings ni ad-libs", "sans featurings"

    def ligne16(variante, debut):
        return t16[(t16.variante == variante) & t16.test.str.startswith(debut)].iloc[0]

    e1, e2 = ligne16(V1, "2025"), ligne16(V2, "2025")
    c100, c50 = ligne16(V1, "contrôle : 100"), ligne16(V1, "contrôle : 50")
    c100b, c50b = ligne16(V2, "contrôle : 100"), ligne16(V2, "contrôle : 50")
    tous = ligne16(V1, "toutes")
    clas_sf = ligne16(V1, "classement")
    w_full = w15[w15.variante == "textes complets"].iloc[0]
    w_sans = w15[w15.variante == "sans ad-libs entre parenthèses"].iloc[0]
    w_2019 = w15[w15.variante.str.startswith("web7 depuis")].iloc[0]
    w_inv = w15[w15.variante.str.startswith("inverse")].iloc[0]
    n_cred, n_ziak = int(credits.web7.sum()), len(credits)
    mots = lambda x: f"{int(x):,}".replace(",", "&nbsp;")

    # =========================================================================
    # Corps de l'article
    # =========================================================================
    P("Qui se cache derrière Ziak&nbsp;?", "titre")
    P("Une enquête stylométrique sur 37&nbsp;307 chansons de rap français", "soustitre")
    P("Tassilo Westphalen", "auteur")
    P("CPES Sciences des données, Arts et Cultures — Université PSL / Lycée Louis-le-Grand",
      "date")

    P("<b>Résumé.</b> Le rappeur Ziak est apparu en 2020 cagoulé et sans identité "
      "publique. Trois hypothèses circulent&nbsp;: il serait un rappeur déjà établi "
      "revenu sous un pseudonyme&nbsp;; il serait Mikeysem&nbsp;; ou ses textes "
      "seraient écrits par web7, anciennement 7 Jaws. Toutes trois sont "
      "testables&nbsp;: si un auteur en cache un autre, les textes doivent porter la "
      "même signature statistique. Nous conduisons ce test sur le corpus LRFAF "
      "(37&nbsp;307 chansons de rap français) au moyen d'un protocole d'attribution "
      "d'auteur à taille contrôlée, validé sur 177 artistes dont la réponse est "
      "connue, puis sur dix-sept liens d'auteur réels. "
      f"Avec une méthode qui retrouve le bon auteur dans {pct(best.recall_at_1, 0)} des "
      "cas, <b>aucun des 392 artistes éligibles du corpus ne présente la signature de "
      "Ziak</b>&nbsp;; Mikeysem, absent du corpus et collecté pour l'occasion, se "
      f"classe {mrang}<super>e</super> sur {mcand}&nbsp;; web7, enfin, n'apparaît "
      "jamais dans les premiers rangs, mais il est crédité co-auteur de "
      f"{n_cred} titres, et ceux qu'il co-signe sur l'album <i>Essonne History X</i> "
      "sont effectivement plus proches de son écriture "
      f"(p&nbsp;=&nbsp;{num(e1.p_valeur, 2)}). L'enquête livre en outre un résultat "
      "méthodologique&nbsp;: l'approche intuitive, qui compare des corpus d'artistes "
      "entiers, désigne un coupable avec assurance et n'a raison que dans "
      f"{pct(var.iloc[0].recall_at_1, 0)} des cas, car elle mesure surtout la quantité "
      "de texte disponible. Ziak écrit au centre de son genre, sans excentricité "
      "mesurable, mais sans proche parent identifiable.", "abstract")
    doc.gap(4)
    P("<b>Mots-clés&nbsp;:</b> stylométrie, attribution d'auteur, rap français, "
      "Delta de Burrows, méthode des imposteurs, ghostwriting, LRFAF.", "abstract")
    doc.gap(6)

    P("<b>Comment lire cet article.</b> Il se lit à deux niveaux. La <b>section 1</b> "
      "répond à la question en français courant, sans aucun prérequis&nbsp;: elle "
      "suffit à savoir ce que l'étude établit et ce qu'elle ne peut pas établir. Les "
      "sections suivantes exposent la méthode, les vérifications et les chiffres. "
      "Chaque passage technique y est suivi d'un encadré <b>«&nbsp;En "
      "clair&nbsp;»</b> qui en donne le sens en langage ordinaire, et le "
      "<b>lexique</b>, en fin d'article, définit les termes employés.", "note")
    doc.gap(6)

    # =====================================================================
    P("1. L'essentiel", "h1")
    P("Ziak publie son premier morceau en 2020. Il apparaît cagoulé, ne donne aucune "
      "identité civile et cultive ouvertement le mystère. Très vite, les auditeurs "
      "avancent des noms&nbsp;: ce masque cacherait un rappeur déjà connu, qui aurait "
      "recommencé une carrière sous un autre nom&nbsp;; ou bien Mikeysem, un rappeur "
      "de la même région&nbsp;; ou bien encore Ziak ne serait qu'une voix, les textes "
      "étant écrits par un autre — web7, anciennement 7 Jaws.")
    P("Ces rumeurs ont une propriété rare&nbsp;: on peut les mettre à l'épreuve. "
      "Depuis un siècle, on sait que la manière d'écrire trahit son auteur. Pas les "
      "thèmes — tous les rappeurs parlent d'argent, de rue et de réussite — mais la "
      "plomberie du texte&nbsp;: la fréquence des petits mots («&nbsp;de&nbsp;», "
      "«&nbsp;que&nbsp;», «&nbsp;mais&nbsp;»), la façon de couper les mots, les "
      "enchaînements de syllabes. Ces habitudes sont trop nombreuses et trop peu "
      "conscientes pour être maquillées&nbsp;; changer de nom ne les change pas. "
      "Compter ces habitudes sur des milliers de chansons revient donc à relever une "
      "empreinte, puis à demander si celle de Ziak se retrouve ailleurs.")
    P("Nous l'avons fait sur le corpus LRFAF, qui rassemble 37&nbsp;307 chansons de "
      "rap français. Voici la réponse.")

    doc.tableau(
        [["L'hypothèse", "Ce que l'analyse trouve", "Verdict"],
         ["Ziak est un rappeur du corpus,<br/>revenu sous un autre nom",
          "Aucun des 392 artistes testés ne porte sa signature. Les quatre analyses "
          "menées en parallèle ne s'accordent même pas sur un favori — alors qu'elles "
          "s'accordent neuf fois sur dix quand la réponse existe.",
          "<b>Écartée</b><br/><i>preuve solide</i>"],
         ["Ziak est Mikeysem",
          f"Testé à part, car absent du corpus&nbsp;: il arrive "
          f"{mrang}<super>e</super> sur {mcand}, derrière six artistes que personne ne "
          f"soupçonne. Mais on ne dispose que de {mots(MIKE_MOTS)} mots de lui.",
          "<b>Non soutenue</b><br/><i>preuve limitée</i>"],
         ["web7 (ex-7 Jaws) écrit<br/>les textes de Ziak",
          f"web7 n'apparaît jamais parmi les proches de Ziak. Mais Genius le crédite "
          f"co-auteur de {n_cred} titres sur {n_ziak}, et sur l'album de 2025 les "
          f"titres qu'il co-signe sont bien plus proches de son écriture que les "
          f"autres.",
          "<b>Version forte écartée,<br/>co-écriture réelle</b><br/><i>preuve "
          "moyenne</i>"]],
        "Les trois hypothèses et ce que les données en disent. Le détail de chaque "
        "ligne occupe respectivement les sections 6, 8 et 9.",
        widths=[4.4, 8.4, 3.6], gauche=(0, 1))

    P("1.1 Pourquoi l'on peut croire un résultat négatif", "h2")
    P("Dire «&nbsp;la méthode n'a rien trouvé&nbsp;» n'a de valeur que si la méthode "
      "sait trouver quand il y a quelque chose à trouver. C'est la vérification "
      "centrale de ce travail, et elle occupe plus de place que le test lui-même.")
    P("Nous avons donc joué 1&nbsp;770 fois à un jeu dont nous connaissions la "
      "réponse&nbsp;: prendre un artiste, lui cacher la moitié de son œuvre, et "
      "demander à la méthode de retrouver l'auteur de cette moitié parmi des "
      f"centaines de candidats. Elle y parvient {pct(best.recall_at_1, 0)} du temps. "
      "Puis nous l'avons confrontée à des cas réels plutôt qu'à des exercices&nbsp;: "
      "des rappeurs qui ont réellement changé de nom au milieu de leur carrière. "
      "<b>Joke devenu Ateyaba est retrouvé du premier coup, parmi 393 candidats, dans "
      "la totalité des essais</b> — alors même qu'il avait annoncé vouloir "
      "«&nbsp;tuer&nbsp;» son ancien nom. Changer d'identité n'efface pas la manière "
      "d'écrire.")
    P("Appliquée à Ziak, cette même méthode ne se contente pas de ne rien "
      "trouver&nbsp;: elle place son meilleur candidat exactement là où elle place "
      "les cas d'auteurs qu'elle sait absents du corpus. C'est une réponse, pas un "
      "silence.")

    P("1.2 Ce que cette étude ne dit pas", "h2")
    P("Elle ne donne <b>aucun nom d'état civil</b>, et n'aurait pas pu en donner "
      "un&nbsp;: la stylométrie rapproche des textes, pas des personnes. Elle ne "
      "couvre que les artistes présents dans le corpus, plus Mikeysem et web7 ajoutés "
      "pour l'occasion&nbsp;; un rappeur que Genius ne documente pas serait resté "
      "invisible. Le corpus de Mikeysem est mince, ce qui rend cette ligne-là moins "
      "solide que les autres. Enfin — et c'est la limite la plus gênante — un auteur "
      "de l'ombre non crédité, écrivant délibérément dans la voix de Ziak, ne serait "
      "détecté par aucun des tests employés ici. La section 9 montre précisément "
      "cela&nbsp;: la co-écriture de web7 n'est devenue visible que parce que les "
      "crédits disaient où regarder.")
    P("Ce que l'enquête établit est donc négatif et borné&nbsp;: <b>parmi les "
      "candidats qu'il était possible de tester, aucun n'est Ziak</b>. L'anonymat "
      "tient.", "note")

    # =====================================================================
    P("2. Une question qui se teste", "h1")
    P("Ziak publie son premier titre en 2020. Cagoulé en public, il ne divulgue "
      "aucune identité civile et cultive ouvertement le mystère. Cette discrétion a "
      "nourri parmi les auditeurs une hypothèse récurrente&nbsp;: le nom masquerait "
      "un artiste déjà établi, qui aurait recommencé une carrière sous un "
      "pseudonyme. D'autres versions attribuent plutôt l'écriture de ses textes à "
      "un autre rappeur.")
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
      "Nous montrons en section 4 qu'une démarche intuitive, appliquée aux mêmes "
      "données, désigne un artiste avec assurance et se trompe trois fois sur "
      "quatre. La contribution de cet article tient donc autant à la manière de "
      "poser la question qu'à la réponse obtenue.")

    P("3. Les données", "h1")
    P("3.1 Le corpus LRFAF", "h2")
    P("Le corpus LRFAF (de Courson, 2024) rassemble 37&nbsp;307 textes de rap "
      "français issus de genius.com, obtenus en croisant les catégories Wikipédia "
      "et Wikidata avec l'API de Genius, puis en extrayant les paroles au moyen du "
      "paquet <i>lyricsgenius</i>. Il fournit, outre les textes, des métadonnées "
      "(artiste, année, vues, contributeurs) et une batterie de mesures "
      "lexicométriques.")
    P("Après nettoyage — suppression des textes de moins de 100 mots et des "
      "doublons de paroles entre artistes, qui correspondent à des featurings ou à "
      "des rééditions — l'analyse porte sur 32&nbsp;923 titres et 596 artistes.")

    P("3.2 Le corpus de Ziak", "h2")
    P("Ziak y est représenté par 43 titres publiés entre 2020 et 2024, soit "
      "23&nbsp;886 mots. C'est un volume modeste mais très au-dessus du seuil usuel "
      "de quelques milliers de mots requis par les méthodes employées ici. La "
      "section 6.1 vérifiera empiriquement que cette matière suffit.")
    P("Un point de vocabulaire commande tout le reste. Nous cherchons une "
      "<b>vérification d'auteur en ensemble ouvert</b>&nbsp;: la bonne réponse peut "
      "ne pas figurer dans le corpus. C'est une tâche plus difficile que "
      "l'attribution en ensemble fermé, où l'on sait que l'auteur est l'un des "
      "candidats — une méthode d'ensemble fermé désigne <i>toujours</i> quelqu'un. "
      "Il faut donc un protocole capable de répondre «&nbsp;personne&nbsp;».")
    P("<b>En clair.</b> La plupart des outils d'attribution d'auteur fonctionnent "
      "comme une séance d'identification où le suspect est forcément dans la "
      "rangée&nbsp;: on doit désigner quelqu'un. Ici, le suspect peut très bien ne "
      "pas être là. Tout l'article consiste à construire un dispositif qui a le "
      "droit de répondre «&nbsp;aucun de ceux-là&nbsp;» — et à mesurer à quel point "
      "on peut se fier à cette réponse.", "clair")

    # =====================================================================
    P("4. Première leçon&nbsp;: la démarche intuitive désigne un coupable, et se "
      "trompe", "h1")
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
    P("<b>En clair.</b> Un artiste très prolifique est comme un portrait-robot très "
      "vague&nbsp;: à force de tout contenir, il finit par ressembler à tout le "
      "monde. Le classement «&nbsp;naïf&nbsp;» met donc en tête les rappeurs qui ont "
      "beaucoup publié, et non ceux qui écrivent comme Ziak. Pour corriger cela, on "
      "donne à chaque candidat exactement la même quantité de texte&nbsp;: à armes "
      "égales, seul le style peut encore les départager.", "clair")
    P("L'argument décisif n'est pas qu'un classement change, mais qu'on peut "
      "mesurer lequel a raison. Nous soumettons trois variantes au même protocole "
      "de vérité-terrain&nbsp;: on prélève chez un artiste un échantillon de la "
      "taille du corpus de Ziak, on le traite comme un texte anonyme, et l'on "
      "regarde si la méthode le rattache au reste de son œuvre.")

    doc.tableau(
        [["Variante", "Rang 1", "Top 5", "Top 20", "Rang médian"]] +
        [[r.variante, pct(r.recall_at_1, 1), pct(r.recall_at_5, 1),
          pct(r.recall_at_20, 1), f"{r.rang_median:.0f}"] for r in var.itertuples()],
        "Puissance des trois variantes, évaluée sur 177 artistes dont la réponse est "
        "connue. Le contrôle de la taille des documents apporte 48 points, la "
        "standardisation des traits 18 de plus.",
        widths=[6.6, 2.4, 2.4, 2.4, 2.6])

    P("L'approche naïve identifie le bon auteur dans un quart des cas seulement, "
      "alors qu'elle produit un classement d'allure sérieuse. Ramener tous les "
      "documents à une taille identique porte ce taux à 74&nbsp;%, et standardiser "
      "les traits à 92&nbsp;%. Sans étalonnage, un classement de distances ne "
      "permet aucune conclusion, quelle que soit sa netteté apparente.")

    doc.figure("02_1_biais_de_taille.png",
           "Ce que mesure l'approche naïve. (a) la distance à Ziak décroît avec la "
           "taille du corpus du candidat&nbsp;; (b) les rangs se réorganisent "
           "entièrement une fois la taille neutralisée — Mister You passe du 9<super>e</super> "
           "au 138<super>e</super> rang&nbsp;; (c) puissance comparée sur vérité-terrain.")

    # =====================================================================
    P("5. Le protocole retenu", "h1")
    P("Le protocole repose sur trois choix. <b>Les traits</b> sont les "
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

    doc.tableau(
        [["Traits", "Distance", "Rang 1", "Top 5", "Top 20"]] +
        [[r.features, r.metric.replace("_", " "), pct(r.recall_at_1, 1),
          pct(r.recall_at_5, 1), pct(r.recall_at_20, 1)]
         for r in puis.itertuples()],
        "Puissance des quatre combinaisons, sur 1&nbsp;770 essais couvrant 177 "
        "artistes. Les 4-grammes de caractères associés au Cosine Delta dominent.",
        widths=[2.6, 3.4, 2.6, 2.6, 2.6])

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
    P("<b>En clair.</b> Un jury sommé de désigner un coupable en désigne toujours "
      "un&nbsp;: le fait qu'un nom arrive en tête ne prouve donc rien. La question "
      "utile est celle de l'écart&nbsp;: le premier se détache-t-il nettement du "
      "deuxième, ou l'emporte-t-il d'un cheveu&nbsp;? Le score de séparation mesure "
      "cela. Nous savons à quoi il ressemble quand le bon auteur est présent, et à "
      "quoi il ressemble quand il est absent&nbsp;: il suffit alors de regarder "
      "duquel des deux cas Ziak se rapproche.", "clair")

    doc.figure("03_1_validation.png",
           "Validation du protocole. (a) courbe de rappel de la meilleure "
           "combinaison&nbsp;; (b) les quatre combinaisons testées&nbsp;; (c) puissance "
           "par génération d'artistes — elle est maximale sur celle de Ziak.")

    # =====================================================================
    P("6. Résultat principal&nbsp;: aucun rappeur du corpus n'est Ziak", "h1")
    P("6.1 Le style de Ziak est-il seulement détectable&nbsp;?", "h2")
    P("Avant d'interpréter un échec d'identification, il faut écarter l'explication "
      "triviale&nbsp;: un corpus trop petit ou trop hétérogène pour porter une "
      "signature. Nous coupons donc le corpus de Ziak en deux moitiés disjointes et "
      "cherchons la seconde depuis la première. <b>La moitié cible ressort au "
      "premier rang dans 100&nbsp;% des tirages</b> avec la meilleure combinaison, "
      "parmi 392 candidats. Le style de Ziak est donc parfaitement détectable et "
      "son corpus suffisamment homogène&nbsp;: aucun échec ultérieur ne pourra être "
      "imputé à une insuffisance de matière.")

    P("6.2 Aucun candidat ne s'impose", "h2")
    P("Nous classons ensuite les 392 autres artistes, sur 30 rééchantillonnages et "
      "avec les quatre combinaisons. Le résultat est éloquent par son "
      "incohérence&nbsp;: chaque méthode a son favori — Beendo Z, Rimkus, Zkr, "
      "L'Animalerie — et <b>aucun candidat ne s'impose d'une méthode à l'autre</b>. "
      "Pour les artistes de contrôle, dont la réponse est connue, les quatre "
      "combinaisons convergent dans neuf cas sur dix. Ici, elles divergent.")

    P("6.3 Verdict", "h2")
    doc.tableau(
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
        widths=[2.0, 2.8, 2.6, 3.2, 3.2, 2.6])

    P("Autrement dit, ce n'est pas seulement que la méthode ne trouve pas&nbsp;: "
      "c'est qu'elle trouve activement l'absence de correspondance.")

    doc.figure("04_1_verdict_H0_H1.png",
           "Ziak confronté aux deux hypothèses, pour les quatre combinaisons. La "
           "distribution verte correspond aux cas où l'auteur est réellement dans le "
           "corpus, la grise aux cas où il en est absent&nbsp;; le trait rouge marque "
           "la position de Ziak.")

    P("6.4 Robustesse", "h2")
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
    P("<b>En clair.</b> Au lieu de demander «&nbsp;qui ressemble le plus à "
      "Ziak&nbsp;?&nbsp;», on prend un candidat précis et on le met en concurrence "
      "avec des inconnus tirés au sort, des dizaines de fois&nbsp;: le score est la "
      "part des duels qu'il remporte. Un artiste qui est réellement le même auteur "
      "gagne presque tous ses duels. Les proches de Ziak gagnent nettement plus "
      "souvent que le hasard — ils appartiennent à la même famille musicale — mais "
      "bien moins souvent qu'un véritable alias. Ressembler à un courant n'est pas "
      "être la même personne&nbsp;; c'est exactement la distinction qu'un classement "
      "brut de distances ne permet jamais de faire.", "clair")

    doc.figure("05_1_candidats_imposteurs.png",
           "(a) aucun candidat n'est stable d'une méthode à l'autre&nbsp;; (b) même le "
           "meilleur candidat reste loin du niveau qu'atteint un véritable alias.")

    # =====================================================================
    P("7. La méthode tient-elle sur des cas réels&nbsp;?", "h1")
    P("Tout ce qui précède repose sur une validation par jumeaux fabriqués&nbsp;: on "
      "coupe l'œuvre d'un artiste en deux et l'on cherche une moitié depuis l'autre. "
      "C'est une tâche <i>facile</i> — les deux moitiés partagent la même époque, les "
      "mêmes thèmes, le même producteur. La puissance qu'on y mesure est donc une "
      "borne optimiste, et c'est l'objection la plus sérieuse qu'on puisse opposer au "
      "verdict. Cette section y répond avec des cas où la vérité est connue "
      "indépendamment du corpus.")

    P("7.1 Recouvrements entre un artiste et son groupe", "h2")
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

    doc.tableau(
        [["Artiste → groupe", "Rappeurs", "Rang médian (sur 392)"]] +
        [[f"{r.solo} → {r.groupe}", str(int(r.n_rappeurs)), f"{int(r.rang_median)}"]
         for r in areel.sort_values("rang_median").itertuples()],
        "Rang du groupe, interrogé depuis les textes solo de l'un de ses membres. "
        "L'effet de dilution est net&nbsp;: un duo se retrouve aisément, un groupe de "
        "huit se perd dans le classement.",
        widths=[7.4, 2.4, 4.6])

    P(f"Le lien est retrouvé dans le top 20 pour "
      f"{pct((areel.rang_median <= 20).mean(), 0)} des paires — et "
      f"{pct((duos.rang_median <= 20).mean(), 0)} des duos, cas le plus proche d'un "
      f"alias. <b>La puissance réelle est donc inférieure aux "
      f"{pct(best.recall_at_1, 0)} mesurés sur jumeaux simulés</b>&nbsp;: il faut "
      "compter avec une chance sur cinq à une sur trois de manquer un lien, selon le "
      "degré de dilution.")
    P("<b>En clair.</b> Retrouver un auteur qui n'a écrit qu'une partie d'un disque, "
      "c'est reconnaître une voix dans un chœur&nbsp;: à deux, c'est facile&nbsp;; à "
      "huit, elle se noie. Cela borne la portée du verdict rendu sur Ziak&nbsp;: il "
      "vaut pleinement contre l'hypothèse d'un rappeur qui écrirait seul sous deux "
      "noms, beaucoup moins contre celle d'une contribution noyée dans un "
      "collectif.", "clair")

    doc.figure("13_1_alias_reels.png",
           "Validation sur quatorze recouvrements d'auteur réels. (a) rang du groupe "
           "vu depuis le solo&nbsp;; (b) plus l'auteur est dilué dans un collectif, "
           "moins il est détectable&nbsp;; (c) le meilleur candidat de Ziak se détache "
           "moins que dans la quasi-totalité des cas à lien réel.")

    P("7.2 Un changement d'identité efface-t-il la signature&nbsp;?", "h2")
    P("Reste l'objection de fond&nbsp;: un artiste qui se réinvente sous un autre nom "
      "change peut-être aussi de manière d'écrire. La fusion opérée par Genius permet "
      "justement de le tester, en découpant ces artistes <b>de part et d'autre de "
      "leur changement d'identité</b>. On interroge la période antérieure et l'on "
      "cherche la période postérieure, placée dans le pool sous une autre étiquette. "
      "Trois cas sont documentés dans le corpus, et le premier est le plus "
      "net&nbsp;: Ateyaba a publiquement déclaré vouloir «&nbsp;tuer Joke&nbsp;».")

    doc.tableau(
        [["Changement d'identité", "Rang médian", "Séparation", "Trouvé au rang 1"]] +
        [[r.libelle, f"{int(r.rang_median)}", num(r.sep_cible), pct(r.taux_rang1, 0)]
         for r in chg.itertuples()] +
        [[f"<i>Contrôles sans changement (n = {len(ctl)})</i>",
          f"<i>{ctl.rang_median.median():.0f}</i>", f"<i>{num(ctl.sep_cible.mean())}</i>",
          f"<i>{pct((ctl.rang_median == 1).mean(), 0)}</i>"]],
        "Période postérieure au changement de nom, recherchée depuis la période "
        "antérieure. La dernière ligne donne le repère&nbsp;: des artistes découpés au "
        "même endroit de leur carrière, mais qui n'ont jamais changé de nom.",
        widths=[6.4, 2.8, 2.8, 3.2])

    P("Le résultat est net, et il lève l'objection plutôt qu'il ne la confirme. "
      "<b>Joke → Ateyaba est retrouvé au premier rang sur 393 candidats, dans la "
      "totalité des tirages</b>, avec une séparation de −4,98 — alors même que le "
      "changement d'identité était revendiqué. Sur les trois cas, le rang médian "
      f"passe de {ctl.rang_median.median():.0f} (artistes sans changement) à "
      f"{chg.rang_median.median():.0f}, et la séparation reste inchangée "
      f"({num(ctl.sep_cible.mean())} contre {num(chg.sep_cible.mean())}). Changer de "
      "nom, de registre et d'époque ne suffit pas à effacer la signature.")

    doc.figure("14_1_alias_temporel.png",
           "Ce que coûte un changement d'identité. (a) les trois cas documentés&nbsp;; "
           "(b) leur rang comparé à celui d'artistes n'ayant jamais changé de "
           "nom&nbsp;; (c) séparation du meilleur candidat, seule grandeur comparable "
           "au cas Ziak.")

    P("Ces deux tests tirent dans des directions opposées, et il faut les lire "
      "ensemble. La puissance est <i>plus faible</i> qu'annoncée dès lors que "
      "l'auteur recherché ne signe qu'une partie des textes. Mais elle ne s'effondre "
      "<i>pas</i> lorsqu'il change d'identité, ce qui était la crainte principale. "
      "Or c'est bien cette seconde situation qui correspond à l'hypothèse testée sur "
      "Ziak. Sur la seule grandeur comparable — la séparation du meilleur candidat du "
      "classement — son favori se détache moins bien que dans "
      f"{pct((atemp_b.sep_top1 < zsep.sep_top1.mean()).mean(), 0)} de ces tests à lien "
      "réel.", "note")

    # =====================================================================
    P("8. Ce n'est pas Mikeysem", "h1")
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
      f"{mots(MIKE_MOTS)} mots — trois fois moins que la taille "
      "de référence employée jusqu'ici. Les autres n'ont aucune transcription "
      "disponible, dont l'intégralité du projet <i>Prochains Héritiers</i> (10 titres, "
      "2022). Aller chercher ces paroles ailleurs romprait la compatibilité "
      "méthodologique avec LRFAF, dont toutes les transcriptions proviennent de "
      "Genius et de ses conventions.")
    P("Cette petitesse impose de recalibrer avant d'interpréter. À "
      f"{mots(MIKE_MOTS)} mots de candidat, la méthode place le vrai auteur au "
      f"premier rang dans {pct((mh1.rank_twin == 1).mean(), 0)} des cas et dans le "
      f"top 20 dans {pct((mh1.rank_twin <= 20).mean(), 0)}&nbsp;: la puissance "
      "baisse, mais un alias authentique resterait très majoritairement détectable.")

    doc.tableau(
        [["Artiste", f"Rang parmi {mcand} candidats"]] +
        [[a, f"{int(r)}"] for a, r in mcomp.rang_median.items()],
        "Rangs médians vus depuis Ziak, à taille strictement égale. Six artistes que "
        "personne ne soupçonne sont plus proches de Ziak que Mikeysem.",
        widths=[6.0, 5.4])

    P("Aucune des quatre combinaisons ne place Mikeysem dans le top 20&nbsp;: son "
      f"rang médian est de {mrang} sur {mcand}. Son score de séparation "
      f"({num(mrr.sep_mikeysem.mean())}) est même plus faible que celui d'un auteur "
      "typiquement absent du corpus&nbsp;: il n'est pas un candidat ordinaire ayant "
      "manqué la première place, mais un artiste particulièrement éloigné. Au test "
      f"des imposteurs, il obtient {num(mscore)}, à peine au-dessus du hasard "
      f"({num(1 / 26)}), là où un alias authentique atteint {num(mref.median())} en "
      f"médiane&nbsp;; seuls {pct((mref <= mscore).mean(), 0)} des vrais jumeaux font "
      "moins bien.")

    doc.figure("10_1_test_mikeysem.png",
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

    # =====================================================================
    P("9. Ce n'est pas 7 Jaws non plus — mais il a écrit avec lui", "h1")
    P("Une autre version de la rumeur, rapportée par la presse musicale "
      "(Générations), ne fait pas de Ziak un seul artiste mais en répartit les "
      "rôles&nbsp;: <b>web7</b>, anciennement 7 Jaws, écrirait les textes&nbsp;; "
      "Mikeysem les interpréterait sous le masque&nbsp;; le producteur Seezy serait "
      "«&nbsp;Hellboy&nbsp;». Aucune preuve n'est citée, et web7 a refusé d'en parler "
      "sans démentir. Cette version rendrait d'ailleurs compte du résultat de la "
      "section 8&nbsp;: si Mikeysem interprète sans écrire, ses propres textes n'ont "
      "aucune raison de ressembler à ceux de Ziak.")
    P("C'est une hypothèse de <i>ghostwriting</i>, qui ne se teste pas tout à fait "
      "comme un alias. Un auteur qui écrit pour la voix d'un autre adapte son "
      "registre, ce qui dilue sa signature. Surtout, deux composantes du texte "
      "transcrit n'appartiennent pas à l'auteur&nbsp;: les <b>ad-libs</b> de "
      "l'interprète («&nbsp;uh&nbsp;», «&nbsp;huh&nbsp;», «&nbsp;gang&nbsp;»), que "
      "Genius note surtout entre parenthèses — c'est le cas de 87&nbsp;% des "
      "«&nbsp;uh&nbsp;» de Ziak —, et les <b>couplets d'invités</b>, écrits par "
      "d'autres. Les deux ont été retirés.")

    P("9.1 Données et nettoyage", "h2")
    P("web7 ne figure pas dans LRFAF. Sa page Genius, qui répertorie «&nbsp;7 "
      f"Jaws&nbsp;» parmi ses noms, compte {int(w_full.n_titres_web7)} titres "
      f"satisfaisant aux critères du corpus, soit {mots(w_full.mots_web7)} mots — "
      "davantage que les 47 titres recensés sur Deezer, Genius conservant ses premiers "
      "EP et ses freestyles. Contrairement au cas Mikeysem, la matière ne manque pas.")
    P("Les couplets d'invités ont été retirés à partir des balises de section de "
      "Genius, qui nomment l'interprète de chaque passage. La règle retenue est "
      "stricte&nbsp;: une section n'est conservée que si elle revient au seul artiste "
      "principal, une section partagée étant écartée faute de pouvoir savoir qui en a "
      f"écrit quoi. Cela retire {pct(e1.pct_featurings_ziak / 100)} des mots de Ziak et "
      f"{pct(e1.pct_featurings_web7 / 100)} de ceux de web7. Ce nettoyage n'est possible "
      "que sur des paroles collectées directement&nbsp;: LRFAF a supprimé ces balises "
      "avant publication, si bien que ses artistes conservent leurs featurings. Ils ne "
      "servent ici qu'à étalonner l'échelle des traits.")
    P("<b>En clair.</b> Sur un morceau, tout le texte n'est pas du rappeur dont le "
      "nom est sur la pochette&nbsp;: un invité vient poser son couplet, et "
      "l'interprète lâche des interjections qu'aucun parolier n'a écrites. Laisser "
      "ces morceaux de texte fausserait le calcul, en rapprochant artificiellement "
      "les artistes qui collaborent. Les balises de Genius disent qui chante "
      "quoi&nbsp;; nous ne gardons que les passages attribués au seul artiste "
      "principal.", "clair")

    P("9.2 Au niveau de l'artiste&nbsp;: aucune proximité", "h2")
    doc.tableau(
        [["Variante", "Rang médian de web7", "Dans le top 20"]] +
        [[lib, f"{r_:.0f} / {n_:.0f}", pct(t_ / 100, 0)] for lib, r_, n_, t_ in [
            ("Textes complets", w_full.rang_median_web7, w_full.n_candidats, w_full["top20_%"]),
            ("Sans ad-libs", w_sans.rang_median_web7, w_sans.n_candidats, w_sans["top20_%"]),
            ("web7 depuis 2019, sans ad-libs", w_2019.rang_median_web7,
             w_2019.n_candidats, w_2019["top20_%"]),
            ("Sans featurings ni ad-libs", clas_sf.rang_median, w_sans.n_candidats,
             100 * clas_sf.top20),
            ("<i>Test inverse&nbsp;: Ziak vu depuis web7</i>", w_inv.rang_median_web7,
             w_inv.n_candidats, w_inv["top20_%"]),
        ]],
        "Rang de web7 parmi les candidats, interrogé depuis les textes de Ziak. Aucune "
        "variante ne le rapproche des premiers rangs, ni le retrait des ad-libs, ni "
        "celui des featurings, ni la restriction à sa production contemporaine.",
        widths=[7.2, 4.4, 3.4])
    P("Quelle que soit la variante, web7 se classe au-delà du centième rang et "
      "n'entre jamais dans le top 20. Sa séparation "
      f"({num(w_full.sep_web7)}) est sans commune mesure avec celles des liens d'auteur "
      "réels de la section 7. Dans le même temps, les voisins habituels de Ziak — "
      "Kerchak, Beendo Z, Werenoi — restent dans les premiers rangs&nbsp;: la méthode "
      "fonctionne normalement, elle ne trouve simplement pas web7.")

    P("9.3 Ce que disent les crédits", "h2")
    P("L'examen de la collecte a pourtant fait apparaître un fait documentaire. "
      f"<b>Genius crédite web7 comme co-auteur de {n_cred} des {n_ziak} titres de "
      "Ziak</b>, toujours aux côtés de Ziak lui-même&nbsp;: deux en 2021 "
      "(<i>Akimbo</i>, <i>Parasite</i>), huit en 2025, aucun en 2020 ni entre 2022 et "
      f"2024. Les auteurs sont renseignés pour chacun des {n_ziak} titres&nbsp;: "
      "l'absence de web7 sur les autres est donc une information, non une lacune.")
    P("Les deux constats se concilient. Parmi les 43 titres de Ziak qu'interroge le "
      "test principal, <b>deux seulement</b> sont crédités à web7, soit moins de "
      "5&nbsp;%. Or la section 7 a montré que la détection s'effondre quand l'auteur "
      "recherché ne signe qu'une petite fraction des textes. Un test mené au niveau de "
      "l'artiste ne pouvait donc pas voir une contribution aussi minoritaire — encore "
      "fallait-il vérifier qu'elle laisse une trace lorsqu'on la cherche au bon "
      "endroit. Ces crédits, saisis par la communauté de Genius et généralement "
      "recopiés des crédits officiels, restent à confirmer par une source indépendante.")

    P("9.4 Une expérience naturelle&nbsp;: <i>Essonne History X</i>", "h2")
    P("L'année 2025 offre une situation presque expérimentale. Sur les "
      f"{int(e1.n_titres_2025)} titres de Ziak parus cette année-là, 22 appartiennent "
      "au même album, <i>Essonne History X</i>, dont les "
      f"{int(e1.n_titres)} titres co-crédités à web7&nbsp;; {int(e1.n_non_credites)} "
      "titres ne le sont pas. Même artiste, même année, même disque&nbsp;: si la "
      "contribution de web7 laisse une trace, les titres crédités doivent être plus "
      "proches de ses propres textes que ne l'est un ensemble quelconque de titres "
      "de la même période.")
    P("Le test compare la distance entre le profil agrégé des huit titres crédités et "
      "celui de web7 à la même distance calculée pour 5&nbsp;000 ensembles de huit "
      "titres tirés au hasard parmi les 23. La taille du groupe restant identique, "
      "l'effet de taille est neutralisé. Deux contrôles établissent la puissance du "
      "test&nbsp;: on remplace les titres crédités par des textes de web7 lui-même, "
      "écartés de son profil de référence, puis par un mélange à parts égales de "
      "textes de web7 et de Ziak.")
    doc.tableau(
        [["", "Sans featurings<br/>ni ad-libs", "Sans featurings<br/>seulement"],
         ["Distance à web7 — 8 titres crédités", num(e1.distance, 3), num(e2.distance, 3)],
         ["Distance à web7 — 15 titres non crédités",
          num(e1.distance_non_credites, 3), num(e2.distance_non_credites, 3)],
         ["Médiane de 5&nbsp;000 tirages de 8 titres",
          num(e1.distance_mediane_hasard, 3), num(e2.distance_mediane_hasard, 3)],
         ["<b>p-valeur</b>", f"<b>{num(e1.p_valeur, 3)}</b>", f"<b>{num(e2.p_valeur, 3)}</b>"],
         ["Contrôle 100&nbsp;% web7 — détecté",
          pct(c100.taux_detection_p05, 0), pct(c100b.taux_detection_p05, 0)],
         ["Contrôle 50&nbsp;% web7 — détecté",
          pct(c50.taux_detection_p05, 0), pct(c50b.taux_detection_p05, 0)]],
        "Expérience naturelle sur les titres de Ziak parus en 2025. La p-valeur est la "
        "part des tirages aléatoires de huit titres plus proches de web7 que les huit "
        "titres qui lui sont crédités.",
        widths=[7.8, 3.6, 3.6])
    P("<b>Les titres co-écrits avec web7 sont significativement plus proches de son "
      "écriture que les autres titres de la période</b> "
      f"(p&nbsp;=&nbsp;{num(e1.p_valeur, 3)} sans featurings ni ad-libs, "
      f"{num(e2.p_valeur, 3)} sans featurings seulement). Les contrôles montrent que le "
      "test pouvait le voir&nbsp;: un texte entièrement de web7 est détecté dans "
      f"{pct(c100.taux_detection_p05, 0)} des tirages, un mélange à parts égales dans "
      f"{pct(c50.taux_detection_p05, 0)}.")
    P("<b>En clair.</b> On a mis d'un côté les huit titres crédités à web7, et de "
      "l'autre 5&nbsp;000 paquets de huit titres tirés au hasard dans le même album. "
      "Si web7 n'y était pour rien, les huit vrais titres devraient ressembler à "
      "n'importe quel paquet de huit. Ce n'est pas le cas&nbsp;: seuls environ "
      f"{pct(e1.p_valeur, 0)} des tirages au hasard font aussi bien. C'est ce que "
      "signifie la p-valeur — la probabilité d'obtenir un résultat au moins aussi net "
      "par pure coïncidence. Le seuil d'usage est de 5&nbsp;%&nbsp;: on est en "
      "dessous, mais de peu.", "clair")
    doc.figure("16_1_web7.png",
               "L'hypothèse web7. (a) au niveau de l'artiste, web7 se classe loin "
               "derrière les voisins de Ziak&nbsp;; (b) sur l'album <i>Essonne History "
               "X</i>, les huit titres co-crédités sont plus proches de web7 que des "
               "tirages aléatoires de huit titres&nbsp;; (c) contrôles de "
               "puissance&nbsp;: le test détecte un texte entièrement ou à moitié écrit "
               "par web7.")
    P("Trois précisions bornent la portée de ce résultat. Une p-valeur de cet ordre "
      "est modeste&nbsp;: elle repose sur une seule expérience de huit titres, parmi "
      "plusieurs tests apparentés. Le biais de taille joue <i>contre</i> elle&nbsp;: les "
      f"titres crédités sont plus courts ({int(e1.mots_credites / e1.n_titres)} mots en "
      f"moyenne contre {int(e1.mots_non_credites / e1.n_non_credites)}), donc plus "
      "bruités, et en principe plus éloignés de tout profil. Enfin, le signal observé "
      "est plus faible que celui d'une co-écriture à parts égales dans le style propre "
      "de web7, ce qui cadre avec une contribution partielle, ou écrite dans la voix de "
      "Ziak. Toutes années confondues, sans contrôle de l'époque, la tendance persiste "
      f"sans atteindre le seuil (p&nbsp;=&nbsp;{num(tous.p_valeur, 3)}).", "note")
    P("Crédits et stylométrie convergent donc vers une lecture sobre. web7 est un "
      "co-auteur réel d'une partie des titres de Ziak, et sa contribution laisse une "
      "trace mesurable lorsqu'on la cherche là où elle est créditée. Rien, en "
      "revanche, ne soutient la version forte de la rumeur&nbsp;: sur 2020-2024, ni les "
      "crédits ni les textes n'indiquent que web7 écrive l'essentiel des textes de "
      "Ziak, et rien ne concerne le masque ni l'identité de l'interprète. La leçon de "
      "méthode est nette&nbsp;: une attribution menée au niveau de l'artiste ne voit pas "
      "un co-auteur minoritaire, qu'un contraste ciblé à l'intérieur de l'œuvre permet "
      "de détecter.")

    # =====================================================================
    P("10. Portrait stylométrique de Ziak", "h1")
    P("À défaut d'identifier Ziak, on peut le caractériser. Deux mesures doivent "
      "être distinguées&nbsp;: la distance médiane à l'ensemble du corpus, qui dit "
      "s'il est atypique&nbsp;; et la distance à son plus proche voisin, qui dit "
      "s'il a un parent.")
    P("Ziak n'a rien d'un excentrique&nbsp;: il se situe au rang "
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
      "dépasse à peine ce que produit une coupe aléatoire de son œuvre "
      f"(z&nbsp;=&nbsp;+{num(stab['z_vs_moitiés_aleatoires'])}, sous le seuil usuel "
      "de 2)&nbsp;: son style évolue un peu, sans rupture.")

    doc.figure("06_1_profil_ziak.png",
           "Portrait stylométrique. (a) marqueurs lexicaux&nbsp;; (b) Ziak n'a pas de "
           "proche parent&nbsp;; (c) l'élision, un faux marqueur qui vient du "
           "transcripteur et non de l'auteur.")

    # =====================================================================
    P("11. Ce que l'étude ne peut pas dire", "h1")
    P("<b>Le corpus n'est pas le rap français.</b> LRFAF couvre 596 artistes, et "
      "l'analyse n'en retient que 393 — ceux disposant d'au moins 12&nbsp;000 mots. "
      "Un artiste peu documenté, ou absent de Genius, ne pouvait pas être détecté. "
      "Les sections 8 et 9 lèvent ce point pour les deux candidats qui comptaient "
      "vraiment, mais il en reste d'autres, hors corpus et non testés.")
    P("<b>Les featurings ne sont séparés que pour les données collectées.</b> Les "
      "paroles de LRFAF ne comportent aucune balise de section&nbsp;: le couplet d'un "
      "invité y est attribué à l'artiste principal. Ce bruit affecte Ziak comme les "
      "candidats, et tend à rapprocher artificiellement les artistes qui collaborent "
      "— donc à faciliter une détection, non à l'empêcher. Pour les textes collectés "
      "directement sur Genius (section 9), les couplets d'invités ont été retirés.")
    P("<b>Les transcriptions sont médiées.</b> Comme le montre le cas de l'élision, "
      "une partie du signal apparent vient des contributeurs de Genius plutôt que "
      "des artistes. Les 4-grammes de caractères y sont moins sensibles que les "
      "mots, sans y être immunisés.")
    P(f"<b>Le corpus de Mikeysem est mince et partiel</b>&nbsp;: {mots(MIKE_MOTS)} "
      f"mots, couvrant {int(disco.page_genius.sum())} de ses {len(disco)} titres. "
      "C'est la conclusion la moins solidement étayée de ce travail.")
    P("<b>La puissance dépend de ce que l'on cherche.</b> La section 7 l'a mesurée "
      "sur des liens réels plutôt que simulés&nbsp;: elle chute nettement quand "
      "l'auteur recherché ne signe qu'une partie des textes (un membre parmi huit se "
      "perd au-delà du centième rang), mais résiste à un changement d'identité "
      "revendiqué. La conclusion «&nbsp;Ziak n'est personne du corpus&nbsp;» vaut "
      "donc pour un alias qui écrirait seul&nbsp;; elle serait plus fragile s'il "
      "s'agissait d'une participation diluée dans un collectif.")
    P("<b>Un auteur non crédité reste difficile à voir.</b> La section 9 le "
      "montre&nbsp;: un co-auteur présent sur une minorité de titres échappe à "
      "l'analyse menée au niveau de l'artiste, et ne se détecte que par un contraste "
      "ciblé — à condition de savoir où chercher, c'est-à-dire de disposer des crédits. "
      "Un auteur de l'ombre non crédité, écrivant dans la voix d'un autre, pourrait "
      "n'être détecté par aucune des deux approches.")

    # =====================================================================
    P("12. Conclusion", "h1")
    P("L'hypothèse du pseudonyme était testable, et le test est concluant&nbsp;: "
      "aucun des 392 autres artistes éligibles du corpus LRFAF ne présente la "
      "signature stylométrique de Ziak, alors qu'un protocole validé sur 177 cas "
      "connus retrouve le bon auteur neuf fois sur dix — et dans 97,7&nbsp;% des cas "
      "pour sa génération. Les quatre analyses convergent vers l'hypothèse d'un "
      "auteur absent du corpus, et le test par paire montre que même le meilleur "
      "candidat reste en deçà de 97,5&nbsp;% des alias authentiques.")
    P("Les deux noms que la rumeur avance le plus souvent ont été testés "
      f"séparément. <b>Mikeysem</b>, absent du corpus et collecté pour l'occasion, se "
      f"classe {mrang}<super>e</super> sur {mcand}, derrière six artistes que "
      "personne ne soupçonne, et son score au test par paire dépasse à peine le "
      "hasard&nbsp;; la minceur de son corpus interdit toutefois d'en faire plus "
      "qu'un faisceau convergent. <b>web7</b>, ex-7 Jaws, appelait une réponse plus "
      "nuancée&nbsp;: il n'apparaît jamais parmi les proches de Ziak, quelle que soit "
      f"la variante, mais Genius le crédite co-auteur de {n_cred} des {n_ziak} titres, "
      "et, sur l'album <i>Essonne History X</i>, les titres qu'il co-signe sont "
      "stylométriquement plus proches de son écriture que les autres "
      f"(p&nbsp;=&nbsp;{num(e1.p_valeur, 2)}). La version forte de la rumeur ne tient "
      "pas&nbsp;; une co-écriture ponctuelle, elle, est documentée et mesurable.")
    P("Cette puissance a été éprouvée sur des liens d'auteur <b>réels</b> et non plus "
      "simulés (section 7)&nbsp;: la méthode retrouve Joke → Ateyaba au premier rang "
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
      "vérifiable n'étaye. La section 9 s'appuie en outre sur des crédits d'auteur "
      "publics&nbsp;: ils documentent une co-écriture, non une identité, et ne disent "
      "rien de la personne qui porte le masque. Le résultat de cette enquête est, à "
      "sa manière, une confirmation de la robustesse de cet anonymat face aux "
      "méthodes quantitatives.", "note")

    P("Lexique", "h1")
    doc.tableau(
        [["Terme", "Ce qu'il désigne"],
         ["Stylométrie",
          "Mesure statistique du style d'écriture, fondée sur des traits involontaires "
          "(petits mots, enchaînements de lettres) plutôt que sur le sens."],
         ["4-gramme de caractères",
          "Suite de quatre caractères consécutifs, espaces compris&nbsp;: "
          "«&nbsp;j'ai&nbsp;», «&nbsp;ai l&nbsp;»… Compter leurs fréquences capte la "
          "musique de la phrase sans passer par les mots."],
         ["Mot-outil",
          "Mot grammatical sans contenu propre («&nbsp;de&nbsp;», «&nbsp;que&nbsp;», "
          "«&nbsp;mais&nbsp;»). Employé sans y penser, donc difficile à maquiller."],
         ["Delta<br/>(de Burrows, Cosine)",
          "Façon de mesurer l'écart entre deux textes une fois chaque trait ramené à "
          "une échelle commune, pour qu'un trait fréquent ne pèse pas plus lourd "
          "qu'un trait rare."],
         ["Score de séparation",
          "De combien le premier du classement devance les autres, exprimé en "
          "écarts-types. Répond à «&nbsp;se détache-t-il&nbsp;?&nbsp;» et non à "
          "«&nbsp;qui est premier&nbsp;?&nbsp;»."],
         ["H<sub>1</sub> / H<sub>0</sub>",
          "Les deux situations de référence&nbsp;: le bon auteur est dans la liste "
          "(H<sub>1</sub>), ou il n'y est pas (H<sub>0</sub>). On compare Ziak à "
          "l'une et à l'autre."],
         ["Puissance",
          "Part des cas où la méthode retrouve le bon auteur quand on connaît déjà la "
          "réponse. Sans elle, un échec ne veut rien dire."],
         ["Test des imposteurs",
          "Au lieu de classer tout le monde, on fait affronter un candidat précis à "
          "des inconnus tirés au sort&nbsp;; son score est la part des duels gagnés."],
         ["p-valeur",
          "Probabilité d'observer un résultat au moins aussi net par pure "
          "coïncidence. Plus elle est basse, moins le hasard suffit à expliquer ce "
          "que l'on voit&nbsp;; 5&nbsp;% est le seuil d'usage."],
         ["Ad-lib",
          "Interjection lâchée par l'interprète («&nbsp;uh&nbsp;», "
          "«&nbsp;gang&nbsp;»), notée entre parenthèses par Genius. Elle vient de la "
          "voix, pas du parolier."],
         ["Featuring",
          "Couplet d'un artiste invité. Il est écrit par l'invité, mais reste attribué "
          "au propriétaire du morceau dans les paroles publiées."],
         ["Ghostwriting",
          "Écriture par un tiers non visible. Contrairement à l'alias, l'auteur "
          "adapte sa plume à la voix de l'interprète, ce qui dilue sa signature."]],
        "Les termes techniques employés dans l'article, dans l'ordre où ils "
        "apparaissent.",
        widths=[3.8, 12.6], gauche=(0, 1))

    P("Reproductibilité", "h2")
    P("L'ensemble des analyses est reproductible depuis le dépôt du projet. Les "
      "scripts s'exécutent dans l'ordre&nbsp;: construction du cache de compteurs, "
      "diagnostic du biais de taille, validation du protocole, application à Ziak, "
      "robustesse, portrait, collecte de Mikeysem et test de l'hypothèse, "
      "validation sur liens réels, collecte de web7 et de Ziak sur Genius, tests de "
      "l'hypothèse web7, figures. "
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
        "Générations. «&nbsp;Ziak&nbsp;: 7 Jaws/Web 7 se cache-t-il sous le "
        "masque&nbsp;? Sa réponse cash&nbsp;!&nbsp;» <i>generations.fr</i>.",
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
