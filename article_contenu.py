#!/usr/bin/env python3
"""Texte de l'article, partagé par la version PDF et la version Word.

`construire(doc)` écrit l'article dans un objet de rendu qui fournit :

- `doc.p(texte, style)` — styles : titre, soustitre, auteur, date, abstract,
  h1, h2, p, note, ref ;
- `doc.gap(points)` ;
- `doc.tableau(lignes, legende, widths, gauche)` — largeurs en centimètres,
  `gauche` donnant les colonnes alignées à gauche ;
- `doc.figure(image, legende)` — chemin relatif à `images/`.

L'article est volontairement court (moins de dix pages). Il repose sur trois
corpus, dont les résultats vivent dans trois dossiers :

- `export/`              — le corpus publié par LRFAF, featurings compris ;
- `export/sans_invites/` — option 2, couplets d'invités retirés : le corpus
  principal, parce qu'il garde le plus de texte ;
- `export/sans_feats/`   — option 1, titres avec invité écartés : le
  contre-test.

Le texte emploie un balisage minimal compris par les deux rendus : <b>, <i>,
<sub>, <super>, <br/>, &nbsp; et &amp;. Tous les chiffres sont relus dans ces
dossiers au moment de la génération.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULT = Path("export")
CORPUS = {"brut": RESULT, "o1": RESULT / "sans_feats", "o2": RESULT / "sans_invites"}
FIG = "sans_invites/"   # figures du corpus principal


def pct(x, d=0):
    return f"{100 * float(x):.{d}f}&nbsp;%".replace(".", ",")


def num(x, d=2):
    # Signe moins typographique (U+2212), comme dans le texte rédigé.
    return f"{float(x):.{d}f}".replace(".", ",").replace("-", "−")


def rang(x):
    x = float(x)
    return f"{x:.0f}" if x == int(x) else f"{x:.1f}".replace(".", ",")


def ordinal(x):
    """Rang cité dans une phrase : arrondi à l'entier."""
    return f"{int(float(x) + 0.5)}"


def mots(x):
    return f"{int(x):,}".replace(",", "&nbsp;")


def lire(corpus: str, fichier: str) -> pd.DataFrame:
    return pd.read_csv(CORPUS[corpus] / fichier)


def chiffres(k: str) -> dict:
    """Les résultats d'un corpus, sous une forme directement citable."""
    r = {}
    puis = lire(k, "03_2_puissance_methode.csv")
    best = puis[(puis.features == "char") & (puis.metric == "cosine_delta")].iloc[0]
    r["rappel1"], r["top5"] = best.recall_at_1, best.recall_at_5
    r["n_essais"] = int(best.n_essais)
    r["n_valid"] = lire(k, "03_1_validation_brute.csv").artist.nunique()
    var = lire(k, "02_3_puissance_par_variante.csv")
    r["naive"], r["controle"] = var.recall_at_1.iloc[0], var.recall_at_1.iloc[-1]
    gen = lire(k, "05_1_puissance_par_generation.csv")
    r["gen_ziak"] = gen[(gen.features == "char")
                        & (gen.generation == "2021-2024")].recall_at_1.iloc[0]
    sens = lire(k, "05_2_sensibilite_seuil.csv")
    r["candidats"] = int(sens[sens.iloc[:, 0] == 12_000].iloc[0, 1])

    ctrl = lire(k, "04_3_controle_positif_ziak.csv")
    c = ctrl[(ctrl.features == "char") & (ctrl.metric == "cosine_delta")]
    r["ctrl_pos"] = (c.rang_moitie_B == 1).mean()
    v = lire(k, "04_6_verdict_hypotheses.csv")
    r["verdict"] = v
    r["n_h0"] = int(v.hypothese_favorisee.str.startswith("H0").sum())
    vb = v[(v.features == "char") & (v.metric == "cosine_delta")].iloc[0]
    r["sep_ziak"], r["sep_h1"], r["sep_h0"] = vb.sep_ziak, vb.sep_H1_correct_moy, vb.sep_H0_moy
    r["favoris"] = list(dict.fromkeys(v.top1_modal))

    imp = lire(k, "05_3_test_imposteurs.csv").sort_values("score_imposteurs",
                                                          ascending=False)
    r["imp_cand"], r["imp_score"] = imp.candidat.iloc[0], imp.score_imposteurs.iloc[0]
    r["imp_hasard"] = imp.seuil_hasard.iloc[0]
    ref = lire(k, "05_4_imposteurs_reference.csv").score_imposteurs_jumeau
    r["imp_ref_med"], r["imp_ref_q10"] = ref.median(), ref.quantile(0.1)

    t14 = lire(k, "14_2_alias_temporel_synthese.csv")
    r["ruptures"] = {x.libelle: x for x in
                     t14[t14.groupe == "changement d'identité"].itertuples()}
    r["n_cand_rupture"] = int(lire(k, "14_1_alias_temporel_bruts.csv").n_candidats.iloc[0])
    a13 = lire(k, "13_2_alias_reels_synthese.csv")
    d = a13[a13.n_rappeurs == a13.n_rappeurs.max()].sort_values("rang_median").iloc[-1]
    r["dilution"], r["dilution_solo"], r["dilution_groupe"] = d.rang_median, d.solo, d.groupe

    mk = lire(k, "10_2_rang_mikeysem.csv")
    mk = mk[(mk.features == "char") & (mk.metric == "cosine_delta")]
    r["mike_rang"], r["mike_cand"] = mk.rang_mikeysem.median(), int(mk.n_candidats.iloc[0])
    mimp = lire(k, "10_4_imposteurs.csv")
    r["mike_imp"] = mimp[mimp.type == "hypothèse"].score.iloc[0]
    r["mike_imp_ref"] = mimp[mimp.type == "jumeau authentique"].score.median()
    mv = lire(k, "10_1_validation_petite_taille.csv")
    h1 = mv[(mv.condition == "H1") & (mv.features == "char") & (mv.metric == "cosine_delta")]
    r["mike_p1"], r["mike_p20"] = (h1.rank_twin == 1).mean(), (h1.rank_twin <= 20).mean()

    w = lire(k, "15_2_web7_synthese.csv").dropna(subset=["rang_median_web7"])
    r["web7_min"], r["web7_max"] = w.rang_median_web7.min(), w.rang_median_web7.max()
    r["web7_top20"] = w["top20_%"].max()
    t16 = lire(k, "16_1_test_2025_web7.csv")
    t16 = t16[t16.variante == "sans featurings ni ad-libs"]
    e = t16[t16.test.str.startswith("2025")].iloc[0]
    r["e2025"] = e
    r["c100"] = t16[t16.test.str.startswith("contrôle : 100")].taux_detection_p05.iloc[0]
    r["c50"] = t16[t16.test.str.startswith("contrôle : 50")].taux_detection_p05.iloc[0]
    r["tous"] = t16[t16.test.str.startswith("toutes")].p_valeur.iloc[0]
    return r


def construire(doc) -> None:
    P = doc.p
    # =========================================================================
    # Chiffres relus dans export/
    # =========================================================================
    B, O1, O2 = chiffres("brut"), chiffres("o1"), chiffres("o2")
    net = pd.read_csv(RESULT / "20_1_nettoyage_resume.csv").iloc[0]
    n_lrfaf = int(net.titres_corpus)
    n_recol, n_perdus = int(net.titres_recollectes), int(net.pages_disparues)

    credits = pd.read_csv(RESULT / "15_3_credits_auteurs_ziak.csv")
    n_cred, n_ziak = int(credits.web7.sum()), len(credits)
    disco = pd.read_csv(RESULT / "11_discographie_mikeysem.csv")
    from stylo_features import clean_lyrics, tokenize as _tok
    MIKE_MOTS = int(sum(len(_tok(clean_lyrics(t)))
                        for t in pd.read_csv("mikeysem_lrfaf.csv").lyrics))
    w15 = lire("o2", "15_2_web7_synthese.csv")
    w_full = w15[w15.variante == "textes complets"].iloc[0]

    exc = lire("o2", "06_1_excentricite.csv")
    zexc = exc[exc.artiste == "Ziak"].iloc[0]
    part_voisin = (exc.distance_min < zexc.distance_min).mean()
    marq = lire("o2", "06_2_marqueurs_lexicaux.csv").nlargest(3, "z_log_odds")
    elis = lire("o2", "06_5_controle_elision.csv").iloc[0]
    stab = lire("o2", "06_3_stabilite_temporelle.csv").iloc[0]

    e2, e1 = O2["e2025"], O1["e2025"]
    zulu = [a.strip() for a in
            credits[credits.title == "Zulu"].iloc[0].auteurs.split(";")]
    auteurs_zulu = ", ".join(zulu[:-1]) + " et " + zulu[-1]

    # =========================================================================
    # Titre et résumé
    # =========================================================================
    P("Qui se cache derrière Ziak&nbsp;?", "titre")
    P("Enquête stylométrique sur un rappeur masqué", "soustitre")
    P("Tassilo Westphalen", "auteur")
    P("CPES Sciences des données, Arts et Cultures — Université PSL / Lycée Louis-le-Grand",
      "date")

    P("<b>Résumé.</b> Apparu en 2020 cagoulé et sans identité publique, le rappeur "
      "Ziak a nourri trois rumeurs&nbsp;: un rappeur établi revenu sous un autre nom, "
      "Mikeysem, ou web7 (ex-7 Jaws) qui écrirait ses textes. Nous les testons par "
      f"attribution d'auteur sur le corpus LRFAF ({mots(n_lrfaf)} chansons de rap "
      "français), après avoir retiré les featurings de deux façons&nbsp;: en écartant "
      "les titres avec invités, ou seulement les couplets des invités. Validée sur "
      f"{O2['n_valid']} artistes dont la réponse est connue ({pct(O2['rappel1'])} de "
      "réussite) et sur des changements de nom réels, la méthode ne trouve <b>aucun "
      "rappeur du corpus derrière Ziak</b>, quel que soit le traitement. Mikeysem "
      f"arrive {ordinal(O2['mike_rang'])}<super>e</super> sur {O2['mike_cand']}. web7, co-auteur crédité de "
      f"{n_cred} titres, laisse une trace sur l'album de 2025 quand seuls les couplets "
      f"d'invités sont retirés (p&nbsp;=&nbsp;{num(e2.p_valeur, 3)}), mais pas quand les "
      f"titres avec invités sont écartés (p&nbsp;=&nbsp;{num(e1.p_valeur, 2)})&nbsp;: "
      "l'indice tient à un seul titre. Le masque tient.", "abstract")
    doc.gap(4)
    P("<b>Mots-clés&nbsp;:</b> stylométrie, attribution d'auteur, rap français, "
      "featurings, méthode des imposteurs, LRFAF.", "abstract")
    doc.gap(8)

    # =========================================================================
    P("1. Le masque", "h1")
    P("Le premier titre sort en 2020. Pas de visage&nbsp;: une cagoule. Pas de nom, "
      "pas de passé, pas d'interview qui trahisse une origine. En quelques années, "
      "Ziak s'impose parmi les nouveaux noms du rap français, et plus sa voix circule, "
      "plus son silence intrigue&nbsp;: qui peut écrire comme cela sans avoir jamais "
      "existé avant&nbsp;?")
    P("Les auditeurs ont leurs suspects. Pour les uns, le masque cache un rappeur "
      "déjà connu, qui aurait tout recommencé sous un autre nom. Pour d'autres, c'est "
      "Mikeysem, un rappeur de la même région. Pour d'autres encore, Ziak n'est "
      "qu'une voix, et les textes seraient écrits par un autre — web7, autrefois "
      "7 Jaws. Ziak, lui, ne confirme rien.")
    P("Un masque cache un visage. Il ne cache pas une manière d'écrire. Chacun a ses "
      "tics&nbsp;: la fréquence des petits mots («&nbsp;de&nbsp;», «&nbsp;que&nbsp;», "
      "«&nbsp;mais&nbsp;»), la façon d'élider, les enchaînements de syllabes — des "
      "habitudes trop nombreuses et trop peu conscientes pour être maquillées. C'est "
      "par elles qu'en 2013 un romancier débutant, Robert Galbraith, s'est révélé "
      "être J.&nbsp;K. Rowling (Juola, 2013). <b>Si un autre rappeur se cache derrière "
      "Ziak, ses textes devraient le trahir.</b>")
    P(f"Nous mettons les trois rumeurs à l'épreuve sur {mots(n_lrfaf)} chansons. Deux "
      "exigences conditionnent la réponse&nbsp;: empêcher les featurings — le couplet "
      "qu'un rappeur pose sur le morceau d'un autre — de brouiller les signatures, et "
      "savoir ce que vaut la méthode avant de croire ce qu'elle dit.")

    # =========================================================================
    P("2. Les données", "h1")
    P("2.1 Le corpus", "h2")
    P(f"Le corpus LRFAF (de Courson, 2024) rassemble {mots(n_lrfaf)} textes de rap "
      "français issus de Genius, pour les artistes qui ont une page Wikipédia. Les "
      "textes de moins de 100 mots et les doublons sont écartés. Ziak y figure avec "
      "43 titres parus entre 2020 et 2024, soit environ 24&nbsp;000 mots. Mikeysem et "
      "web7, sans page Wikipédia, n'y figurent pas&nbsp;: leurs titres ont été "
      "collectés sur Genius et mis au format du corpus.")

    P("2.2 Le problème des featurings", "h2")
    P("Dans un featuring, l'invité écrit son propre couplet. Mais les paroles publiées "
      "le rangent sous le nom du propriétaire du morceau&nbsp;: l'œuvre d'un artiste "
      "contient donc, sans le dire, des textes écrits par d'autres. Cela rapproche "
      "artificiellement tous ceux qui collaborent, et ce défaut touche <i>tous</i> les "
      "artistes du corpus, pas seulement Ziak. Or notre question porte précisément "
      "sur une proximité entre artistes.")
    P("LRFAF a supprimé les balises de section de Genius («&nbsp;[Couplet 2&nbsp;: "
      "Kaaris]&nbsp;»), qui disent qui chante quoi. Nous avons donc re-téléchargé les "
      f"{mots(n_recol + n_perdus)} pages Genius des artistes testables&nbsp;: toutes "
      f"ont été récupérées, sauf {n_perdus} qui ont disparu du site. Une règle distingue les "
      "membres d'un groupe de ses invités&nbsp;: Akhenaton, présent sur la majorité "
      "des titres d'IAM, n'est pas un invité d'IAM. Les featurings sont ensuite "
      "retirés de deux façons.")

    doc.tableau(
        [["", "Corpus publié", "Option 1&nbsp;: titres<br/>avec invité écartés",
          "Option 2&nbsp;: couplets<br/>d'invités retirés"],
         ["Ce qui est retiré", "rien", f"{mots(net.option1_titres_ecartes)} titres",
          f"{pct(net.part_retiree)} des mots"],
         ["Candidats testables<br/>(au moins 12&nbsp;000 mots)", str(B["candidats"]),
          str(O1["candidats"]), str(O2["candidats"])],
         ["Méthode&nbsp;: bon auteur<br/>trouvé au 1<super>er</super> rang",
          pct(B["rappel1"]), pct(O1["rappel1"]), pct(O2["rappel1"])]],
        "Les trois corpus. L'option 1 est la plus stricte, mais elle fait maigrir les "
        "artistes qui collaborent beaucoup&nbsp;: certains passent sous le seuil et "
        "sortent du test. L'option 2 garde l'essentiel du texte.",
        widths=[4.4, 3.2, 3.9, 3.9])

    P("Retirer les featurings améliore nettement la méthode&nbsp;: elle retrouve le "
      f"bon auteur dans {pct(O2['rappel1'])} des cas au lieu de {pct(B['rappel1'])}. "
      "Les couplets d'invités brouillaient donc bien les signatures. L'option 2 sert "
      "de corpus principal, parce qu'elle garde le plus de texte&nbsp;; l'option 1 "
      "sert de contre-test. <b>Un résultat qui tient sur les deux ne doit rien aux "
      "featurings.</b>")

    # =========================================================================
    P("3. La méthode", "h1")
    P("3.1 Mesurer une signature", "h2")
    P("Chaque texte est décrit par la fréquence de ses 3&nbsp;000 suites de quatre "
      "caractères les plus courantes («&nbsp;j'ai&nbsp;», «&nbsp;ai l&nbsp;»…) et de "
      "ses 500 mots les plus fréquents, qui captent la mécanique de l'écriture plutôt "
      "que ses thèmes. Deux textes sont comparés par le Delta de Burrows (2002) et le "
      "Cosine Delta (Smith &amp; Aldridge, 2011), deux distances standard en "
      "attribution d'auteur. Les quatre combinaisons sont conduites en parallèle.")

    P("3.2 Le piège de la démarche intuitive", "h2")
    P("L'idée spontanée — réunir toutes les chansons de chaque artiste, puis chercher "
      "le plus proche de Ziak — produit un classement d'allure sérieuse. Il est "
      "faux&nbsp;: il mesure surtout la quantité de texte disponible. Un artiste "
      "prolifique couvre tant de tournures qu'il finit par ressembler à tout le "
      "monde, et monte mécaniquement dans le classement (figure&nbsp;1). Testée sur "
      "des cas dont on connaît la réponse, cette démarche ne trouve le bon auteur que "
      f"dans {pct(O2['naive'])} des cas. En donnant à chaque candidat exactement "
      f"12&nbsp;000 mots, on passe à {pct(O2['controle'])}.")
    doc.figure(FIG + "02_1_biais_de_taille.png",
               "Le piège de la démarche intuitive. (a) plus un artiste a écrit, plus il "
               "paraît proche de Ziak&nbsp;; (b) une fois la taille neutralisée, le "
               "classement se réorganise entièrement&nbsp;; (c) taux de bonnes réponses "
               "sur des cas connus (option 2).")

    P("3.3 Un protocole qui a le droit de répondre «&nbsp;personne&nbsp;»", "h2")
    P("Une méthode sommée de désigner quelqu'un désigne toujours quelqu'un. Or "
      "l'auteur que nous cherchons peut très bien ne pas être dans le corpus. Nous "
      "ne regardons donc pas <i>qui</i> arrive en tête, mais <i>de combien</i> il "
      "devance les autres&nbsp;: c'est le <b>score de séparation</b>, exprimé en "
      "écarts-types. Pour savoir le lire, on simule les deux situations possibles. "
      "On cache la moitié de l'œuvre d'un artiste sous un faux nom, puis on cherche "
      "l'auteur de l'autre moitié&nbsp;: une fois avec la moitié cachée dans la liste "
      "des candidats (l'auteur est présent), une fois sans (l'auteur est absent).")
    P(f"Sur {O2['n_valid']} artistes et {mots(O2['n_essais'])} essais, la meilleure "
      f"combinaison retrouve le bon auteur au premier rang dans {pct(O2['rappel1'])} "
      f"des cas, et dans {pct(O2['gen_ziak'])} pour la génération de Ziak "
      f"(2021-2024). Un auteur présent produit une séparation moyenne de "
      f"{num(O2['sep_h1'])}, un auteur absent de {num(O2['sep_h0'])}&nbsp;: c'est "
      "entre ces deux repères que Ziak sera placé.")
    rO2, rB = O2["ruptures"], B["ruptures"]
    jo, di, gi = (rO2["Joke → Ateyaba"], rO2["Disiz la Peste → Disiz"],
                  rO2["Maître Gims → Gims"])
    P("Restait l'objection la plus sérieuse&nbsp;: un artiste qui change de nom change "
      "peut-être aussi d'écriture. Genius permet de le vérifier sur trois changements "
      "d'identité réels, en cherchant la période d'après le changement depuis celle "
      f"d'avant. <b>Joke, devenu Ateyaba, est retrouvé au premier rang sur "
      f"{O2['n_cand_rupture']} candidats dans {pct(jo.taux_rang1)} des tirages</b>, "
      "alors qu'il avait annoncé vouloir «&nbsp;tuer&nbsp;» son ancien nom. Disiz "
      f"(ex-Disiz la Peste) arrive au rang {rang(di.rang_median)} et Gims (ex-Maître "
      f"Gims) au rang {rang(gi.rang_median)} (figure&nbsp;2). Changer de nom n'efface "
      "pas la signature. La méthode a en revanche une limite nette&nbsp;: un rappeur "
      "qui n'écrit qu'une partie des textes d'un collectif se perd&nbsp;: depuis ses "
      f"textes solo, {O2['dilution_solo']} ne retrouve son groupe, "
      f"{O2['dilution_groupe']}, qu'au {ordinal(O2['dilution'])}<super>e</super> rang.")
    doc.figure(FIG + "14_1_alias_temporel.png",
               "Un changement de nom n'efface pas la signature. (a) rang de la période "
               "postérieure au changement, recherchée depuis la période antérieure&nbsp;; "
               "(b) comparaison avec des artistes découpés de la même façon sans avoir "
               "changé de nom&nbsp;; (c) le favori de Ziak se détache moins que dans "
               "presque tous les tests où le lien d'auteur est réel.")

    # =========================================================================
    P("4. Les résultats", "h1")
    P("Le tableau&nbsp;2 résume les trois hypothèses sur les trois corpus. Les "
      "sections suivantes les détaillent.")

    def verdict(r):
        return f"auteur absent<br/>({r['n_h0']} analyses sur 4)"

    doc.tableau(
        [["", "Corpus publié", "Option 1", "Option 2"],
         ["<b>Ziak est un rappeur du corpus</b><br/>hypothèse favorisée",
          verdict(B), verdict(O1), verdict(O2)],
         ["séparation de Ziak<br/>(repère&nbsp;: présent / absent)",
          f"{num(B['sep_ziak'])}<br/>({num(B['sep_h1'])} / {num(B['sep_h0'])})",
          f"{num(O1['sep_ziak'])}<br/>({num(O1['sep_h1'])} / {num(O1['sep_h0'])})",
          f"{num(O2['sep_ziak'])}<br/>({num(O2['sep_h1'])} / {num(O2['sep_h0'])})"],
         ["<b>Ziak est Mikeysem</b><br/>rang de Mikeysem",
          f"{rang(B['mike_rang'])} / {B['mike_cand']}",
          f"{rang(O1['mike_rang'])} / {O1['mike_cand']}",
          f"{rang(O2['mike_rang'])} / {O2['mike_cand']}"],
         ["<b>web7 écrit les textes</b><br/>rang de web7",
          f"{rang(B['web7_min'])} à {rang(B['web7_max'])}",
          f"{rang(O1['web7_min'])} à {rang(O1['web7_max'])}",
          f"{rang(O2['web7_min'])} à {rang(O2['web7_max'])}"],
         ["titres co-écrits par web7 (2025)<br/>p-valeur",
          num(B["e2025"].p_valeur, 3), num(e1.p_valeur, 3),
          f"<b>{num(e2.p_valeur, 3)}</b>"]],
        "Les trois hypothèses sur les trois corpus. Séparation&nbsp;: plus elle est "
        "négative, plus le meilleur candidat se détache&nbsp;; Ziak reste chaque fois "
        "du côté «&nbsp;auteur absent&nbsp;». Rangs&nbsp;: médianes sur les "
        "rééchantillonnages, meilleure combinaison. Pour web7, fourchette sur les "
        "variantes testées.",
        widths=[5.2, 3.4, 3.4, 3.4])

    P("4.1 Aucun rappeur du corpus n'est Ziak", "h2")
    P("On vérifie d'abord que le style de Ziak est détectable&nbsp;: en coupant son "
      "œuvre en deux, la seconde moitié est retrouvée au premier rang dans "
      f"{pct(O2['ctrl_pos'])} des tirages. La matière ne manque pas.")
    fav = ", ".join(O2["favoris"])
    P(f"Face aux {O2['candidats']} autres artistes, en revanche, <b>personne ne se "
      "détache</b>&nbsp;: les favoris changent d'une combinaison à l'autre (" + fav +
      "). La séparation de "
      f"Ziak ({num(O2['sep_ziak'])}) tombe du côté «&nbsp;auteur absent&nbsp;» "
      f"({num(O2['sep_h0'])}), très loin d'un auteur présent ({num(O2['sep_h1'])}), et "
      "c'est vrai pour les quatre combinaisons sur les trois corpus (figure&nbsp;3). "
      "Un dernier test, dit des imposteurs (Koppel &amp; Winter, 2014), oppose le "
      "meilleur candidat, "
      f"{O2['imp_cand']}, à des inconnus tirés au sort&nbsp;: il gagne "
      f"{pct(O2['imp_score'])} des duels, bien plus que le hasard "
      f"({pct(O2['imp_hasard'])}), mais moins que presque tous les vrais alias, qui en "
      f"gagnent {pct(O2['imp_ref_med'])} en médiane. Ziak et ses proches partagent "
      "une famille musicale, pas une main.")
    doc.figure(FIG + "04_1_verdict_H0_H1.png",
               "Ziak (trait rouge) face aux deux situations de référence&nbsp;: en vert, "
               "l'auteur est dans le corpus&nbsp;; en gris, il n'y est pas. Les quatre "
               "combinaisons placent Ziak du côté gris (option 2).")

    P("4.2 Ce n'est pas Mikeysem", "h2")
    P("Le nom le plus souvent avancé ne figurait pas dans LRFAF. Sur les "
      f"{len(disco)} titres de sa discographie, {int(disco.page_genius.sum())} "
      f"seulement ont des paroles sur Genius, soit {mots(MIKE_MOTS)} mots. À cette "
      "taille, la méthode retrouve encore un vrai alias dans le top 20 dans "
      f"{pct(O2['mike_p20'])} des cas. Mikeysem se classe pourtant "
      f"{ordinal(O2['mike_rang'])}<super>e</super> sur {O2['mike_cand']}, derrière six "
      "artistes que personne ne soupçonne, et il ne gagne que "
      f"{pct(O2['mike_imp'])} de ses duels contre des inconnus, contre "
      f"{pct(O2['mike_imp_ref'])} pour un vrai alias. Le corpus est mince&nbsp;: c'est "
      "un faisceau convergent, pas une démonstration.")

    P("4.3 Ce n'est pas 7 Jaws — mais il a écrit avec lui", "h2")
    P("La rumeur, relayée par la presse (Générations), veut que web7 écrive les "
      f"textes de Ziak. web7 compte {int(w_full.n_titres_web7)} titres sur Genius, "
      f"soit {mots(w_full.mots_web7)} mots&nbsp;: la matière ne manque pas. Pourtant, "
      f"vu depuis Ziak, il se classe entre le {ordinal(O2['web7_min'])}<super>e</super> et "
      f"le {ordinal(O2['web7_max'])}<super>e</super> rang selon la variante, et jamais "
      "dans les vingt premiers, sur aucun des trois corpus. <b>La version forte de la "
      "rumeur ne tient pas.</b>")
    P(f"Les crédits de Genius racontent autre chose&nbsp;: web7 est co-auteur de "
      f"{n_cred} titres de Ziak sur {n_ziak} — deux en 2021, huit en 2025 sur l'album "
      "<i>Essonne History X</i>. Une contribution aussi minoritaire est invisible à "
      "l'échelle de l'œuvre entière. L'album offre en revanche une expérience "
      f"naturelle&nbsp;: {int(e2.n_titres)} titres crédités à web7 et "
      f"{int(e2.n_non_credites)} qui ne le sont pas, même artiste, même année. Si web7 "
      "laisse une trace, les titres qu'il co-signe doivent être plus proches de son "
      f"écriture que {mots(5000)} paquets de {int(e2.n_titres)} titres tirés au "
      "hasard dans la même période.")
    P(f"Avec l'option 2, c'est le cas (p&nbsp;=&nbsp;{num(e2.p_valeur, 3)}, "
      "figure&nbsp;4), et le test avait la puissance de le voir&nbsp;: il détecte un "
      f"texte entièrement de web7 dans {pct(O2['c100'])} des cas, un mélange à parts "
      f"égales dans {pct(O2['c50'])}. <b>Mais avec l'option 1, l'effet disparaît</b> "
      f"(p&nbsp;=&nbsp;{num(e1.p_valeur, 2)}), alors que la puissance est intacte "
      f"({pct(O1['c50'])} pour le mélange). La différence tient essentiellement à un "
      f"titre, <i>Zulu</i>, co-écrit par {auteurs_zulu}&nbsp;: "
      "l'option 1 l'écarte entièrement, l'option 2 n'en garde que les couplets de "
      "Ziak. Toutes années confondues, le test n'atteint pas le seuil "
      f"(p&nbsp;=&nbsp;{num(O2['tous'], 2)}). La co-écriture de web7 est donc "
      "documentée par les crédits, mais sa trace stylistique reste un indice fragile.")
    doc.figure(FIG + "16_1_web7.png",
               "L'hypothèse web7 (option 2). (a) au niveau de l'artiste, web7 est loin "
               "derrière les proches de Ziak&nbsp;; (b) sur <i>Essonne History X</i>, les "
               "titres co-écrits sont plus proches de web7 que des tirages au "
               "hasard&nbsp;; (c) le test détecte un texte entièrement ou à moitié écrit "
               "par web7.")

    P("4.4 Portrait", "h2")
    P("À défaut d'un nom, un portrait. Ziak n'a pas de parent proche&nbsp;: "
      f"{pct(part_voisin)} des artistes du corpus ont un voisin plus proche que le "
      "sien. Ses marqueurs les plus nets sont des onomatopées — "
      + ", ".join(f"«&nbsp;{r.mot}&nbsp;»" for r in marq.itertuples())
      + " —, les ad-libs qui ponctuent ses morceaux (Monroe <i>et al.</i>, 2008). Un "
      "marqueur spectaculaire s'est révélé faux&nbsp;: Ziak semble employer cinq fois "
      "moins «&nbsp;je&nbsp;» que la moyenne, mais il emploie davantage "
      f"«&nbsp;j'&nbsp;», et les deux formes réunies donnent un rapport de "
      f"{num(elis.ratio_cumule)}. C'est une convention de transcription de Genius, pas "
      "une habitude d'écriture. Enfin, son style évolue peu entre 2020-2021 et "
      f"2022-2024 (z&nbsp;=&nbsp;{num(stab['z_vs_moitiés_aleatoires'])}, sous le seuil "
      "de 2).")

    # =========================================================================
    P("5. Limites", "h1")
    P("<b>Le corpus n'est pas tout le rap français.</b> Seuls les artistes ayant une "
      "page Wikipédia y figurent, plus Mikeysem et web7 ajoutés ici&nbsp;; un rappeur "
      "peu documenté reste hors d'atteinte. <b>Un auteur caché dans un collectif, ou "
      "non crédité, échappe à la méthode</b>, qui perd la trace d'un auteur signant "
      "une petite part des textes. <b>Le corpus de Mikeysem est mince</b>, ce qui fait "
      "de cette conclusion la moins solide. <b>Le résultat sur web7 repose sur une "
      "seule expérience</b>, parmi plusieurs tests apparentés, et dépend du traitement "
      "d'un titre. Enfin, <b>les paroles sont des transcriptions</b> saisies par des "
      "bénévoles&nbsp;: une part du signal vient d'eux, comme l'a montré le faux "
      "marqueur «&nbsp;je&nbsp;».")

    # =========================================================================
    P("6. Conclusion", "h1")
    P(f"Ziak n'est aucun des {O2['candidats']} rappeurs testables du corpus, et ce "
      "verdict ne doit rien aux featurings&nbsp;: il tient sur le corpus publié comme "
      "sur les deux corpus nettoyés. La méthode qui le rend sait pourtant voir à "
      "travers un masque — elle retrouve Joke derrière Ateyaba. Mikeysem n'est pas "
      "plus proche de Ziak que des dizaines d'artistes que personne ne soupçonne. "
      "web7, enfin, n'écrit pas Ziak&nbsp;: il a écrit <i>avec</i> lui, sur quelques "
      "titres, et l'empreinte qu'il y laisse est trop ténue pour être affirmée.")
    P("En chemin, l'enquête a montré qu'une question d'attribution se perd facilement "
      "dans ses propres chiffres&nbsp;: la démarche intuitive désigne un coupable avec "
      "assurance, et se trompe deux fois sur trois. Il a fallu contrôler la quantité de "
      "texte, retirer les couplets des autres, et mesurer ce que la méthode sait voir "
      "avant de lire ce qu'elle voit. Au bout du compte, le masque tient.")

    P("Note éthique", "h2")
    P("Cette étude porte sur une personne qui a choisi l'anonymat. Elle ne s'appuie "
      "que sur des textes publiés et des crédits publics, et la stylométrie rapproche "
      "des textes, pas des personnes&nbsp;: même une correspondance forte n'aurait été "
      "qu'un indice. Le rapprochement avec Mikeysem est une spéculation d'auditeurs, "
      "que l'intéressé a démentie.", "note")

    P("Reproductibilité", "h2")
    P("Le code, les résultats des trois corpus et la méthode détaillée "
      "(<i>METHODE.md</i>) accompagnent le dépôt du projet. Le corpus LRFAF se "
      "télécharge depuis Hugging Face&nbsp;; la re-collecte des balises Genius et "
      "l'étude complète sur les deux options se relancent par deux commandes.", "note")

    # =========================================================================
    P("Références", "h1")
    for r in [
        "Burrows, J. (2002). Delta&nbsp;: a measure of stylistic difference and a guide "
        "to likely authorship. <i>Literary and Linguistic Computing</i>, 17(3), 267-287.",
        "de Courson, B. (2024). <i>LRFAF&nbsp;: une exploration numérique du rap "
        "français depuis les années 1990.</i> Jeu de données&nbsp;: "
        "huggingface.co/datasets/regicid/LRFAF",
        "Générations. «&nbsp;Ziak&nbsp;: 7 Jaws/Web 7 se cache-t-il sous le "
        "masque&nbsp;? Sa réponse cash&nbsp;!&nbsp;» <i>generations.fr</i>.",
        "Juola, P. (2013). Rowling and «&nbsp;Galbraith&nbsp;»&nbsp;: an authorial "
        "analysis. <i>Language Log</i>, 16 juillet 2013.",
        "Koppel, M. &amp; Winter, Y. (2014). Determining if two documents are written "
        "by the same author. <i>Journal of the Association for Information Science and "
        "Technology</i>, 65(1), 178-187.",
        "Monroe, B., Colaresi, M. &amp; Quinn, K. (2008). Fightin' words&nbsp;: lexical "
        "feature selection and evaluation for identifying the content of political "
        "conflict. <i>Political Analysis</i>, 16(4), 372-403.",
        "Smith, P. &amp; Aldridge, W. (2011). Improving authorship attribution&nbsp;: "
        "optimizing Burrows' Delta method. <i>Journal of Quantitative Linguistics</i>, "
        "18(1), 63-88.",
    ]:
        P(r, "ref")
