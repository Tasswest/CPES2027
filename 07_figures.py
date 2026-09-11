#!/usr/bin/env python3
"""Figures de l'article sur l'attribution stylométrique de Ziak."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

IMAGES_DIR = Path("images")
RESULT_DIR = Path("result")
IMAGES_DIR.mkdir(exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

BLEU = "#2f6f9f"
ROUGE = "#c0392b"
GRIS = "#8d98a3"
VERT = "#3d8b6d"
ORANGE = "#d98c3f"


def fig_biais_taille() -> None:
    """Le classement naïf est gouverné par la taille des corpus."""
    comp = pd.read_csv(RESULT_DIR / "02_1_classement_naif_vs_controle.csv")
    perf = pd.read_csv(RESULT_DIR / "02_3_puissance_par_variante.csv")
    # La colonne mêle valeurs numériques et noms d'artistes : pandas la lit
    # comme du texte, il faut reconvertir ce qui est chiffré.
    synth = pd.read_csv(RESULT_DIR / "02_4_synthese_diagnostic.csv",
                        index_col=0).iloc[:, 0]
    synth = pd.to_numeric(synth, errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    ax = axes[0]
    ax.scatter(comp["n_tokens_corpus"] / 1000, comp["distance_naive"],
               s=13, alpha=0.5, c=GRIS, edgecolors="none")
    top20 = comp.nsmallest(20, "distance_naive")
    ax.scatter(top20["n_tokens_corpus"] / 1000, top20["distance_naive"],
               s=26, c=ROUGE, edgecolors="none", label="20 « plus proches » de Ziak")
    ax.set_xscale("log")
    ax.set_xlabel("Taille du corpus de l'artiste (milliers de tokens, échelle log)")
    ax.set_ylabel("Distance à Ziak (approche naïve)")
    ax.set_title(f"a) Ce que mesure l'approche naïve\nρ = {synth['spearman_distance_vs_taille']:.2f} "
                 f"— {synth['part_variance_expliquee_par_taille']:.0%} de la variance")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    sub = comp.nsmallest(12, "distance_naive").iloc[::-1]
    y = np.arange(len(sub))
    ax.hlines(y, sub["rang_naif"], sub["rang_controle"], color=GRIS, lw=1.2, zorder=1)
    ax.scatter(sub["rang_naif"], y, s=34, c=ROUGE, zorder=2, label="rang naïf")
    ax.scatter(sub["rang_controle"], y, s=34, c=BLEU, zorder=2, label="rang à taille contrôlée")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["artiste"], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Rang parmi les candidats (échelle log)")
    ax.set_title("b) Ce que devient ce classement\nune fois la taille neutralisée")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[2]
    libelles = ["A. naïve\n(corpus complets)", "B. taille\ncontrôlée",
                "C. taille contrôlée\n+ Cosine Delta"]
    couleurs = [ROUGE, ORANGE, VERT]
    bars = ax.bar(libelles, perf["recall_at_1"], color=couleurs, width=0.62)
    for b, v in zip(bars, perf["recall_at_1"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontweight="bold", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Taux de bonne identification (rang 1)")
    ax.set_title("c) Puissance mesurée sur vérité-terrain\n(177 artistes dont on connaît la réponse)")
    ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Pourquoi la démarche intuitive désigne le mauvais artiste",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "02_1_biais_de_taille.png", bbox_inches="tight")
    plt.close(fig)


def fig_validation() -> None:
    """Puissance du protocole retenu, par méthode et par génération."""
    val = pd.read_csv(RESULT_DIR / "03_1_validation_brute.csv")
    perf = pd.read_csv(RESULT_DIR / "03_2_puissance_methode.csv")
    gen = pd.read_csv(RESULT_DIR / "05_1_puissance_par_generation.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))

    ax = axes[0]
    h1 = val[(val["condition"] == "H1_jumeau_present")
             & (val["features"] == "char") & (val["metric"] == "cosine_delta")]
    ks = np.arange(1, 21)
    rec = [(h1["rank_twin"] <= k).mean() for k in ks]
    ax.plot(ks, rec, marker="o", ms=3.5, color=BLEU, lw=1.8)
    ax.axhline(1.0, color=GRIS, ls=":", lw=0.9)
    ax.set_ylim(0.85, 1.02)
    ax.set_xlabel("k (rang considéré comme un succès)")
    ax.set_ylabel("Proportion de bonnes réponses")
    ax.set_title(f"a) Courbe de rappel\n4-grammes + Cosine Delta "
                 f"(rappel@1 = {rec[0]:.0%})")
    ax.annotate(f"{rec[0]:.1%}", (1, rec[0]), textcoords="offset points",
                xytext=(9, -11), fontsize=9, color=BLEU, fontweight="bold")

    ax = axes[1]
    p = perf.sort_values("recall_at_1")
    lab = [f"{r['features']}\n{r['metric'].replace('_', ' ')}" for _, r in p.iterrows()]
    cols = [VERT if (r["features"] == "char" and r["metric"] == "cosine_delta") else GRIS
            for _, r in p.iterrows()]
    bars = ax.barh(lab, p["recall_at_1"], color=cols, height=0.6)
    for b, v in zip(bars, p["recall_at_1"]):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Taux de bonne identification (rang 1)")
    ax.set_title("b) Les quatre combinaisons testées")
    ax.tick_params(axis="y", labelsize=8)

    ax = axes[2]
    g = gen[gen["features"] == "char"]
    x = np.arange(len(g))
    cols = [ROUGE if str(s) == "2021-2024" else BLEU for s in g["generation"]]
    bars = ax.bar(x, g["recall_at_1"], color=cols, width=0.62)
    for b, v in zip(bars, g["recall_at_1"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(g["generation"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Taux de bonne identification")
    ax.set_title("c) Puissance par génération d'artistes\n(en rouge : celle de Ziak)")

    fig.suptitle("La méthode sait retrouver un auteur quand la réponse est connue",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "03_1_validation.png", bbox_inches="tight")
    plt.close(fig)


def fig_verdict() -> None:
    """Figure centrale : Ziak confronté aux deux hypothèses."""
    val = pd.read_csv(RESULT_DIR / "03_1_validation_brute.csv")
    seps = pd.read_csv(RESULT_DIR / "04_2_separation_ziak.csv")

    combos = [("char", "cosine_delta"), ("mfw", "cosine_delta"),
              ("char", "burrows_delta"), ("mfw", "burrows_delta")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7))

    for ax, (f, m) in zip(axes.ravel(), combos):
        h1 = val[(val["condition"] == "H1_jumeau_present") & (val["features"] == f)
                 & (val["metric"] == m) & val["top1_is_twin"]]["sep_top1"]
        h0 = val[(val["condition"] == "H0_jumeau_absent") & (val["features"] == f)
                 & (val["metric"] == m)]["sep_top1"]
        zk = seps[(seps["features"] == f) & (seps["metric"] == m)]["sep_top1"]

        grid = np.linspace(min(h1.min(), h0.min()) - 0.5, max(h1.max(), h0.max()) + 0.5, 400)
        for data, col, lab in [
            (h1, VERT, "H1 — l'auteur EST dans le corpus\n(jumeaux authentiques)"),
            (h0, GRIS, "H0 — l'auteur n'est PAS dans le corpus"),
        ]:
            dens = stats.gaussian_kde(data)(grid)
            ax.fill_between(grid, dens, color=col, alpha=0.35)
            ax.plot(grid, dens, color=col, lw=1.5, label=lab)

        ax.axvline(zk.mean(), color=ROUGE, lw=2.2, label=f"Ziak ({zk.mean():.2f})")
        ax.set_xlabel("Score de séparation du meilleur candidat")
        ax.set_ylabel("Densité")
        ax.set_title(f"{f} / {m.replace('_', ' ')}")
        if (f, m) == ("char", "cosine_delta"):
            ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Ziak tombe du côté « auteur absent du corpus », dans les quatre analyses",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "04_1_verdict_H0_H1.png", bbox_inches="tight")
    plt.close(fig)


def fig_candidats() -> None:
    """Instabilité des candidats et test des imposteurs."""
    top5 = pd.read_csv(RESULT_DIR / "04_4_stabilite_top5.csv")
    imp = pd.read_csv(RESULT_DIR / "05_3_test_imposteurs.csv")
    ref = pd.read_csv(RESULT_DIR / "05_4_imposteurs_reference.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    combos = [("char", "cosine_delta"), ("mfw", "cosine_delta"),
              ("char", "burrows_delta"), ("mfw", "burrows_delta")]
    noms = sorted({a for _, r in top5.iterrows() for a in [r["artiste"]]
                   if r["freq_top5"] >= 0.4})
    data = np.zeros((len(noms), len(combos)))
    for j, (f, m) in enumerate(combos):
        sub = top5[(top5["features"] == f) & (top5["metric"] == m)]
        d = dict(zip(sub["artiste"], sub["freq_top5"]))
        for i, n in enumerate(noms):
            data[i, j] = d.get(n, 0.0)

    order = np.argsort(-data.sum(axis=1))
    data, noms = data[order], [noms[i] for i in order]
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels([f"{f}\n{m.replace('_', ' ')}" for f, m in combos], fontsize=7.5)
    ax.set_yticks(range(len(noms)))
    ax.set_yticklabels(noms, fontsize=8)
    for i in range(len(noms)):
        for j in range(len(combos)):
            if data[i, j] > 0:
                ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white" if data[i, j] > 0.55 else "#333")
    ax.set_title("a) Aucun candidat n'est stable d'une méthode à l'autre\n"
                 "(fréquence de présence dans le top-5)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="fréquence")

    ax = axes[1]
    imp = imp.sort_values("score_imposteurs")
    y = np.arange(len(imp))
    ax.barh(y, imp["score_imposteurs"], color=BLEU, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(imp["candidat"], fontsize=8)
    med = ref["score_imposteurs_jumeau"].median()
    d1 = ref["score_imposteurs_jumeau"].quantile(0.1)
    ax.axvline(med, color=VERT, lw=2,
               label=f"jumeaux authentiques : médiane {med:.2f}")
    ax.axvspan(d1, 1.0, color=VERT, alpha=0.12,
               label=f"9 vrais jumeaux sur 10 (≥ {d1:.2f})")
    ax.axvline(imp["seuil_hasard"].iloc[0], color=GRIS, ls="--", lw=1.2,
               label=f"hasard ({imp['seuil_hasard'].iloc[0]:.2f})")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score au test des imposteurs")
    ax.set_title("b) Même le meilleur candidat reste loin\ndu niveau d'un véritable alias")
    ax.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "05_1_candidats_imposteurs.png", bbox_inches="tight")
    plt.close(fig)


def fig_profil() -> None:
    """Portrait : marqueurs lexicaux, position stylistique, élision."""
    marq = pd.read_csv(RESULT_DIR / "06_2_marqueurs_lexicaux.csv")
    exc = pd.read_csv(RESULT_DIR / "06_1_excentricite.csv")
    el = pd.read_csv(RESULT_DIR / "06_5_controle_elision.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    ax = axes[0]
    # Trié sur le ratio, qui est la grandeur portée par les barres.
    top = marq.nlargest(14, "z_log_odds").nlargest(12, "ratio").iloc[::-1]
    ax.barh(np.arange(len(top)), top["ratio"], color=BLEU, height=0.62)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["mot"], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("Sur-emploi par rapport au corpus (échelle log)")
    ax.set_title("a) Ce qui signe l'écriture de Ziak\n(onomatopées et argot)")
    for i, (r, n) in enumerate(zip(top["ratio"], top["n_occurrences_ziak"])):
        ax.text(r * 1.1, i, f"×{r:.0f}", va="center", fontsize=7.5, color="#444")

    ax = axes[1]
    ax.hist(exc["distance_min"], bins=34, color=GRIS, alpha=0.65,
            label="tous les artistes")
    zk = exc[exc["artiste"] == "Ziak"].iloc[0]
    ax.axvline(zk["distance_min"], color=ROUGE, lw=2.2, label="Ziak")
    ax.axvline(exc["distance_min"].median(), color=BLEU, ls="--", lw=1.5,
               label="médiane du corpus")
    ax.set_xlabel("Distance au voisin stylistique le plus proche")
    ax.set_ylabel("Nombre d'artistes")
    rang = int((exc["distance_min"] < zk["distance_min"]).sum()) + 1
    ax.set_title(f"b) Ziak n'a pas de proche parent\n{100 * rang / len(exc):.0f} % des artistes "
                 f"ont un voisin plus proche")
    ax.legend(fontsize=8)

    ax = axes[2]
    e = el.head(6).iloc[::-1]
    y = np.arange(len(e))
    ax.barh(y - 0.21, e["ratio_forme_pleine"], height=0.2, color=ROUGE,
            label="forme pleine (je)")
    ax.barh(y, e["ratio_forme_elidee"], height=0.2, color=ORANGE,
            label="forme élidée (j')")
    ax.barh(y + 0.21, e["ratio_cumule"], height=0.2, color=VERT,
            label="les deux cumulées")
    ax.axvline(1.0, color="#333", lw=1, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.forme_pleine} / {r.forme_elidee}" for r in e.itertuples()],
                       fontsize=8.5)
    ax.set_xlabel("Rapport à la norme du corpus")
    ax.set_xlim(0, 1.85)
    ax.set_title("c) Un faux marqueur : l'élision\nvient du transcripteur, pas de l'auteur")
    ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Portrait stylométrique de Ziak", fontsize=12.5,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "06_1_profil_ziak.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for nom, fn in [("biais de taille", fig_biais_taille),
                    ("validation", fig_validation),
                    ("verdict", fig_verdict),
                    ("candidats", fig_candidats),
                    ("profil", fig_profil)]:
        fn()
        print(f"  figure « {nom} » écrite")
    print(f"\n{len(list(IMAGES_DIR.glob('0[2-6]_*.png')))} figures dans {IMAGES_DIR}/")


def fig_mikeysem() -> None:
    """Test de l'hypothèse Ziak = Mikeysem."""
    imp = pd.read_csv(RESULT_DIR / "10_4_imposteurs.csv")
    rk = pd.read_csv(RESULT_DIR / "10_2_rang_mikeysem.csv")
    val = pd.read_csv(RESULT_DIR / "10_1_validation_petite_taille.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # a) puissance atteinte a cette taille de corpus
    ax = axes[0]
    h1 = val[(val.condition == "H1") & (val.features == "char")
             & (val.metric == "cosine_delta")]
    ks = np.arange(1, 31)
    rec = [(h1["rank_twin"] <= k).mean() for k in ks]
    ax.plot(ks, rec, color=BLEU, lw=1.8)
    ax.axhline(1.0, color=GRIS, ls=":", lw=0.9)
    med = int(rk[(rk.features == "char") & (rk.metric == "cosine_delta")]
              ["rang_mikeysem"].median())
    ax.axvline(20, color=VERT, ls="--", lw=1.3,
               label=f"top 20 : {rec[19]:.0%} des vrais auteurs")
    ax.set_xlabel("k (rang compté comme un succès)")
    ax.set_ylabel("Proportion de bonnes réponses")
    ax.set_title("a) Puissance à 3 745 mots de candidat\n"
                 f"(rappel@1 = {rec[0]:.0%})")
    ax.legend(fontsize=7.5, loc="lower right")

    # b) rangs compares, a taille egale
    ax = axes[1]
    ordre = ["Kerchak", "Beendo Z", "ISK", "Werenoi", "Zkr", "Rimkus", "Mikeysem"]
    rangs = [4, 8, 10, 12, 18, 25, med]
    cols = [ROUGE if a == "Mikeysem" else GRIS for a in ordre]
    y = np.arange(len(ordre))[::-1]
    ax.barh(y, rangs, color=cols, height=0.62)
    for yy, r in zip(y, rangs):
        ax.text(r + 1.5, yy, str(r), va="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(ordre, fontsize=8.5)
    ax.set_xlabel("Rang parmi 493 candidats (vus depuis Ziak)")
    ax.set_title("b) Six artistes non soupçonnés\nsont plus proches que Mikeysem")

    # c) test des imposteurs
    ax = axes[2]
    ref = imp[imp.type == "jumeau authentique"]["score"]
    mk = float(imp[imp.type == "hypothèse"]["score"].iloc[0])
    ax.hist(ref, bins=16, color=VERT, alpha=0.55, label="jumeaux authentiques")
    ax.axvline(mk, color=ROUGE, lw=2.4, label=f"Ziak vs Mikeysem ({mk:.2f})")
    ax.axvline(1 / 26, color=GRIS, ls="--", lw=1.2, label="hasard")
    ax.set_xlabel("Score au test des imposteurs")
    ax.set_ylabel("Nombre d'artistes")
    ax.set_title("c) Un vrai alias score bien plus haut\n"
                 f"(médiane {ref.median():.2f})")
    ax.legend(fontsize=7.5, loc="upper center")

    fig.suptitle("Ziak et Mikeysem ne se ressemblent pas plus que des voisins de genre",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "10_1_test_mikeysem.png", bbox_inches="tight")
    plt.close(fig)


def fig_alias_reels() -> None:
    """Validation sur des recouvrements d'auteur réels (solo / groupe)."""
    syn = pd.read_csv(RESULT_DIR / "13_2_alias_reels_synthese.csv")
    bruts = pd.read_csv(RESULT_DIR / "13_1_alias_reels_bruts.csv")
    zsep = pd.read_csv(RESULT_DIR / "04_2_separation_ziak.csv")
    zsep = zsep[(zsep.features == "char") & (zsep.metric == "cosine_delta")]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5),
                             gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # a) rang du vrai partenaire, par paire
    ax = axes[0]
    s = syn.sort_values("rang_median", ascending=False)
    y = np.arange(len(s))
    cols = [VERT if r <= 20 else ORANGE for r in s.rang_median]
    ax.barh(y, s.rang_median, color=cols, height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} → {b}" for a, b in zip(s.solo, s.groupe)], fontsize=7.6)
    ax.set_xscale("log")
    ax.axvline(20, color=GRIS, ls="--", lw=1.2, label="seuil top 20")
    for yy, r in zip(y, s.rang_median):
        ax.text(r * 1.15, yy, f"{int(r)}", va="center", fontsize=7.5)
    ax.set_xlabel("Rang du groupe parmi 392 candidats (log)")
    ax.set_title("a) Le lien d'auteur est-il retrouvé ?\n(vert : oui, dans le top 20)")
    ax.legend(fontsize=7.5, loc="lower right")

    # b) effet de dilution
    ax = axes[1]
    ax.scatter(syn.n_rappeurs, syn.rang_median, s=46, c=BLEU, alpha=0.75,
               edgecolors="none")
    g = syn.groupby("n_rappeurs").rang_median.median()
    ax.plot(g.index, g.values, color=ROUGE, lw=1.8, marker="o", ms=4,
            label="médiane")
    ax.set_yscale("log")
    ax.axhline(20, color=GRIS, ls="--", lw=1.2)
    ax.set_xlabel("Nombre de rappeurs dans le groupe")
    ax.set_ylabel("Rang du groupe (log)")
    ax.set_title("b) Plus l'auteur est dilué,\nmoins il est détectable")
    ax.legend(fontsize=7.5)

    # c) separation du top-1 : cas reels vs Ziak
    ax = axes[2]
    ax.hist(bruts.sep_top1, bins=18, color=VERT, alpha=0.6,
            label="cas à lien réel")
    zv = float(zsep.sep_top1.mean())
    ax.axvline(zv, color=ROUGE, lw=2.4, label=f"Ziak ({zv:.2f})")
    ax.set_xlabel("Séparation du meilleur candidat")
    ax.set_ylabel("Fréquence")
    part = 100 * (bruts.sep_top1 > zv).mean()
    ax.set_title(f"c) Le favori de Ziak se détache moins\nque dans {100 - part:.0f} % des cas réels")
    ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Validation sur des recouvrements d'auteur réels, non simulés",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "13_1_alias_reels.png", bbox_inches="tight")
    plt.close(fig)


def fig_alias_temporel() -> None:
    """Ce que coûte un changement d'identité artistique."""
    res = pd.read_csv(RESULT_DIR / "14_1_alias_temporel_bruts.csv")
    syn = pd.read_csv(RESULT_DIR / "14_2_alias_temporel_synthese.csv")
    chg = syn[syn.groupe == "changement d'identité"].sort_values("rang_median")
    ctl = syn[syn.groupe == "sans changement"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    # a) les trois cas de changement d'identite
    ax = axes[0]
    y = np.arange(len(chg))[::-1]
    ax.barh(y, chg.rang_median, color=ROUGE, height=0.5)
    for yy, r in zip(y, chg.rang_median):
        ax.text(r + 0.4, yy, f"{int(r)}", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(chg.libelle, fontsize=8.5)
    ax.set_xlim(0, max(chg.rang_median) * 1.5)
    ax.set_xlabel("Rang de la période postérieure (sur 393)")
    ax.set_title("a) L'artiste change de nom —\nla méthode le retrouve quand même")

    # b) distribution des rangs : changement vs controle
    ax = axes[1]
    bins = np.logspace(0, np.log10(max(syn.rang_median.max(), 10)) + 0.1, 18)
    ax.hist(ctl.rang_median, bins=bins, color=GRIS, alpha=0.65,
            label=f"sans changement (n={len(ctl)})")
    for _, r in chg.iterrows():
        ax.axvline(r.rang_median, color=ROUGE, lw=1.8)
    ax.axvline(chg.rang_median.median(), color=ROUGE, lw=0, label="changement d'identité")
    ax.set_xscale("log")
    ax.set_xlabel("Rang de la période postérieure (log)")
    ax.set_ylabel("Nombre d'artistes")
    ax.set_title(f"b) Le changement d'identité coûte peu\n(médiane {ctl.rang_median.median():.0f} → "
                 f"{chg.rang_median.median():.0f})")
    ax.legend(fontsize=7.5)

    # c) separation : ces tests vs Ziak
    ax = axes[2]
    z = pd.read_csv(RESULT_DIR / "04_2_separation_ziak.csv")
    z = z[(z.features == "char") & (z.metric == "cosine_delta")]
    # Grandeur comparable : la séparation du MEILLEUR candidat du classement,
    # dans les deux cas — et non celle d'une cible désignée, que Ziak n'a pas.
    ax.hist(res.sep_top1, bins=20, color=VERT, alpha=0.6,
            label="top-1, tests à lien réel")
    zv = float(z.sep_top1.mean())
    ax.axvline(zv, color=ROUGE, lw=2.4, label=f"Ziak, top-1 ({zv:.2f})")
    ax.set_xlabel("Séparation du meilleur candidat")
    ax.set_ylabel("Fréquence")
    part = 100 * (res.sep_top1 < zv).mean()
    ax.set_title(f"c) Le favori de Ziak se détache moins\nque dans {part:.0f} % des tests à lien réel")
    ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("Un changement de nom n'efface pas la signature stylométrique",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "14_1_alias_temporel.png", bbox_inches="tight")
    plt.close(fig)
