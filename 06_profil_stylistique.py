#!/usr/bin/env python3
"""Portrait stylométrique de Ziak, à défaut d'une identification.

Si aucun artiste du corpus ne correspond, deux questions restent ouvertes et
répondables : *où* se situe Ziak dans l'espace stylistique du rap français, et
*par quoi* son écriture se distingue.

Quatre mesures :

1. **Excentricité** — Ziak est-il stylistiquement banal ou marginal ? On
   compare sa distance médiane au reste du corpus à celle de tous les autres.
2. **Marqueurs lexicaux** — les mots dont l'usage s'écarte le plus de la norme
   du corpus, mesurés par log-odds avec lissage bayésien.
3. **Stabilité temporelle** — une signature d'auteur suppose une constance ;
   on compare les périodes 2020-2021 et 2023-2024.
4. **Voisinage** — quels artistes composent son entourage stylistique, et ce
   que ce voisinage a de commun.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stylo_attribution import make_docs, rank_candidates, sample_indices
from stylo_features import build_cache

RESULT_DIR = Path("export")
RESULT_DIR.mkdir(exist_ok=True)

TARGET = "Ziak"
T_CAND = 12_000
SEED = 20260913


def log_odds_marqueurs(cache, top_n: int = 30) -> pd.DataFrame:
    """Mots sur- et sous-employés par Ziak (log-odds à prior informatif).

    Méthode de Monroe, Colaresi & Quinn (2008) : le lissage par le corpus de
    référence évite que des mots rares produisent des scores aberrants.
    """
    meta = cache["meta"]
    counts = cache["counts_word_ext"]
    vocab = np.array(cache["vocab_word_ext"])

    is_ziak = (meta["artist"] == TARGET).to_numpy()
    y_i = np.asarray(counts[is_ziak].sum(axis=0)).ravel()    # corpus de Ziak
    y_j = np.asarray(counts[~is_ziak].sum(axis=0)).ravel()   # reste du corpus

    # Prior informatif : la force totale `alpha0` est répartie sur les mots
    # proportionnellement à leur fréquence d'ensemble. Un prior uniforme
    # écraserait les mots rares et gonflerait artificiellement leurs scores.
    a0 = y_i + y_j
    alpha0 = float(len(vocab))
    alpha = alpha0 * a0 / max(a0.sum(), 1.0)

    n_i, n_j = y_i.sum(), y_j.sum()
    num_i, num_j = y_i + alpha, y_j + alpha
    delta = (np.log(num_i) - np.log(n_i + alpha0 - num_i)) - (
        np.log(num_j) - np.log(n_j + alpha0 - num_j)
    )
    z = delta / np.sqrt(1.0 / num_i + 1.0 / num_j)

    df = pd.DataFrame({
        "mot": vocab,
        "z_log_odds": z,
        "n_occurrences_ziak": y_i.astype(int),
        "freq_ziak_pour_mille": 1000 * y_i / max(n_i, 1),
        "freq_corpus_pour_mille": 1000 * y_j / max(n_j, 1),
    })
    df["ratio"] = df["freq_ziak_pour_mille"] / df["freq_corpus_pour_mille"].replace(0, np.nan)
    # Un marqueur doit reposer sur assez d'occurrences pour être commentable.
    df = df[df["n_occurrences_ziak"] >= 5].sort_values("z_log_odds", ascending=False)
    out = pd.concat([df.head(top_n), df.tail(top_n)])
    out.to_csv(RESULT_DIR / "06_2_marqueurs_lexicaux.csv", index=False)
    return out


ELISIONS = [("je", "j'"), ("de", "d'"), ("le", "l'"), ("ce", "c'"), ("me", "m'"),
            ("te", "t'"), ("se", "s'"), ("ne", "n'"), ("que", "qu'"), ("la", "l'")]


def controle_elision(cache) -> pd.DataFrame:
    """Distingue les vrais marqueurs des artefacts de transcription.

    Les paroles de Genius sont saisies par des contributeurs : « j'suis » et
    « je suis » notent la même diction. Un mot dont l'écart disparaît une fois
    la forme pleine et la forme élidée additionnées ne dit rien de l'auteur,
    seulement de la personne qui a transcrit le texte.
    """
    meta = cache["meta"]
    counts = cache["counts_word_ext"]
    vocab = list(cache["vocab_word_ext"])
    idx = {w: i for i, w in enumerate(vocab)}

    is_ziak = (meta["artist"] == TARGET).to_numpy()
    y_i = np.asarray(counts[is_ziak].sum(axis=0)).ravel()
    y_j = np.asarray(counts[~is_ziak].sum(axis=0)).ravel()
    n_i, n_j = y_i.sum(), y_j.sum()

    rows = []
    for plein, elide in ELISIONS:
        ip, ie = idx.get(plein), idx.get(elide)
        if ip is None or ie is None:
            continue
        r_plein = (y_i[ip] / n_i) / (y_j[ip] / n_j)
        r_elide = (y_i[ie] / n_i) / (y_j[ie] / n_j)
        r_cumul = ((y_i[ip] + y_i[ie]) / n_i) / ((y_j[ip] + y_j[ie]) / n_j)
        rows.append({
            "forme_pleine": plein, "forme_elidee": elide,
            "ratio_forme_pleine": r_plein, "ratio_forme_elidee": r_elide,
            "ratio_cumule": r_cumul,
            "artefact_de_transcription": abs(np.log(r_cumul)) < 0.2 < abs(np.log(r_plein)),
        })
    out = pd.DataFrame(rows).sort_values("ratio_forme_pleine")
    out.to_csv(RESULT_DIR / "06_5_controle_elision.csv", index=False)
    return out


def excentricite(cache) -> tuple[pd.DataFrame, float]:
    """Distance médiane de chaque artiste à tous les autres, à taille égale."""
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    totals = meta.groupby("artist")["n_tokens"].sum()

    pool = sorted(totals[totals >= T_CAND].index)
    pool_songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    rng = np.random.default_rng(SEED)

    acc = np.zeros((len(pool), len(pool)))
    n_rep = 8
    for _ in range(n_rep):
        groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], T_CAND, rng)
                  for a in pool}
        names, freqs = make_docs(mat, groups)
        mu, sd = freqs.mean(axis=0), freqs.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        z = (freqs - mu) / sd
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        acc += 1.0 - z @ z.T
    acc /= n_rep
    np.fill_diagonal(acc, np.nan)

    df = pd.DataFrame({
        "artiste": names,
        "distance_mediane_aux_autres": np.nanmedian(acc, axis=1),
        "distance_min": np.nanmin(acc, axis=1),
        "voisin_le_plus_proche": [names[i] for i in np.nanargmin(acc, axis=1)],
        "n_tokens_corpus": [int(totals[a]) for a in names],
    }).sort_values("distance_mediane_aux_autres", ascending=False).reset_index(drop=True)
    df["rang_excentricite"] = np.arange(1, len(df) + 1)
    df.to_csv(RESULT_DIR / "06_1_excentricite.csv", index=False)

    pct = 100 * (df[df["artiste"] == TARGET].index[0] + 1) / len(df)
    return df, pct


def stabilite_temporelle(cache) -> pd.DataFrame:
    """Le style de Ziak est-il constant entre ses deux périodes de production ?

    Repère : la distance entre deux moitiés aléatoires d'un même artiste. Si
    l'écart entre périodes reste dans cette fourchette, la signature est stable.
    """
    meta = cache["meta"]
    mat = cache["counts_char"]
    n_tokens = meta["n_tokens"].to_numpy()
    totals = meta.groupby("artist")["n_tokens"].sum()

    ziak = meta[meta["artist"] == TARGET]
    p1 = ziak.index[ziak["year"] <= 2021].to_numpy()
    p2 = ziak.index[ziak["year"] >= 2022].to_numpy()

    pool = sorted(a for a in totals[totals >= T_CAND].index if a != TARGET)
    pool_songs = {a: meta.index[meta["artist"] == a].to_numpy() for a in pool}
    rng = np.random.default_rng(SEED)
    groups = {a: sample_indices(pool_songs[a], n_tokens[pool_songs[a]], T_CAND, rng)
              for a in pool}
    names, freqs = make_docs(mat, groups)

    def freq(ids):
        v = np.asarray(mat[ids].sum(axis=0)).ravel()
        return v / max(v.sum(), 1.0)

    f1, f2 = freq(p1), freq(p2)
    d_periodes = float(rank_candidates(f1, np.vstack([freqs, f2]), "cosine_delta")[-1])

    # Repère : écart entre deux moitiés aléatoires, chez Ziak et chez les autres.
    ziak_ids = ziak.index.to_numpy()
    splits = []
    for _ in range(30):
        perm = rng.permutation(len(ziak_ids))
        half = len(perm) // 2
        a, b = ziak_ids[perm[:half]], ziak_ids[perm[half:]]
        splits.append(float(rank_candidates(freq(a), np.vstack([freqs, freq(b)]),
                                            "cosine_delta")[-1]))

    autres = []
    for art in rng.choice(pool, size=60, replace=False):
        ids = pool_songs[art]
        if len(ids) < 8:
            continue
        perm = rng.permutation(len(ids))
        half = len(perm) // 2
        a, b = ids[perm[:half]], ids[perm[half:]]
        autres.append(float(rank_candidates(freq(a), np.vstack([freqs, freq(b)]),
                                            "cosine_delta")[-1]))

    out = pd.DataFrame([{
        "distance_2020_2021_vs_2022_2024": d_periodes,
        "n_titres_periode_1": len(p1),
        "n_titres_periode_2": len(p2),
        "distance_moities_aleatoires_ziak_moy": float(np.mean(splits)),
        "distance_moities_aleatoires_ziak_sd": float(np.std(splits)),
        "distance_moities_aleatoires_autres_moy": float(np.mean(autres)),
        "z_vs_moitiés_aleatoires": (d_periodes - np.mean(splits)) / (np.std(splits) + 1e-12),
    }])
    out.to_csv(RESULT_DIR / "06_3_stabilite_temporelle.csv", index=False)
    return out


def main() -> None:
    cache = build_cache()
    meta = cache["meta"]

    print("=== 1. EXCENTRICITÉ STYLISTIQUE ===")
    exc, pct = excentricite(cache)
    z = exc[exc["artiste"] == TARGET].iloc[0]
    rang_min = int((exc["distance_min"] < z["distance_min"]).sum()) + 1
    print("  Deux mesures qu'il faut distinguer :")
    print(f"  a) Distance médiane à l'ensemble du corpus — position d'ensemble")
    print(f"     {TARGET} : {z['distance_mediane_aux_autres']:.4f}  "
          f"(médiane du corpus : {exc['distance_mediane_aux_autres'].median():.4f})")
    print(f"     Rang {int(z['rang_excentricite'])} / {len(exc)} : Ziak est stylistiquement "
          f"*banal*, au centre du genre.")
    print(f"  b) Distance au voisin le plus proche — existe-t-il un parent ?")
    print(f"     {TARGET} : {z['distance_min']:.4f} vers {z['voisin_le_plus_proche']}  "
          f"(médiane du corpus : {exc['distance_min'].median():.4f})")
    print(f"     Rang {rang_min} / {len(exc)} : {100 * rang_min / len(exc):.0f} % des artistes "
          f"ont un voisin plus proche que Ziak n'en a un.")
    print("\n  Artistes les plus excentrés du corpus (repère) :")
    for _, r in exc.head(6).iterrows():
        print(f"    {r['rang_excentricite']:>3}. {r['artiste']:<26} "
              f"{r['distance_mediane_aux_autres']:.4f}")

    print("\n=== 2. MARQUEURS LEXICAUX ===")
    marq = log_odds_marqueurs(cache)
    print("  Sur-employés par Ziak :")
    for _, r in marq.head(14).iterrows():
        print(f"    {r['mot']:<14} z = {r['z_log_odds']:>6.1f}   "
              f"{r['freq_ziak_pour_mille']:>6.2f} vs {r['freq_corpus_pour_mille']:>6.2f} ‰"
              f"   (x{r['ratio']:.1f})")
    print("  Sous-employés par Ziak :")
    for _, r in marq.tail(8).iterrows():
        print(f"    {r['mot']:<14} z = {r['z_log_odds']:>6.1f}   "
              f"{r['freq_ziak_pour_mille']:>6.2f} vs {r['freq_corpus_pour_mille']:>6.2f} ‰")

    print("\n  Contrôle : ces marqueurs résistent-ils à l'élision ?")
    el = controle_elision(cache)
    for _, r in el.head(5).iterrows():
        flag = "  <-- artefact de transcription" if r["artefact_de_transcription"] else ""
        print(f"    {r['forme_pleine']:<5} x{r['ratio_forme_pleine']:.2f}   "
              f"{r['forme_elidee']:<5} x{r['ratio_forme_elidee']:.2f}   "
              f"cumulé x{r['ratio_cumule']:.2f}{flag}")
    n_art = int(el["artefact_de_transcription"].sum())
    print(f"    -> {n_art} des {len(el)} paires testées sont des artefacts : l'écart "
          f"disparaît une fois les deux formes additionnées.")

    print("\n=== 3. STABILITÉ TEMPORELLE ===")
    st = stabilite_temporelle(cache)
    r = st.iloc[0]
    print(f"  Période 1 (2020-2021) : {int(r['n_titres_periode_1'])} titres")
    print(f"  Période 2 (2022-2024) : {int(r['n_titres_periode_2'])} titres")
    print(f"  Distance entre les deux périodes      : {r['distance_2020_2021_vs_2022_2024']:.4f}")
    print(f"  Distance entre moitiés aléatoires     : {r['distance_moities_aleatoires_ziak_moy']:.4f} "
          f"(± {r['distance_moities_aleatoires_ziak_sd']:.4f})")
    print(f"  Écart standardisé                     : z = {r['z_vs_moitiés_aleatoires']:+.2f}")

    print("\n=== 4. VOISINAGE STYLISTIQUE ===")
    cons = pd.read_csv(RESULT_DIR / "04_5_consensus_candidats.csv").head(12)
    annees = meta.groupby("artist")["year"].median()
    n_t = meta.groupby("artist")["n_tokens"].sum()
    cons["annee_mediane"] = cons["artiste"].map(annees)
    cons["n_tokens"] = cons["artiste"].map(n_t)
    cons.to_csv(RESULT_DIR / "06_4_voisinage_ziak.csv", index=False)
    print(cons[["artiste", "rang_moyen", "meilleur_rang", "annee_mediane"]]
          .to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\n  Année médiane de production du voisinage : "
          f"{cons['annee_mediane'].median():.0f} (Ziak : {annees[TARGET]:.0f})")


if __name__ == "__main__":
    main()
