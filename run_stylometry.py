#!/usr/bin/env python3
"""Stylometric comparison of Ziak against contemporaneous French rappers."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

IMAGES_DIR = Path("images")
RESULT_DIR = Path("result")
IMAGES_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

GENIUS_NOISE = re.compile(
    r"\[(?:paroles[^\]]*|couplet[^\]]*|refrain[^\]]*|pont[^\]]*|intro[^\]]*|"
    r"outro[^\]]*|pre-?refrain[^\]]*|hook[^\]]*|verse[^\]]*|chorus[^\]]*|"
    r"feat[^\]]*|prod[^\]]*)\]",
    flags=re.IGNORECASE,
)
NON_LETTERS = re.compile(r"[^a-zàâäéèêëïîôöùûüçœæ'\-\s]+", flags=re.IGNORECASE)


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = GENIUS_NOISE.sub(" ", text)
    text = text.replace("\n", " ").lower()
    text = NON_LETTERS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


FUNCTION_WORDS = [
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "mais",
    "donc", "car", "je", "j", "tu", "il", "elle", "on", "nous", "vous", "ils",
    "elles", "ce", "cet", "cette", "ces", "qui", "que", "quoi", "dont", "ou",
    "ne", "pas", "plus", "jamais", "toujours", "dans", "sur", "sous", "pour",
    "par", "avec", "sans", "comme", "si", "quand", "tout", "tous", "tres",
    "bien", "trop", "meme", "y", "en", "au", "aux", "a", "c", "ca", "est",
    "suis", "es", "sont", "etait", "ai", "as", "ont", "fait", "faire", "va",
    "vais", "veut", "peut", "faut", "rien", "tout", "peu", "encore", "deja",
    "aussi", "alors", "apres", "avant", "chez", "vers", "entre", "pendant",
]


def function_word_vector(text: str, vocab: list[str]) -> np.ndarray:
    tokens = strip_accents(text).split()
    n = max(len(tokens), 1)
    counts = {w: 0 for w in vocab}
    for tok in tokens:
        if tok in counts:
            counts[tok] += 1
    return np.array([counts[w] / n for w in vocab], dtype=float)


def burrows_delta(freq_matrix: np.ndarray) -> np.ndarray:
    """Manhattan Burrows's Delta on z-scored relative frequencies."""
    means = freq_matrix.mean(axis=0)
    stds = freq_matrix.std(axis=0, ddof=0)
    stds = np.where(stds < 1e-12, 1.0, stds)
    z = (freq_matrix - means) / stds
    n_features = z.shape[1]
    dists = np.abs(z[:, None, :] - z[None, :, :]).sum(axis=2) / n_features
    return dists


def main() -> None:
    df = pd.read_csv("RapFr.csv")
    df = df.dropna(subset=["artist", "lyrics"]).copy()
    df["lyrics_clean"] = df["lyrics"].map(clean_lyrics)
    df["n_chars"] = df["lyrics_clean"].str.len()
    df = df[df["n_chars"] >= 200].copy()

    target = "Ziak"
    ziak = df[df["artist"] == target]
    if ziak.empty:
        raise SystemExit("Ziak introuvable dans RapFr.csv")

    era_min, era_max = 2018, 2024
    min_songs = 25
    era = df[(df["year"] >= era_min) & (df["year"] <= era_max)]
    song_counts = era["artist"].value_counts()
    eligible = song_counts[song_counts >= min_songs].index.tolist()
    if target not in eligible:
        eligible.append(target)
    pool = era[era["artist"].isin(eligible)].copy()

    artist_docs = (
        pool.groupby("artist")["lyrics_clean"]
        .agg(lambda s: " ".join(s.tolist()))
        .sort_index()
    )
    artists = artist_docs.index.tolist()
    texts = artist_docs.tolist()
    n_songs = pool.groupby("artist").size().reindex(artists)

    meta = (
        pool.groupby("artist")
        .agg(
            n_songs=("title", "size"),
            n_words=("n_words", "sum"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            mean_word_len=("means_word_length", "mean"),
            mean_verlan=("n_verlan", "mean"),
            mean_argot=("n_argot", "mean"),
            mean_je=("n_je", "mean"),
            mean_profanity=("n_profanity", "mean"),
        )
        .reindex(artists)
    )
    meta.to_csv(RESULT_DIR / "02_1_artistes_eligibles.csv")

    # --- Character 4-grams ---
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(4, 4),
        min_df=3,
        max_features=8000,
        sublinear_tf=True,
        norm="l2",
    )
    X_char = char_vec.fit_transform(texts)
    sim_char = cosine_similarity(X_char)
    dist_char = 1.0 - sim_char
    dist_char_df = pd.DataFrame(dist_char, index=artists, columns=artists)
    dist_char_df.to_csv(RESULT_DIR / "02_2_distance_cosine_char4grams.csv")

    ziak_idx = artists.index(target)
    ranking_char = (
        dist_char_df.loc[target]
        .drop(target)
        .sort_values()
        .rename("cosine_distance_char4")
        .to_frame()
    )
    ranking_char["rang"] = np.arange(1, len(ranking_char) + 1)
    ranking_char.to_csv(RESULT_DIR / "02_3_classement_ziak_char4grams.csv")

    # --- Word 1-grams (content+function, TF-IDF) ---
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),
        min_df=5,
        max_features=5000,
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    X_word = word_vec.fit_transform(texts)
    dist_word = 1.0 - cosine_similarity(X_word)
    dist_word_df = pd.DataFrame(dist_word, index=artists, columns=artists)
    ranking_word = (
        dist_word_df.loc[target]
        .drop(target)
        .sort_values()
        .rename("cosine_distance_word1")
        .to_frame()
    )

    # --- Burrows Delta on function words ---
    fw_mat = np.vstack([function_word_vector(t, FUNCTION_WORDS) for t in texts])
    delta = burrows_delta(fw_mat)
    delta_df = pd.DataFrame(delta, index=artists, columns=artists)
    delta_df.to_csv(RESULT_DIR / "02_4_burrows_delta_mots_fonction.csv")
    ranking_delta = (
        delta_df.loc[target]
        .drop(target)
        .sort_values()
        .rename("burrows_delta")
        .to_frame()
    )

    combined = (
        ranking_char.join(ranking_word, how="inner")
        .join(ranking_delta, how="inner")
        .join(meta.drop(index=target)[["n_songs", "n_words"]], how="left")
    )
    for col in ["cosine_distance_char4", "cosine_distance_word1", "burrows_delta"]:
        combined[f"z_{col}"] = (combined[col] - combined[col].mean()) / combined[col].std(ddof=0)
    combined["score_agrege"] = combined[
        ["z_cosine_distance_char4", "z_cosine_distance_word1", "z_burrows_delta"]
    ].mean(axis=1)
    combined = combined.sort_values("score_agrege")
    combined["rang_agrege"] = np.arange(1, len(combined) + 1)
    combined.to_csv(RESULT_DIR / "02_5_classement_agrege_ziak.csv")

    # --- Intra-artist baseline: split each artist 50/50 ---
    rng = np.random.default_rng(42)
    intra_rows = []
    half_texts = {}
    for artist, group in pool.groupby("artist"):
        songs = group["lyrics_clean"].tolist()
        if len(songs) < 8:
            continue
        idx = np.arange(len(songs))
        rng.shuffle(idx)
        mid = len(idx) // 2
        a = " ".join(songs[i] for i in idx[:mid])
        b = " ".join(songs[i] for i in idx[mid:])
        half_texts[f"{artist}__A"] = a
        half_texts[f"{artist}__B"] = b
        intra_rows.append(artist)

    half_names = list(half_texts.keys())
    half_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(4, 4), min_df=2, max_features=8000,
        sublinear_tf=True, norm="l2",
    )
    X_half = half_vec.fit_transform([half_texts[n] for n in half_names])
    sim_half = cosine_similarity(X_half)
    sim_half_df = pd.DataFrame(sim_half, index=half_names, columns=half_names)
    intra_d = []
    for artist in intra_rows:
        d = 1.0 - float(sim_half_df.loc[f"{artist}__A", f"{artist}__B"])
        intra_d.append({"artist": artist, "intra_distance_char4": d})
    intra_df = pd.DataFrame(intra_d).set_index("artist")
    intra_df.to_csv(RESULT_DIR / "02_6_distance_intra_artiste.csv")

    ziak_vs = ranking_char.copy()
    ziak_vs["intra_ziak"] = intra_df.loc[target, "intra_distance_char4"]
    ziak_vs["intra_other"] = intra_df["intra_distance_char4"].reindex(ziak_vs.index)
    ziak_vs["ratio_vs_intra_ziak"] = ziak_vs["cosine_distance_char4"] / ziak_vs["intra_ziak"]
    ziak_vs.to_csv(RESULT_DIR / "02_7_ziak_vs_baseline_intra.csv")

    # --- Song-level: Ziak songs vs artist centroids (leave Ziak out of centroid) ---
    song_vec = TfidfVectorizer(
        analyzer="char", ngram_range=(4, 4), min_df=3, max_features=6000,
        sublinear_tf=True, norm="l2",
    )
    other_artists = [a for a in artists if a != target]
    centroids = artist_docs.drop(target)
    song_vec.fit(list(centroids.values) + ziak["lyrics_clean"].tolist())
    C = song_vec.transform(centroids.reindex(other_artists).tolist())
    Z = song_vec.transform(ziak["lyrics_clean"].tolist())
    song_sim = cosine_similarity(Z, C)
    nearest = []
    for i, title in enumerate(ziak["title"].tolist()):
        j = int(np.argmax(song_sim[i]))
        nearest.append(
            {
                "title": title,
                "year": ziak["year"].iloc[i],
                "nearest_artist": other_artists[j],
                "cosine_similarity": float(song_sim[i, j]),
            }
        )
    nearest_df = pd.DataFrame(nearest)
    nearest_df.to_csv(RESULT_DIR / "02_8_plus_proche_par_titre_ziak.csv", index=False)
    vote = nearest_df["nearest_artist"].value_counts().rename("n_titres_ziak_assignes")
    vote.to_csv(RESULT_DIR / "02_9_votes_plus_proche_voisin.csv")

    # --- Supervised: can Ziak be recovered as its own class vs others? ---
    # Sample up to 40 songs per artist among a compact candidate set:
    # Ziak + 12 closest char-4 + 8 random distant controls if needed
    closest12 = ranking_char.head(12).index.tolist()
    clf_artists = [target] + closest12
    clf_df = pd.concat(
        [
            g.sample(n=min(len(g), 35), random_state=0)
            for _, g in pool[pool["artist"].isin(clf_artists)].groupby("artist")
        ]
    )
    pipe = make_pipeline(
        TfidfVectorizer(
            analyzer="char", ngram_range=(4, 4), min_df=2, max_features=4000,
            sublinear_tf=True,
        ),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=4000, dual=True),
    )
    y = clf_df["artist"].values
    X_text = clf_df["lyrics_clean"].values
    counts = pd.Series(y).value_counts()
    keep = counts[counts >= 8].index
    mask = pd.Series(y).isin(keep).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X_text[mask], y[mask], cv=cv, scoring="f1_macro")
    pd.DataFrame({"fold": np.arange(1, 6), "f1_macro": scores}).to_csv(
        RESULT_DIR / "02_10_cv_f1_ziak_vs_proches.csv", index=False
    )

    # Confusion-oriented: train without Ziak, predict Ziak songs among closest 12
    impostors = closest12
    train = pool[pool["artist"].isin(impostors)]
    test = pool[pool["artist"] == target]
    ident_pipe = make_pipeline(
        TfidfVectorizer(
            analyzer="char", ngram_range=(4, 4), min_df=2, max_features=4000,
            sublinear_tf=True,
        ),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=4000, dual=True),
    )
    ident_pipe.fit(train["lyrics_clean"], train["artist"])
    pred = ident_pipe.predict(test["lyrics_clean"])
    proba_like = ident_pipe.decision_function(test["lyrics_clean"])
    classes = ident_pipe.named_steps["linearsvc"].classes_
    pred_df = test[["title", "year"]].copy()
    pred_df["pred_artist"] = pred
    # margin: score of predicted class
    if proba_like.ndim == 1:
        pred_df["decision"] = proba_like
    else:
        pred_df["decision"] = proba_like.max(axis=1)
        for k, cls in enumerate(classes):
            pred_df[f"score_{cls}"] = proba_like[:, k]
    pred_df.to_csv(RESULT_DIR / "02_11_attribution_forcee_sans_classe_ziak.csv", index=False)
    forced = pred_df["pred_artist"].value_counts().rename("n")
    forced.to_csv(RESULT_DIR / "02_12_votes_attribution_forcee.csv")

    # --- PCA of char-4 artist vectors ---
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(X_char.toarray())
    pca_df = pd.DataFrame(xy, index=artists, columns=["pc1", "pc2"])
    pca_df["is_ziak"] = pca_df.index == target
    pca_df.to_csv(RESULT_DIR / "02_13_pca_char4_artistes.csv")

    # ===== Figures =====
    topn = ranking_char.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(x=topn["cosine_distance_char4"], y=topn.index, ax=ax, color="#3b6d8c")
    ax.axvline(intra_df.loc[target, "intra_distance_char4"], color="#c0392b", ls="--",
               label=f"Distance intra-Ziak ({intra_df.loc[target, 'intra_distance_char4']:.3f})")
    ax.set_xlabel("Distance cosinus (1 − similarité) — 4-grammes de caractères")
    ax.set_ylabel("")
    ax.set_title("Artistes les plus proches de Ziak (corpus 2018–2024)")
    ax.legend(loc="lower right")
    fig.savefig(IMAGES_DIR / "02_3_proches_ziak_char4grams.png", bbox_inches="tight")
    plt.close(fig)

    heat_names = [target] + ranking_char.head(14).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        dist_char_df.loc[heat_names, heat_names],
        cmap="mako_r",
        ax=ax,
        square=True,
        cbar_kws={"label": "Distance cosinus"},
    )
    ax.set_title("Distances stylométriques (4-grammes) : Ziak et 14 plus proches")
    fig.savefig(IMAGES_DIR / "02_3_heatmap_ziak_proches.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    others = pca_df[~pca_df["is_ziak"]]
    ax.scatter(others["pc1"], others["pc2"], s=18, alpha=0.55, c="#7f8c9a", label="Autres artistes")
    ax.scatter(
        pca_df.loc[target, "pc1"], pca_df.loc[target, "pc2"],
        s=140, c="#c0392b", zorder=3, label="Ziak",
    )
    for name in ranking_char.head(8).index:
        ax.annotate(name, (pca_df.loc[name, "pc1"], pca_df.loc[name, "pc2"]), fontsize=8)
    ax.annotate("Ziak", (pca_df.loc[target, "pc1"], pca_df.loc[target, "pc2"]),
                fontsize=10, fontweight="bold", color="#c0392b")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    ax.set_title("ACP des profils 4-grammes (centroïdes d'artistes)")
    ax.legend()
    fig.savefig(IMAGES_DIR / "02_13_pca_char4_artistes.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    vote.head(12).plot(kind="barh", ax=ax, color="#3b6d8c")
    ax.invert_yaxis()
    ax.set_xlabel("Nombre de titres de Ziak dont le centroïde le plus proche est cet artiste")
    ax.set_title("Plus proche voisin au niveau du titre")
    fig.savefig(IMAGES_DIR / "02_8_votes_plus_proche_voisin.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    forced.head(12).plot(kind="barh", ax=ax, color="#8c4b3b")
    ax.invert_yaxis()
    ax.set_xlabel("Titres de Ziak attribués (classifieur entraîné sans Ziak)")
    ax.set_title("Attribution forcée parmi les 12 artistes les plus proches")
    fig.savefig(IMAGES_DIR / "02_11_attribution_forcee.png", bbox_inches="tight")
    plt.close(fig)

    # Distribution: intra vs Ziak-other
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        intra_df["intra_distance_char4"], bins=18, alpha=0.55, color="#3b6d8c",
        label="Distances intra-artiste", density=True,
    )
    ax.hist(
        ranking_char["cosine_distance_char4"], bins=18, alpha=0.45, color="#c0392b",
        label="Distances Ziak → autres", density=True,
    )
    ax.axvline(intra_df.loc[target, "intra_distance_char4"], color="#c0392b", ls="--",
               label="Intra-Ziak")
    ax.set_xlabel("Distance cosinus (4-grammes)")
    ax.set_ylabel("Densité")
    ax.set_title("Cohérence interne vs distances de Ziak aux autres")
    ax.legend()
    fig.savefig(IMAGES_DIR / "02_6_intra_vs_inter_ziak.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_ziak_songs": int(len(ziak)),
        "n_artists_pool": int(len(artists)),
        "era": f"{era_min}-{era_max}",
        "min_songs": min_songs,
        "closest_char4": ranking_char.index[0],
        "closest_char4_dist": float(ranking_char.iloc[0, 0]),
        "intra_ziak": float(intra_df.loc[target, "intra_distance_char4"]),
        "median_intra": float(intra_df["intra_distance_char4"].median()),
        "median_ziak_to_others": float(ranking_char["cosine_distance_char4"].median()),
        "closest_vs_intra_ratio": float(
            ranking_char.iloc[0, 0] / intra_df.loc[target, "intra_distance_char4"]
        ),
        "forced_mode": str(forced.index[0]),
        "forced_mode_share": float(forced.iloc[0] / forced.sum()),
        "cv_f1_mean": float(scores.mean()),
        "cv_f1_std": float(scores.std()),
        "pca_var": float(pca.explained_variance_ratio_.sum()),
        "nn_mode": str(vote.index[0]),
        "nn_mode_share": float(vote.iloc[0] / vote.sum()),
        "agrege_top": combined.index[0],
    }
    pd.Series(summary).to_csv(RESULT_DIR / "02_0_synthese_stylometrie.csv")
    print(pd.Series(summary).to_string())
    print("\nTOP 10 char4:\n", ranking_char.head(10).to_string())
    print("\nTOP 10 agrege:\n", combined[["score_agrege", "cosine_distance_char4", "burrows_delta"]].head(10).to_string())
    print("\nForced votes:\n", forced.head(10).to_string())
    print("\nNN votes:\n", vote.head(10).to_string())


if __name__ == "__main__":
    main()
