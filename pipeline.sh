#!/usr/bin/env bash
# Pipeline de traitement complète : du corpus brut à l'article.
#
#   ./pipeline.sh                  # toutes les étapes
#   ./pipeline.sh analyses figures # seulement celles-là
#   ./pipeline.sh --depuis variantes
#   ./pipeline.sh --liste
#
# Chaque étape est idempotente : ce qui existe déjà n'est pas refait. Relancer
# la pipeline après une interruption reprend donc où elle s'est arrêtée, sans
# rejouer les heures de collecte.
#
# L'étape « balises » re-télécharge 34 732 pages Genius et dure environ six
# heures. Elle ne démarre pas sans --oui, pour qu'on ne la lance jamais par
# inadvertance.
#
# Journal horodaté : export/pipeline.log
set -euo pipefail
cd "$(dirname "$0")"

ETAPES=(corpus collectes analyses balises corpus_propres variantes figures article)
JOURNAL="export/pipeline.log"
FORCE=0
OUI=0

# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
log() { printf '%s  %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$JOURNAL"; }

titre() {
  printf '\n\033[1m── %s ─────────────────────────────────────\033[0m\n' "$1"
  printf '\n=== %s (%s) ===\n' "$1" "$(date +%H:%M)" >> "$JOURNAL"
}

# Lance une commande seulement si sa sortie manque (ou si --force).
si_absent() {
  local cible="$1"; shift
  if [ "$FORCE" -eq 0 ] && [ -e "$cible" ]; then
    log "· $cible déjà présent, on passe"
    return 0
  fi
  log "→ $*"
  "$@" >> "$JOURNAL" 2>&1
}

# ----------------------------------------------------------------------------
# 1. Le corpus publié et son cache
# ----------------------------------------------------------------------------
etape_corpus() {
  titre "1/8  corpus publié"
  if [ ! -f corpus.csv ] && [ ! -f RapFr.csv ]; then
    log "→ téléchargement du corpus LRFAF (114 Mo)"
    curl -fL --progress-bar \
      "https://huggingface.co/datasets/regicid/LRFAF/resolve/main/corpus.csv?download=true" \
      -o corpus.csv
  else
    log "· corpus déjà présent"
  fi
  si_absent .cache_stylo/meta.parquet python3 stylo_features.py
}

# ----------------------------------------------------------------------------
# 2. Les artistes absents du corpus, collectés sur Genius
# ----------------------------------------------------------------------------
etape_collectes() {
  titre "2/8  collectes Genius (Mikeysem, web7, Ziak)"
  si_absent .cache_lex/mikeysem_raw.json \
    python3 collecte_genius.py 3152412 .cache_lex/mikeysem_raw.json
  si_absent .cache_lex/web7_raw.json \
    python3 collecte_genius.py 1078135 .cache_lex/web7_raw.json
  si_absent .cache_lex/ziak_raw.json \
    python3 collecte_genius.py 2113831 .cache_lex/ziak_raw.json
}

# ----------------------------------------------------------------------------
# 3. L'étude sur le corpus publié
# ----------------------------------------------------------------------------
etape_analyses() {
  titre "3/8  analyses sur le corpus publié"
  # L'ordre est imposé par les dépendances : 05 lit 03, 06/13/14 lisent 04,
  # 10 lit 08, 16 lit les crédits produits par 15.
  for s in 02_diagnostic_biais_taille 03_validation_protocole 04_attribution_ziak \
           05_robustesse 06_profil_stylistique 08_ajout_mikeysem \
           09_controle_reproduction 10_test_ziak_mikeysem \
           13_validation_alias_reels 14_test_alias_temporel \
           15_test_ziak_7jaws 16_test_2025_web7 18_controle_featurings; do
    log "→ $s.py"
    python3 "$s.py" >> "$JOURNAL" 2>&1
  done
}

# ----------------------------------------------------------------------------
# 4. Les balises de section, re-téléchargées (≈ 6 h)
# ----------------------------------------------------------------------------
etape_balises() {
  titre "4/8  re-collecte des paroles balisées"
  local faits=0
  [ -f .cache_lex/corpus_balises.jsonl ] && faits=$(wc -l < .cache_lex/corpus_balises.jsonl)
  if [ "$faits" -ge 34000 ]; then
    log "· $faits titres déjà collectés, on passe"
    return 0
  fi
  if [ "$OUI" -eq 0 ]; then
    log "! étape longue (~6 h, 34 732 pages Genius) — relancer avec --oui"
    log "  reprise automatique : $faits titres déjà en cache"
    return 0
  fi
  log "→ 19_collecte_corpus.py (reprise à $faits titres)"
  python3 19_collecte_corpus.py >> "$JOURNAL" 2>&1
}

# ----------------------------------------------------------------------------
# 5. Les deux corpus sans featurings
# ----------------------------------------------------------------------------
etape_corpus_propres() {
  titre "5/8  corpus sans featurings (options 1 et 2)"
  if [ ! -f .cache_lex/corpus_balises.jsonl ]; then
    log "! .cache_lex/corpus_balises.jsonl absent — étape « balises » d'abord"
    return 1
  fi
  si_absent corpus_sans_invites.csv python3 20_corpus_sans_invites.py
}

# ----------------------------------------------------------------------------
# 6. L'étude rejouée sur chaque variante
# ----------------------------------------------------------------------------
etape_variantes() {
  titre "6/8  étude rejouée sur les deux variantes"
  ./rejouer_variantes.sh >> "$JOURNAL" 2>&1
}

# ----------------------------------------------------------------------------
# 7. Les figures
# ----------------------------------------------------------------------------
etape_figures() {
  titre "7/8  figures"
  log "→ 07_figures.py (corpus publié)"
  python3 07_figures.py >> "$JOURNAL" 2>&1
  for v in sans_invites sans_feats; do
    [ -d ".cache_stylo_$v" ] || continue
    log "→ 07_figures.py ($v)"
    CORPUS_VARIANTE="$v" python3 07_figures.py >> "$JOURNAL" 2>&1
  done
}

# ----------------------------------------------------------------------------
# 8. L'article, dans les deux formats
# ----------------------------------------------------------------------------
etape_article() {
  titre "8/8  article"
  # Le Word fait foi dès qu'il a été retouché à la main : on prévient plutôt
  # que d'écraser silencieusement des heures de relecture.
  if [ -f article_ziak_stylometrie.docx ] && [ "$FORCE" -eq 0 ] \
     && [ article_ziak_stylometrie.docx -nt article_contenu.py ]; then
    log "! le .docx est plus récent que article_contenu.py — modifié à la main ?"
    log "  régénérer quand même : ./pipeline.sh article --force"
    return 0
  fi
  log "→ 12_article_pdf.py"
  python3 12_article_pdf.py >> "$JOURNAL" 2>&1
  log "→ 17_article_docx.py"
  python3 17_article_docx.py >> "$JOURNAL" 2>&1
  log "→ 21_schema_pipeline.py"
  python3 21_schema_pipeline.py >> "$JOURNAL" 2>&1
  log "→ 23_notebook.py"
  python3 23_notebook.py >> "$JOURNAL" 2>&1
}

# ----------------------------------------------------------------------------
# Aiguillage
# ----------------------------------------------------------------------------
usage() {
  cat <<'FIN'
Usage : ./pipeline.sh [options] [étape...]

Étapes, dans l'ordre :
  corpus          télécharge le corpus LRFAF et construit le cache
  collectes       Mikeysem, web7 et Ziak depuis Genius
  analyses        l'étude sur le corpus publié (13 scripts)
  balises         re-collecte des paroles balisées      (~6 h, exige --oui)
  corpus_propres  reconstruit les deux corpus nettoyés
  variantes       rejoue l'étude sur chaque variante    (~20 min)
  figures         les figures, pour chaque corpus
  article         le PDF, le Word, le schéma imprimable et le notebook

Options :
  --depuis ÉTAPE  démarre à cette étape et continue jusqu'au bout
  --force         refait même ce qui existe déjà
  --oui           autorise l'étape longue « balises »
  --liste         affiche les étapes et sort
FIN
}

CHOISIES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --force)  FORCE=1 ;;
    --oui)    OUI=1 ;;
    --liste)  printf '%s\n' "${ETAPES[@]}"; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    --depuis)
      shift
      depart="${1:-}"
      trouve=0
      for e in "${ETAPES[@]}"; do
        [ "$e" = "$depart" ] && trouve=1
        [ "$trouve" -eq 1 ] && CHOISIES+=("$e")
      done
      if [ "$trouve" -eq 0 ]; then echo "étape inconnue : $depart" >&2; exit 1; fi
      ;;
    -*) echo "option inconnue : $1" >&2; usage; exit 1 ;;
    *)
      if ! printf '%s\n' "${ETAPES[@]}" | grep -qx "$1"; then
        echo "étape inconnue : $1" >&2; usage; exit 1
      fi
      CHOISIES+=("$1")
      ;;
  esac
  shift
done
[ ${#CHOISIES[@]} -eq 0 ] && CHOISIES=("${ETAPES[@]}")

mkdir -p export
debut=$(date +%s)
log "=== pipeline : ${CHOISIES[*]} ==="
for e in "${CHOISIES[@]}"; do "etape_$e"; done
log "=== terminé en $(( ($(date +%s) - debut) / 60 )) min ==="
printf '\n\033[1mTerminé.\033[0m Journal : %s\n' "$JOURNAL"
