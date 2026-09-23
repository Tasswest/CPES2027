#!/usr/bin/env bash
# Rejoue l'étude sur les deux corpus sans featurings, une fois la re-collecte
# (19_collecte_corpus.py) terminée.
#
#   option 1 — sans_feats   : titres avec invité écartés en entier
#   option 2 — sans_invites : strophes d'invités retirées
#
# Chaque variante a son propre cache (.cache_stylo_<variante>/) et ses propres
# résultats (export/<variante>/) : le corpus publié et ses chiffres restent
# intacts. Journal : export/<variante>/journal.txt.
#
# Usage : ./rejouer_variantes.sh [sans_feats] [sans_invites]
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -gt 0 ]; then VARIANTES=("$@"); else VARIANTES=(sans_feats sans_invites); fi

# Ordre imposé par les dépendances : 05 lit 03, 06/13/14 lisent 04,
# 15 lit 13, 16 lit les crédits de export/.
SCRIPTS=(
  02_diagnostic_biais_taille.py
  03_validation_protocole.py
  04_attribution_ziak.py
  05_robustesse.py
  06_profil_stylistique.py
  10_test_ziak_mikeysem.py
  13_validation_alias_reels.py
  14_test_alias_temporel.py
  15_test_ziak_7jaws.py
  16_test_2025_web7.py
)

python3 20_corpus_sans_invites.py

for v in "${VARIANTES[@]}"; do
  mkdir -p "export/$v"
  journal="export/$v/journal.txt"
  : > "$journal"
  # Les comptes dérivés (sans ad-libs) ne sont pas reconstruits par
  # stylo_features.py : les effacer, sinon ils dateraient d'un corpus antérieur.
  rm -f ".cache_stylo_$v"/*sans_parentheses*
  echo "=== $v : cache ===" | tee -a "$journal"
  CORPUS_VARIANTE="$v" python3 stylo_features.py >> "$journal" 2>&1
  for s in "${SCRIPTS[@]}"; do
    echo "=== $v : $s ($(date +%H:%M)) ===" | tee -a "$journal"
    CORPUS_VARIANTE="$v" python3 "$s" >> "$journal" 2>&1
  done
  echo "=== $v : terminé ($(date +%H:%M)) ===" | tee -a "$journal"
done
