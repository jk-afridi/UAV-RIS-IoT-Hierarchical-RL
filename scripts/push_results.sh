#!/usr/bin/env bash
# Push the SMALL result files (json/pdf/png/tex/log/csv) from ./results to a NEW branch of your GitHub repo.
# Model weights (*.pt) and anything bigger than MAX_FILE_MB are skipped. A fresh branch is created every
# time (results-YYYYmmdd-HHMMSS), nothing is overwritten and your code branches are never touched.
#
#   export GITHUB_TOKEN=...            # fine-grained token, Contents: Read and write, THIS repo only
#   DRY=1 bash scripts/push_results.sh # list what would be pushed, push nothing
#   bash scripts/push_results.sh       # push
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/jk-afridi/UAV-RIS-IoT-Hierarchical-RL.git}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MAX_FILE_MB="${MAX_FILE_MB:-5}"
MAX_TOTAL_MB="${MAX_TOTAL_MB:-40}"
BRANCH="${BRANCH:-results-$(date -u +%Y%m%d-%H%M%S)}"
DRY="${DRY:-0}"

[ -d "$RESULTS_DIR" ] || { echo "No '$RESULTS_DIR' folder here. cd to the repo root first."; exit 1; }
if [ "$DRY" != "1" ] && [ -z "${GITHUB_TOKEN:-}" ]; then echo "Set GITHUB_TOKEN first (see header)."; exit 1; fi
SRC_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
total=0; n=0
while IFS= read -r -d '' f; do
  size=$(stat -c %s "$f")
  if [ "$size" -gt $((MAX_FILE_MB*1024*1024)) ]; then echo "skip (>${MAX_FILE_MB}MB): $f"; continue; fi
  rel="${f#"$RESULTS_DIR"/}"
  mkdir -p "$STAGE/results/$(dirname "$rel")"
  cp "$f" "$STAGE/results/$rel"
  total=$((total+size)); n=$((n+1))
done < <(find -L "$RESULTS_DIR" -type f \( -name '*.json' -o -name '*.pdf' -o -name '*.png' -o -name '*.tex' -o -name '*.log' -o -name '*.csv' \) -print0)

echo "files: $n   total: $((total/1024)) KB"
[ "$n" -gt 0 ] || { echo "Nothing to push."; exit 1; }
[ "$total" -le $((MAX_TOTAL_MB*1024*1024)) ] || { echo "Total > ${MAX_TOTAL_MB}MB: lower MAX_FILE_MB."; exit 1; }
if [ "$DRY" = "1" ]; then (cd "$STAGE" && find . -type f | sort | head -200); echo "dry run - nothing pushed"; exit 0; fi

echo "$SRC_COMMIT" > "$STAGE/SOURCE_COMMIT.txt"
cd "$STAGE"
git init -q
git checkout -q -b "$BRANCH"
git add -A
git -c user.name="colab" -c user.email="colab@users.noreply.github.com" commit -q -m "Experiment results ($n files, code commit $SRC_COMMIT)"
AUTH="$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 | tr -d '\n')"
git -c http.extraheader="AUTHORIZATION: basic $AUTH" push -q "$REPO_URL" "HEAD:refs/heads/$BRANCH"
echo "PUSHED branch: $BRANCH"
