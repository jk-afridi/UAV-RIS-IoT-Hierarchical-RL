#!/usr/bin/env bash
# Push the SMALL result files (json/pdf/tex/log/csv/png/txt) from Colab to a NEW branch of your GitHub repo.
# Model weights (*.pt etc.) are never pushed. Usage (from the repo root on Colab):
#     export GH_TOKEN=...            # fine-grained token, Contents: read+write on this repo only
#     bash scripts/push_results.sh [branch-name]
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/jk-afridi/UAV-RIS-IoT-Hierarchical-RL.git}"
BASE_BRANCH="${BASE_BRANCH:-feat/paper-v2}"
RESULTS_DIR="${RESULTS_DIR:-results}"
MAX_FILE_MB="${MAX_FILE_MB:-20}"
MAX_TOTAL_MB="${MAX_TOTAL_MB:-60}"
BRANCH="${1:-results-colab-$(date +%Y%m%d-%H%M)}"
: "${GH_TOKEN:?set GH_TOKEN first}"

ORIG="$PWD"
CODE_COMMIT="$(git -C "$ORIG" rev-parse HEAD 2>/dev/null || echo unknown)"
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
AUTH="Authorization: Basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)"
gitauth() { git -c "http.extraHeader=$AUTH" "$@"; }

gitauth clone -q --depth 1 --branch "$BASE_BRANCH" "$REPO_URL" "$WORK/repo"
git -C "$WORK/repo" checkout -q -b "$BRANCH"
DEST="$WORK/repo/colab_results"; mkdir -p "$DEST"

total=0; n=0
cd "$ORIG/$RESULTS_DIR"                       # works if results/ is a symlink to Google Drive
while IFS= read -r -d '' f; do
  sz=$(stat -c %s "$f")
  if [ "$sz" -gt $((MAX_FILE_MB * 1024 * 1024)) ]; then echo "skip (> ${MAX_FILE_MB} MB): $f"; continue; fi
  mkdir -p "$DEST/$(dirname "$f")"; cp "$f" "$DEST/$f"; total=$((total + sz)); n=$((n + 1))
done < <(find -L . -type f \( -name '*.json' -o -name '*.pdf' -o -name '*.tex' -o -name '*.log' \
                              -o -name '*.csv' -o -name '*.png' -o -name '*.txt' \) -print0)
cd "$WORK/repo"
[ "$n" -gt 0 ] || { echo "No result files found in $ORIG/$RESULTS_DIR"; exit 1; }
[ "$total" -le $((MAX_TOTAL_MB * 1024 * 1024)) ] || { echo "Too large: $((total / 1048576)) MB > ${MAX_TOTAL_MB} MB"; exit 1; }

{ echo "date: $(date -u +%FT%TZ)"; echo "code_commit: $CODE_COMMIT"; echo "python: $(python3 --version 2>&1)"
  python3 -c "import torch;print('torch:',torch.__version__,'cuda:',torch.cuda.is_available())" 2>/dev/null || true
  nvidia-smi -L 2>/dev/null || true; } > "$DEST/RUN_INFO.txt"

git add -f colab_results
git -c user.name="${GIT_NAME:-colab}" -c user.email="${GIT_EMAIL:-colab@users.noreply.github.com}" \
    commit -q -m "Colab results ($n files, code $CODE_COMMIT)"
gitauth push -q origin "$BRANCH"
echo "PUSHED branch: $BRANCH  ($n files, $((total / 1024)) KB)"
