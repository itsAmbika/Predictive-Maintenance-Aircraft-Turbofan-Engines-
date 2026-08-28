#!/usr/bin/env bash
#
# Push a deploy tree to a Hugging Face Docker Space.
#
#   ./deploy/huggingface/sync.sh <hf-username>/<space-name>
#
# Only what the image needs is pushed -- not the notebooks, tests, docs, or the
# ~30MB of raw C-MAPSS files (one 2.2MB sample is kept so the demo works). The
# Space gets its own README, the one with the YAML card metadata Spaces require,
# so the GitHub README stays clean.
#
# Auth: set HF_TOKEN (a write token from https://huggingface.co/settings/tokens),
# or let git prompt for credentials.
set -euo pipefail

SPACE_ID="${1:-${HF_SPACE_ID:-}}"
if [ -z "$SPACE_ID" ]; then
  echo "usage: $0 <hf-username>/<space-name>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

if [ -n "${HF_TOKEN:-}" ]; then
  REMOTE="https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}"
else
  REMOTE="https://huggingface.co/spaces/${SPACE_ID}"
fi

echo "==> cloning Space $SPACE_ID"
git clone --quiet "$REMOTE" "$WORK/space"
cd "$WORK/space"

# Clear the previous deploy, keeping git metadata and HF's LFS rules.
find . -mindepth 1 -maxdepth 1 ! -name .git ! -name .gitattributes -exec rm -rf {} +

echo "==> copying deploy payload"
for path in Dockerfile .dockerignore pyproject.toml uv.lock \
            src api conf models artifacts reports frontend frontend-react; do
  cp -r "$ROOT/$path" "./$(basename "$path")"
done

# Build inputs only -- the image builds the frontend itself.
rm -rf frontend-react/node_modules frontend-react/dist

# One real C-MAPSS file so visitors can hit "Try sample data".
mkdir -p data/raw
cp "$ROOT/data/raw/test_FD001.txt" data/raw/test_FD001.txt

# The Space card (YAML frontmatter + description) replaces the project README.
cp "$ROOT/deploy/huggingface/README.md" README.md

echo "==> pushing"
git add -A
if git diff --cached --quiet; then
  echo "no changes to deploy"
  exit 0
fi
git -c user.name="rul-deploy" -c user.email="deploy@local" \
    commit -qm "Deploy from $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo local)"
git push --quiet origin main

echo "==> done: https://huggingface.co/spaces/${SPACE_ID}"
echo "    the Space rebuilds automatically; watch the Logs tab (first build ~10-15 min)"
