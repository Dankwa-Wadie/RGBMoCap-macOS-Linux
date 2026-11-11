#!/usr/bin/env bash
# Merge new commits from the original Windows repo (toyxyz/toyxyz_RGBMocap) into this fork's branch.
#   tools/sync_upstream.sh            merge upstream/main into the current branch
# Conflicts usually appear where the port touched upstream files; the port's
# platform code lives in mocap/platform_compat.py, cpu_skin.py and torch_mps.py.
set -euo pipefail
cd "$(dirname "$0")/.."
git remote get-url upstream >/dev/null 2>&1 || git remote add upstream https://github.com/toyxyz/toyxyz_RGBMocap.git
git fetch upstream main
new=$(git rev-list --count HEAD..upstream/main)
if [ "$new" = 0 ]; then echo "Already up to date with upstream/main."; exit 0; fi
echo "Upstream has $new new commit(s):"; git log --oneline HEAD..upstream/main | head -20
git merge --no-edit upstream/main
echo "Merged. Re-run the checks: PYTHONPATH=. .venv/bin/python -m pytest -q tests_macos"
