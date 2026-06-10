#!/usr/bin/env bash
set -euo pipefail

cd ./ 

git init
git add .
git commit -m "Initial commit" || true

gh repo create TropicalGT \
    --public \
    --source=. \
    --remote=origin \
    --push 