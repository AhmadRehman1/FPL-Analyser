#!/usr/bin/env bash
# Builds the static shell into public/ for Cloudflare (wrangler.toml points [assets] here).
# Usage: bash scripts/build_cloudflare_site.sh [out_dir]
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-public}"
LIST="scripts/site_files.txt"

rm -rf "$OUT"
mkdir -p "$OUT"

missing=0
while IFS= read -r path || [ -n "$path" ]; do
  case "$path" in ''|'#'*) continue ;; esac
  if [ ! -e "$path" ]; then
    echo "ERROR: expected site file missing: $path" >&2
    missing=1
    continue
  fi
  cp -R "$path" "$OUT/"
done < "$LIST"

if [ "$missing" -ne 0 ]; then
  exit 1
fi

# Cloudflare-only cache rules. GitHub Pages ignores this file, so it isn't in the shared list.
cp scripts/cloudflare/_headers "$OUT/_headers"

echo "Built $OUT/ ($(find "$OUT" -type f | wc -l) files, $(du -sh "$OUT" | cut -f1))"
