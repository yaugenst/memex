#!/usr/bin/env bash
# Build an Apple Silicon app and its CLI from this checkout. No signing credentials needed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$ROOT/../.." && pwd)
cd "$REPO"
export MACOSX_DEPLOYMENT_TARGET=14.0
CARGO=(cargo)
if command -v mbx >/dev/null 2>&1; then
  CARGO=(mbx)
else
  echo "mbx unavailable; using uncached cargo builds." >&2
fi
"${CARGO[@]}" build --locked --release --target aarch64-apple-darwin --target-dir "$ROOT/build/cli"
ARCHES=arm64 MEMEX_CLI="$ROOT/build/cli/aarch64-apple-darwin/release/memex" "$ROOT/scripts/build.sh" release
