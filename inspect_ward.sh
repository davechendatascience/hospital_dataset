#!/usr/bin/env bash
# Convenience wrapper: inspect Ward0505.usd via Isaac Sim's python.
# Usage: ./inspect_ward.sh  [path-to-usd, default = Collected_Ward0505/Ward0505.usd]
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-$HERE/Collected_Ward0505/Ward0505.usd}"
"$HOME/isaac-sim/python.sh" "$HERE/inspect_stage.py" "$STAGE"
