#!/usr/bin/env bash
# Check a custom dataset split (schema, geometry, depth) and render GT boxes.
#
#   scripts/validate.sh --custom DIR [--split NAME] [--vis N] [--out DIR]
#
#   --custom DIR     custom dataset directory (see dv/datasets/README.md)  [required]
#   --split NAME     annotation file stem      [every annotations/*.json in DIR]
#   --vis N          images to render per split                              [30]
#   --out DIR        render directory                      [$DV_OUT/vis/<split>]
#   --depth-scale S  PNG depth units per meter                             [1000]
#
# Exit code 1 if any split has ERRORs. Fix all of them before training.
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

CUSTOM=""; SPLIT=""; NVIS=30; OUT=""; DEPTH_SCALE=""
while [ $# -gt 0 ]; do
    case $1 in
        --custom) CUSTOM=$2; shift 2 ;;
        --split) SPLIT=$2; shift 2 ;;
        --vis) NVIS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --depth-scale) DEPTH_SCALE=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
[ -n "$CUSTOM" ] || { echo "--custom DIR is required" >&2; usage; exit 1; }
CUSTOM="$(realpath "$CUSTOM")"
[ -z "$OUT" ] || OUT="$(realpath -m "$OUT")"
source "$HERE/env.sh"
[ -z "$DEPTH_SCALE" ] || export DV_CUSTOM_DEPTH_SCALE="$DEPTH_SCALE"

if [ -n "$SPLIT" ]; then
    splits=("$SPLIT")
else
    splits=(); for f in "$CUSTOM"/annotations/*.json; do splits+=("$(basename "$f" .json)"); done
fi
rc=0
for s in "${splits[@]}"; do
    python -I "$DV/tools/validate_custom3d.py" --root "$CUSTOM" --name "$s" \
        --depth_scale "$DV_CUSTOM_DEPTH_SCALE" --vis "$NVIS" \
        --out "${OUT:-$DV_OUT/vis/$s}" || rc=1
done
exit $rc
