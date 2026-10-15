#!/usr/bin/env bash
# Reproduce the paper's Omni3D numbers with the repo's official eval configs,
# restricted to the subsets we have locally (default SUNRGBD_test).
#
#   scripts/repro_omni3d.sh [--modes M,...] [--sets S,...] [--gpus N] [--out DIR]
#
#   --modes  text,box_prompt,text_with_depth,box_prompt_with_depth      [all four]
#   --sets   Omni3D test subsets present under data/                  [SUNRGBD_test]
#   --out    output directory; one subdirectory per mode    [$DV_OUT/repro_omni3d]
#
# Output: <out>/<mode>/{eval.log,results.json,...}, <out>/compare.md.
# Paper (SUNRGBD AP): text 38.9, box 43.1, text+depth 51.1, box+depth 58.7.
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

MODES=text,box_prompt,text_with_depth,box_prompt_with_depth
SETS=SUNRGBD_test; GPUS=8; OUT=""
while [ $# -gt 0 ]; do
    case $1 in
        --modes) MODES=$2; shift 2 ;;
        --sets) SETS=$2; shift 2 ;;
        --gpus) GPUS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
[ -z "$OUT" ] || OUT="$(realpath -m "$OUT")"
source "$HERE/env.sh"
OUT=${OUT:-$DV_OUT/repro_omni3d}

runs=()
for mode in ${MODES//,/ }; do
    d="$OUT/$mode"; runs+=("$d"); mkdir -p "$d"
    echo "$(date +%T) repro $mode -> $d"
    DV_OUTPUT_DIR="$d" DV_REPRO_MODE=$mode DV_REPRO_SETS=$SETS vis4d test \
        --config "$DV/dv/configs/repro_omni3d.py" --gpus "$GPUS" \
        --ckpt "$CKPT_RELEASED" > "$d/eval.log" 2>&1 \
        || { echo "$mode failed, see $d/eval.log" >&2; continue; }
    python "$DV/tools/export_metrics.py" run "$d"
done
python "$DV/tools/export_metrics.py" compare "${runs[@]}" --to "$OUT/compare.md"
