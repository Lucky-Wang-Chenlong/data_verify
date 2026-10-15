#!/usr/bin/env bash
# Evaluate a checkpoint on a custom test set and/or SUNRGBD_test.
#
#   scripts/eval.sh --ckpt CKPT|released [--custom DIR] [--prompt box|text] [options]
#
#   --ckpt CKPT       checkpoint path, or "released" (zero-shot baseline, A0)  [required]
#   --custom DIR      custom dataset directory; omit to evaluate SUNRGBD_test only
#   --test-split NAME custom test split        [the single annotations/*_test.json]
#   --prompt P        box (GT 2D box as prompt) | text (category name)          [box]
#   --depth           feed GT depth to the model
#   --no-omni3d       skip SUNRGBD_test
#   --gpus N          number of GPUs                                             [8]
#   --out DIR         output directory          [$DV_OUT/eval_<ckpt>_<prompt>]
#   --depth-scale S   PNG depth units per meter                               [1000]
#
# Output (--out): eval.log, results.json (all "3D/..." metrics; AP_<set> is
# the headline number per test set), log_*.txt with per-class AP tables,
# eval/detection/ predictions.
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

CKPT=""; CUSTOM=""; TEST=""; PROMPT=box; DEPTH=0; OMNI3D=1; GPUS=8; OUT=""
DEPTH_SCALE=""
while [ $# -gt 0 ]; do
    case $1 in
        --ckpt) CKPT=$2; shift 2 ;;
        --custom) CUSTOM=$2; shift 2 ;;
        --test-split) TEST=$2; shift 2 ;;
        --prompt) PROMPT=$2; shift 2 ;;
        --depth) DEPTH=1; shift ;;
        --no-omni3d) OMNI3D=0; shift ;;
        --gpus) GPUS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --depth-scale) DEPTH_SCALE=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
[ -n "$CKPT" ] || { echo "--ckpt is required" >&2; usage; exit 1; }
case $PROMPT in box|text) ;; *) echo "--prompt must be box or text" >&2; exit 1 ;; esac
[ -n "$CUSTOM" ] || [ "$OMNI3D" = 1 ] || { echo "nothing to evaluate" >&2; exit 1; }
# Resolve paths before env.sh changes the working directory.
[ "$CKPT" = released ] || CKPT="$(realpath "$CKPT")"
[ -z "$CUSTOM" ] || CUSTOM="$(realpath "$CUSTOM")"
[ -z "$OUT" ] || OUT="$(realpath -m "$OUT")"
source "$HERE/env.sh"
[ -z "$DEPTH_SCALE" ] || export DV_CUSTOM_DEPTH_SCALE="$DEPTH_SCALE"

if [ "$CKPT" = released ]; then
    tag=released; CKPT="$CKPT_RELEASED"
else
    tag="$(basename "$(dirname "$(dirname "$CKPT")")")"   # run dir of .../checkpoints/x.ckpt
fi
mode="$PROMPT$([ "$DEPTH" = 1 ] && echo _depth || true)"
OUT=${OUT:-$DV_OUT/eval_${tag}_$mode}
mkdir -p "$OUT"

if [ -n "$CUSTOM" ]; then
    dv_custom "$CUSTOM" unused "$TEST"   # the train split is not needed here
    export DV_EVAL_CUSTOM=1
else
    export DV_EVAL_CUSTOM=0
fi
export DV_OUTPUT_DIR="$OUT" DV_EXP_NAME="eval_$mode" DV_PROMPT="$PROMPT" \
    DV_WITH_DEPTH="$DEPTH" DV_EVAL_OMNI3D="$OMNI3D"
echo "$(date +%T) eval $mode: $CKPT -> $OUT"
set +e
vis4d test --config "$DV/dv/configs/evaluate.py" --gpus "$GPUS" \
    --ckpt "$CKPT" > "$OUT/eval.log" 2>&1
rc=$?
set -e
[ "$rc" -eq 0 ] || { echo "eval failed ($rc), see $OUT/eval.log" >&2; exit "$rc"; }
python "$DV/tools/export_metrics.py" run "$OUT"
