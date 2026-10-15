#!/usr/bin/env bash
# Finetune the released WildDet3D checkpoint on standard data mixed with a
# custom dataset, then evaluate the final checkpoint and export all metrics.
#
#   scripts/train.sh --custom DIR [--ratio R] [--out DIR] [options]
#
#   --custom DIR        custom dataset directory (see dv/datasets/README.md)   [required]
#   --ratio R           fraction of each batch drawn from the custom train set  [0.2]
#                       0 = standard data only (control, A1); 1 = custom only
#   --out DIR           output directory                [$DV_OUT/<dataset>_ratio<R>]
#   --gpus N            number of GPUs                                           [8]
#   --epochs N          epochs; LR x0.1 after 2/3 of them                        [3]
#   --samples N         global samples per epoch, same for every ratio         [8000]
#   --lr X              base learning rate (pretrained parts use 0.1x)         [1e-4]
#   --train-split NAME  custom train split   [the single annotations/*_train.json]
#   --test-split NAME   custom test split    [the single annotations/*_test.json]
#   --depth-scale S     PNG depth units per meter (1000 = millimeters)         [1000]
#   --no-val-omni3d     skip SUNRGBD_test validation each epoch (faster)
#   --no-eval           skip the final box/text evaluation
#   --overfit           sanity check: custom data only, 10 epochs x 800 samples,
#                       validated and evaluated on the *train* split
#
# Output (--out):
#   checkpoints/         last.ckpt (+ the latest epoch=*.ckpt)
#   metrics/train.csv    loss terms and LR every 10 steps
#   metrics/val.csv      validation AP etc. after every epoch
#   results/{box,text}/  final evaluation of last.ckpt (logs, per-class AP)
#   results/summary.md   final AP per test set and prompt mode (+ summary.json)
#   logs/train.log       full stdout; log_*.txt / config_*.yaml: vis4d's own
#   events.out.tfevents* tensorboard (tensorboard --logdir OUT)
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

CUSTOM=""; RATIO=0.2; OUT=""; GPUS=8; EPOCHS=""; SAMPLES=""; LR=""
TRAIN=""; TEST=""; DEPTH_SCALE=""; VAL_OMNI3D=1; EVAL=1; OVERFIT=0
while [ $# -gt 0 ]; do
    case $1 in
        --custom) CUSTOM=$2; shift 2 ;;
        --ratio) RATIO=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --gpus) GPUS=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --samples) SAMPLES=$2; shift 2 ;;
        --lr) LR=$2; shift 2 ;;
        --train-split) TRAIN=$2; shift 2 ;;
        --test-split) TEST=$2; shift 2 ;;
        --depth-scale) DEPTH_SCALE=$2; shift 2 ;;
        --no-val-omni3d) VAL_OMNI3D=0; shift ;;
        --no-eval) EVAL=0; shift ;;
        --overfit) OVERFIT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
[ -n "$CUSTOM" ] || { echo "--custom DIR is required" >&2; usage; exit 1; }
CUSTOM="$(realpath "$CUSTOM")"                 # resolve before env.sh cd's
[ -z "$OUT" ] || OUT="$(realpath -m "$OUT")"
source "$HERE/env.sh"
dv_custom "$CUSTOM" "$TRAIN" "$TEST"
[ -z "$DEPTH_SCALE" ] || export DV_CUSTOM_DEPTH_SCALE="$DEPTH_SCALE"

if [ "$OVERFIT" = 1 ]; then
    RATIO=1; EPOCHS=${EPOCHS:-10}; SAMPLES=${SAMPLES:-800}; VAL_OMNI3D=0
    export DV_CUSTOM_TEST="$DV_CUSTOM_TRAIN"
fi
NAME="$(basename "$CUSTOM")_$([ "$OVERFIT" = 1 ] && echo overfit || echo "ratio$RATIO")"
OUT=${OUT:-$DV_OUT/$NAME}
if [ -e "$OUT/checkpoints/last.ckpt" ]; then
    echo "$OUT already holds a finished run; pick another --out" >&2; exit 1
fi
mkdir -p "$OUT/logs"

export DV_OUTPUT_DIR="$OUT" DV_EXP_NAME="$NAME" DV_GPUS="$GPUS" \
    DV_RATIO="$RATIO" DV_EPOCHS="$EPOCHS" DV_SAMPLES_PER_EPOCH="$SAMPLES" \
    DV_LR="$LR" DV_VAL_OMNI3D="$VAL_OMNI3D"
echo "$(date +%T) train $NAME: custom=$DV_CUSTOM_ROOT ($DV_CUSTOM_TRAIN / $DV_CUSTOM_TEST)" \
     "ratio=$RATIO gpus=$GPUS -> $OUT"
set +e
vis4d fit --config "$DV/dv/configs/finetune.py" --gpus "$GPUS" \
    --ckpt "$CKPT_RELEASED" 2>&1 | tee "$OUT/logs/train.log"
rc=${PIPESTATUS[0]}
set -e
python "$DV/tools/export_metrics.py" run "$OUT" || true
[ "$rc" -eq 0 ] || { echo "training failed ($rc), see $OUT/logs/train.log" >&2; exit "$rc"; }

if [ "$EVAL" = 1 ]; then
    extra=(); [ "$OVERFIT" = 1 ] && extra=(--no-omni3d)
    for prompt in box text; do
        bash "$HERE/eval.sh" --ckpt "$OUT/checkpoints/last.ckpt" \
            --custom "$CUSTOM" --test-split "$DV_CUSTOM_TEST" \
            --prompt "$prompt" --gpus "$GPUS" --out "$OUT/results/$prompt" \
            "${extra[@]}"
    done
    python "$DV/tools/export_metrics.py" run "$OUT"
fi
echo "$(date +%T) done: $OUT"
