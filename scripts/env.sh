# Source me: shared environment for all data_verify scripts.
DV="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]:-$0}")")")"  # this codebase
WS="$(dirname "$DV")"                     # workspace: shared repo, data, envs, outputs
REPO="$WS/WildDet3D"
export WS DV REPO
export PATH="$WS/envs/wilddet3d/bin:$PATH"
export PYTHONPATH="$DV:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$WS/.cache/huggingface}   # keep if already set
export TORCH_HOME=${TORCH_HOME:-$WS/.cache/torch}
export MIXED_PRECISION=${MIXED_PRECISION:-bf16}
# Default parent of run directories (each script also takes --out).
export DV_OUT=${DV_OUT:-$WS/outputs/data_verify}
export DV_CUSTOM_DEPTH_SCALE=${DV_CUSTOM_DEPTH_SCALE:-1000}

# dv_custom DIR [TRAIN_SPLIT] [TEST_SPLIT]
# Point the configs at a custom dataset directory (see dv/datasets/README.md).
# Splits default to the single annotations/*_train.json / *_test.json found.
dv_custom() {
    local root train test
    root="$(realpath "$1")"
    [ -d "$root/annotations" ] || { echo "no annotations/ in $root" >&2; return 1; }
    train=${2:-$(_dv_split "$root" train)} || return 1
    test=${3:-$(_dv_split "$root" test)} || return 1
    export DV_CUSTOM_ROOT="$root" DV_CUSTOM_TRAIN="$train" DV_CUSTOM_TEST="$test"
}
_dv_split() {  # works when sourced from bash or zsh
    [ -n "${ZSH_VERSION:-}" ] && setopt localoptions nullglob
    local f hit="" n=0
    for f in "$1"/annotations/*_"$2".json; do
        [ -f "$f" ] && { n=$((n + 1)); hit="$f"; }
    done
    if [ "$n" -ne 1 ]; then
        echo "need exactly one annotations/*_$2.json in $1 (or pass the split name)" >&2
        return 1
    fi
    basename "$hit" .json
}
CKPT_RELEASED="$REPO/ckpt/wilddet3d_alldata_all_prompt_v1.0.pt"
export CKPT_RELEASED
# The repo resolves data/ and pretrained/ relative to its root.
cd "$REPO"
