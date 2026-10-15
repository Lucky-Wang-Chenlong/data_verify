#!/usr/bin/env bash
# Download the standard data (Omni3D subset) and build the toy custom set.
#
#   scripts/download_data.sh [DATA_DIR]                    [DATA_DIR = $WS/data]
#
# Then links WildDet3D/data -> DATA_DIR: the upstream code reads Omni3D from
# data/ relative to its root, so that link is how every script finds the data.
# Steps whose output exists are skipped, so re-running resumes. Needs ~25 GB
# while downloading (SUNRGBD.zip), ~13 GB afterwards:
#   DATA_DIR/omni3d/annotations/  SUNRGBD_{train,val,test}.json, ARKitScenes_*.json,
#                                 ARKitScenes_train_sub5k.json (5k subset, seed 0)
#   DATA_DIR/SUNRGBD.hdf5         SUNRGBD images + depth            (6.9 GB)
#   DATA_DIR/ARKitScenes.hdf5     images of the 5k subset            (2.2 GB)
#   DATA_DIR/ARKitScenes_depth.hdf5  depth from 3D-MOOD             (3.4 GB)
#   DATA_DIR/custom/toy/          toy custom set: chair from SUNRGBD_val
set -euo pipefail
HERE="$(dirname "$(realpath "$0")")"
D="$(realpath -m "${1:-$(dirname "$(dirname "$HERE")")/data}")"
mkdir -p "$D"
REPO_DATA="$(dirname "$(dirname "$HERE")")/WildDet3D/data"
if [ -e "$REPO_DATA" ] && [ ! -L "$REPO_DATA" ]; then
    echo "$REPO_DATA exists and is not a symlink; move it away first" >&2; exit 1
fi
ln -sfn "$D" "$REPO_DATA"
echo "data dir: $D  (WildDet3D/data -> $D)"
source "$HERE/env.sh"   # cd's into WildDet3D/, so data/ below is DATA_DIR
DL="$D/_downloads"
mkdir -p "$DL" "$D/omni3d/annotations"

fetch() { [ -f "$DL/$(basename "$1")" ] || wget -c -q -P "$DL" "$1"; }

# Omni3D annotations (SUNRGBD + ARKitScenes splits only).
if [ ! -f "$D/omni3d/annotations/SUNRGBD_test.json" ]; then
    fetch https://dl.fbaipublicfiles.com/omni3d_data/Omni3D_json.zip
    unzip -q -j -o "$DL/Omni3D_json.zip" 'datasets/Omni3D/SUNRGBD_*' \
        'datasets/Omni3D/ARKitScenes_*' -d "$D/omni3d/annotations"
fi

# SUNRGBD images + depth -> SUNRGBD.hdf5 (raw folder removed afterwards).
if [ ! -f "$D/SUNRGBD.hdf5" ]; then
    fetch https://rgbd.cs.princeton.edu/data/SUNRGBD.zip
    (cd "$D" && unzip -q -n "$DL/SUNRGBD.zip" -d . \
        && python -m vis4d.data.io.to_hdf5 -p SUNRGBD && rm -rf SUNRGBD)
fi

# ARKitScenes depth (from 3D-MOOD) + a 5k-image train subset (range requests,
# seed 0, only images that have depth) -> ARKitScenes.hdf5.
[ -f "$D/ARKitScenes_depth.hdf5" ] || wget -c -q -P "$D" \
    https://huggingface.co/datasets/RoyYang0714/3D-MOOD/resolve/main/ARKitScenes_depth.hdf5
if [ ! -f "$D/ARKitScenes.hdf5" ]; then
    python "$DV/tools/make_arkitscenes_subset.py" --data_root data --num 5000
    (cd "$D" && python -m vis4d.data.io.to_hdf5 -p ARKitScenes && rm -rf ARKitScenes)
fi

# Toy custom set (chair from SUNRGBD_val): format example + smoke tests.
[ -d "$D/custom/toy" ] || python "$DV/tools/make_toy_custom.py" \
    --category chair --out data/custom/toy --prefix Toy

rm -rf "$DL"
python "$DV/tools/check_omni3d_subset.py"
