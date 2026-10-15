#!/usr/bin/env bash
# One-time setup of the workspace (the parent directory of this repo):
#   1. WildDet3D repo (+ submodules, MoGe pinned) next to this codebase
#   2. micromamba env with its own CUDA 12.1 toolkit
#   3. released checkpoint + LingBot-Depth weights
# Data is downloaded separately: scripts/download_data.sh [DATA_DIR].
set -euo pipefail
DV="$(dirname "$(realpath "$0")")"
WS="$(dirname "$DV")"
REPO="$WS/WildDet3D"
ENV="$WS/envs/wilddet3d"
mkdir -p "$WS/envs"
exec > >(tee -a "$WS/envs/setup_env.log") 2>&1

# Download caches: keep any that are already configured, else <workspace>/.cache.
CACHE="$WS/.cache"
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$CACHE/micromamba}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$CACHE/pip}
export HF_HOME=${HF_HOME:-$CACHE/huggingface}
MM="$(command -v micromamba || echo "$CACHE/bin/micromamba")"
if [ ! -x "$MM" ]; then
    mkdir -p "$CACHE"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj -C "$CACHE" bin/micromamba
fi

# ---- 1. repo -------------------------------------------------------------
[ -d "$REPO" ] || git clone --recurse-submodules https://github.com/allenai/WildDet3D "$REPO"
[ -d "$REPO/third_party/moge" ] || git clone -q https://github.com/microsoft/moge.git "$REPO/third_party/moge"
# MoGe V3 (2026-08) changed the loss signatures WildDet3D calls; pin the last pre-V3 commit.
git -C "$REPO/third_party/moge" checkout -q 925b8ed
# The repo reads data/ relative to its root; download_data.sh links it.
# Keep that link out of the repo's git status.
grep -qx data "$REPO/.git/info/exclude" || echo data >> "$REPO/.git/info/exclude"

# ---- 2. env --------------------------------------------------------------
[ -d "$ENV" ] || $MM create -y -p "$ENV" -c nvidia/label/cuda-12.1.1 -c conda-forge python=3.11 cuda-toolkit gxx_linux-64=11 git
PY="$ENV/bin/python"
export CUDA_HOME="$ENV" PATH="$ENV/bin:$PATH" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"  # 9.0 = H100

$PY -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
$PY -m pip install vis4d==1.0.0
$PY -m pip install git+https://github.com/SysCV/vis4d_cuda_ops.git --no-build-isolation --no-cache-dir
$PY -m pip install -r "$REPO/requirements.txt"
$PY -m pip install remotezip h5py "transformers<5"  # transformers 5.x needs torch>=2.6
$PY -c "import torch, vis4d, vis4d_cuda_ops; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- 3. weights ----------------------------------------------------------
# SAM3 weights are not needed: the released checkpoint contains every weight.
cd "$REPO" && $PY - <<'EOF'
from huggingface_hub import hf_hub_download
print(hf_hub_download("allenai/WildDet3D", "wilddet3d_alldata_all_prompt_v1.0.pt", local_dir="ckpt"))
print(hf_hub_download("robbyant/lingbot-depth-postrain-dc-vitl14", "model.pt",
                      local_dir="pretrained/lingbot-depth/postrain-dc-vitl14"))
EOF
