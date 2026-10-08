#!/usr/bin/env bash
# Prepare a GPU box for eval/run_eval.py. Runs as root on the box from main's harness checkout (eval/bot.py ships
# it there) and is safe to run again. Built for the vast.ai VM image docker.io/vastai/kvm:ubuntu_cli_22.04-2025-11-21,
# which has the NVIDIA driver (580, CUDA 13.0), Docker with the NVIDIA runtime, git and uv.
#
# Produces:
#   /data/venv                  the judge's Python: CPU torch, numpy, safetensors, openai-harmony, huggingface_hub
#                               (pinned in reference/requirements.txt)
#   /data/models/gpt-oss-20b    the checkpoint, verified against reference/weights.lock.json
#   gptoss-eval:1               the eval image (eval/image/Dockerfile)
#   /data/gptoss-eval/{runs,goldens}
set -euo pipefail
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA=/data
REQ="$HARNESS/reference/requirements.txt"
pin() { grep -E "^$1==" "$REQ"; }

echo "== GPU"
nvidia-smi --query-gpu=name,driver_version,power.limit,memory.total --format=csv,noheader
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
[ "${driver%%.*}" -ge 580 ] || { echo "driver $driver is older than 580: the eval image needs CUDA 13" >&2; exit 1; }

echo "== Docker with GPU access"
if ! docker info --format '{{json .Runtimes}}' | grep -q nvidia; then
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi

echo "== Judge environment"
command -v uv >/dev/null || pip3 install --quiet uv
[ -x "$DATA/venv/bin/python" ] || uv venv --quiet --python 3.12 "$DATA/venv"
uv pip install --quiet --python "$DATA/venv/bin/python" --index-url https://download.pytorch.org/whl/cpu "$(pin torch)"
uv pip install --quiet --python "$DATA/venv/bin/python" "$(pin numpy)" "$(pin safetensors)" "$(pin huggingface_hub)" \
    "$(pin openai-harmony)"
# The judge's whole import path, plus the harmony encoding, which openai-harmony downloads on first use.
PYTHONPATH="$HARNESS/reference:$HARNESS/eval" "$DATA/venv/bin/python" -c "
import policy, gptoss_ref.compare, gptoss_ref.golden, gptoss_ref.harmony_render
from openai_harmony import HarmonyEncodingName, load_harmony_encoding
load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)"

echo "== Weights"
"$DATA/venv/bin/python" "$HARNESS/reference/scripts/fetch_weights.py" --out "$DATA/models/gpt-oss-20b"

echo "== Eval image"
docker build --quiet -t gptoss-eval:1 "$HARNESS/eval/image"
docker run --rm --gpus all --network none gptoss-eval:1 nvidia-smi -L

mkdir -p "$DATA/gptoss-eval/runs" "$DATA/gptoss-eval/goldens"
echo "== provisioned"
