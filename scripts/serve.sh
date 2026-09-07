#!/usr/bin/env bash
# Ironclad production pin: Flash-Next NVFP4 on SGLang, :8000, 64k ctx, 8 slots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
load_hf_token
require_spark

SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"
PAGE_SIZE="${PAGE_SIZE:-64}"
MAMBA_STRATEGY="${MAMBA_STRATEGY:-extra_buffer}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

if [[ "${SPEC:-nextn}" == "off" ]]; then
  SPEC_ARGS=()
else
  SPEC_ARGS=(
    --speculative-algorithm NEXTN
    --speculative-num-steps "${SPEC_STEPS}"
    --speculative-eagle-topk "${SPEC_TOPK}"
    --speculative-num-draft-tokens "${SPEC_DRAFT}"
    --speculative-draft-model-quantization unquant
  )
fi

docker image inspect "${IMAGE}" >/dev/null
[[ -f "${QWEN4_BACKEND}" && -f "${QSA_BACKEND}" ]] || {
  echo "patches missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}
[[ -f "${SNAPSHOT}/config.json" ]] || {
  echo "checkpoint missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}
[[ -f "${BUILD}/path_qwen4_exp.txt" && -f "${BUILD}/path_qsa.txt" ]] || {
  echo "in-image paths missing. run ${SCRIPT_DIR}/prepare.sh first." >&2
  exit 1
}

QWEN4_IN_IMAGE="$(cat "${BUILD}/path_qwen4_exp.txt")"
QSA_IN_IMAGE="$(cat "${BUILD}/path_qsa.txt")"
UIDGID="$(docker_user)"
extra_gpu_groups
mkdir -p "${PLE_DIR}" "${SGLANG_CACHE}"

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

docker run -d --name "${CONTAINER}" --init \
  --restart "${RESTART_POLICY}" \
  --user "${UIDGID}" \
  "${extra_groups[@]}" \
  --gpus all \
  --ipc host \
  --shm-size 16g \
  --memory 116g \
  --memory-swap 116g \
  --cpuset-cpus "${CPUSET}" \
  --workdir /tmp \
  -p "${BIND_ADDR}:${PORT}:30000" \
  -e HOME=/tmp \
  -e PYTHONUNBUFFERED=1 \
  -e HF_TOKEN \
  -e HF_HOME=/huggingface \
  -e SGLANG_QWEN4_PLE_MMAP_DIR=/ple \
  -v "${HF_CACHE}:/huggingface" \
  -v "${SGLANG_CACHE}:/tmp/.cache/sglang" \
  -v "${PLE_DIR}:/ple" \
  -v "${QWEN4_BACKEND}:${QWEN4_IN_IMAGE}:ro" \
  -v "${QSA_BACKEND}:${QSA_IN_IMAGE}:ro" \
  "${IMAGE}" \
  sglang serve \
    --model-path "${MODEL}" \
    --revision "${REVISION}" \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size "${PAGE_SIZE}" \
    --mamba-radix-cache-strategy "${MAMBA_STRATEGY}" \
    --mamba-track-interval 64 \
    --max-mamba-cache-size "${MAMBA_CACHE}" \
    --mamba-ssm-dtype float32 \
    --chunked-prefill-size "${PREFILL}" \
    --max-running-requests "${MAX_RUNNING}" \
    --max-total-tokens "${MAX_TOTAL}" \
    --context-length "${CONTEXT}" \
    --mem-fraction-static "${MEMFRAC}" \
    --allow-auto-truncate \
    --ple-offload-embedding \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
    --prefill-attention-backend triton \
    --decode-attention-backend trtllm_mha \
    --disable-prefill-cuda-graph \
    --disable-flashinfer-autotune \
    --enable-metrics \
    --enable-cache-report \
    --enable-gdn-replayssm-spec \
    "${SPEC_ARGS[@]}" \
    "${EXTRA[@]}"

echo "started ${CONTAINER} on ${BIND_ADDR}:${PORT} as ${UIDGID} (model ${SERVED_NAME})"
echo "first PLE fill 45-60 min; later boots ~10 min. poll: ${SCRIPT_DIR}/wait_ready.sh"
