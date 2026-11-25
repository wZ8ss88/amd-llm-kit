#!/usr/bin/env bash
# serve_vllm.sh -- start a vLLM OpenAI-compatible server on AMD Instinct GPUs
# with a preflight VRAM check and the flags that matter on ROCm.
#
# Usage:
#   bash serve_vllm.sh <model> [max_model_len] [tp_size]
#
#   bash serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct 8192 1
#
# Why these flags (vLLM serve docs):
#   --gpu-memory-utilization  leaves headroom for the vision of a sudden
#                             allocation spike; 0.90 is the usual safe value.
#   --max-model-len           the cache is allocated from max_model_len, so a
#                             too-generous value is a silent OOM at startup.
#   --kv-cache-dtype fp8      halves the cache on CDNA3+ (gfx942/gfx950).
#   --tensor-parallel-size    must divide the attention head count.
#   ROCm-specific: HIP_VISIBLE_DEVICES picks cards, and RCCL (NCCL API) needs
#   a working ring between them for TP > 1.
#
# Nothing is guessed: every value below is a plain flag you could type.

set -euo pipefail

MODEL="${1:?usage: serve_vllm.sh <model> [max_model_len] [tp_size]}"
MAX_LEN="${2:-8192}"
TP="${3:-1}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm not on PATH -- install it in the venv you intend to use" >&2
  exit 1
fi

# 1. environment sanity: the doctor exits non-zero on FAIL lines
echo "== environment =="
if command -v python3 >/dev/null 2>&1; then
  python3 "$here/doctor.py" | tail -n 4 || true
fi

# 2. VRAM preflight, informational: refuse only when the card is already busy
echo
echo "== vram before start =="
if python3 "$here/vramwatch.py" --once --alert-pct 90; then
  :
else
  rc=$?
  if [ "$rc" -eq 2 ]; then
    echo "a card is already above 90% VRAM -- stop the other process or lower GPU_UTIL" >&2
    exit 2
  fi
fi

# 3. cache budget, so an absurd max_model_len fails fast and loudly
echo
echo "== planned cache (weights excluded) =="
if [ -n "${LAYERS:-}" ] && [ -n "${KV_HEADS:-}" ]; then
  python3 "$here/kvcalc.py" --layers "$LAYERS" --kv-heads "$KV_HEADS" \
    --head-dim "${HEAD_DIM:-128}" --seq "$MAX_LEN" --batch "${BATCH:-1}" \
    --dtype "${KV_DTYPE:-bf16}" || true
else
  echo "set LAYERS and KV_HEADS to get a cache estimate for this model"
fi

# 4. launch
args=(serve "$MODEL"
      --port "$PORT"
      --tensor-parallel-size "$TP"
      --max-model-len "$MAX_LEN"
      --gpu-memory-utilization "$GPU_UTIL"
      --dtype bfloat16)

if [ "${KV_DTYPE:-}" = "fp8" ]; then
  args+=(--kv-cache-dtype fp8)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  args+=($EXTRA_ARGS)
fi

echo
echo "== starting: vllm ${args[*]} =="
exec vllm "${args[@]}"
