#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
VLLM_URL=${VLLM_URL:-http://localhost:8100/v1}
SAVE_DIR=${SAVE_DIR:-outputs/surprisenav-wsivqa}

: "${QWEN_CKPT:?Set QWEN_CKPT to the Qwen checkpoint or served model identifier.}"
: "${PATHO_R1_CKPT:?Set PATHO_R1_CKPT to the Patho-R1 checkpoint.}"
: "${QWEN35_MODEL:?Set QWEN35_MODEL to the vLLM served model name.}"
: "${QUESTIONS_FILE:?Set QUESTIONS_FILE to the WSI-VQA question JSON.}"
: "${FEATURE_DIR_5X:?Set FEATURE_DIR_5X to the low-magnification feature directory.}"
: "${FEATURE_DIR_20X:?Set FEATURE_DIR_20X to the high-magnification feature directory.}"
: "${SVS_DIR:?Set SVS_DIR to the WSI directory.}"
: "${PLIP_CKPT:?Set PLIP_CKPT to the PLIP checkpoint.}"

cd "$REPO"

args=(
  main.py
  --qwen_ckpt "$QWEN_CKPT"
  --patho_r1_ckpt "$PATHO_R1_CKPT"
  --questions_file "$QUESTIONS_FILE"
  --feature_dir_5x "$FEATURE_DIR_5X"
  --feature_dir_20x "$FEATURE_DIR_20X"
  --svs_dir "$SVS_DIR"
  --yaad_dim 768
  --query_navigation
  --plip_ckpt "$PLIP_CKPT"
  --query_alpha 0.5
  --vllm_url "$VLLM_URL"
  --vllm_model "$QWEN35_MODEL"
  --vlm_device "${VLM_DEVICE:-cuda:0}"
  --plip_device "${PLIP_DEVICE:-cuda:0}"
  --no_mcq_debiasing
  --seed 42
  --top_k_jumps 10
  --max_reflection_rounds 1
  --max_vlm_calls 15
  --top_n_per_roi 2
  --multi_scale
  --save_dir "$SAVE_DIR"
)

if [[ -n "${RAG_INDEX:-}" ]]; then
  args+=(--rag_index "$RAG_INDEX")
fi

"$PYTHON" "${args[@]}"
