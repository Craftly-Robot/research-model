#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-artifacts/craftly/checkpoints/active}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN"
