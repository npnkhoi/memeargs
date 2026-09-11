#!/usr/bin/env bash
# Host Qwen2.5-VL 32B AWQ (Qwen/Qwen2.5-VL-32B-Instruct-AWQ) with vLLM as an
# OpenAI-compatible server for the meme2graph pipeline.
#
# The pipeline (agentflow/models.py) sends the meme plus up to 4 few-shot demo
# images as base64 data URIs, so --limit-mm-per-prompt must allow >= 5 images.
# It also sends response_format=json_schema; recent vLLM honours that natively.
#
# Wiring (on the machine that runs run.py):
#   meme2graph/.env  ->  LOCAL_VLM_URL=http://<this-host>:8000/v1
#   config models block:  model_id: Qwen/Qwen2.5-VL-32B-Instruct-AWQ
#
# Usage:
#   bash experimental/vllm_qwen_32B.sh
#   PORT=8000 TP=2 bash experimental/vllm_qwen_32B.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP=${TP:-2}
# Lower MAX_NUM_SEQS (32, 16, 8) if vLLM OOMs "warming up sampler".
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}

vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype float16 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":5,"video":0}' \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs "$MAX_NUM_SEQS"
