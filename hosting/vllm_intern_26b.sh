#!/usr/bin/env bash
# Host InternVL2.5 26B AWQ (OpenGVLab/InternVL2_5-26B-AWQ) with vLLM as an
# OpenAI-compatible server for the meme2graph pipeline.
#
# The pipeline (agentflow/models.py) sends the meme plus up to 4 few-shot demo
# images as base64 data URIs, so --limit-mm-per-prompt must allow >= 5 images.
# It also sends response_format=json_schema; recent vLLM honours that natively.
#
# Wiring (on the machine that runs run.py):
#   meme2graph/.env  ->  LOCAL_VLM_URL=http://<this-host>:8000/v1
#   config models block:  model_id: OpenGVLab/InternVL2_5-26B-AWQ
#
# Usage:
#   bash experimental/vllm_intern_26b.sh
#   PORT=8000 TP=2 bash experimental/vllm_intern_26b.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP=${TP:-2}
# Lower MAX_NUM_SEQS (16, 8) if vLLM OOMs "warming up sampler".
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}

# InternVL tiles each input image into up to `max_dynamic_patch` sub-images of
# 256 tokens. With 5 images per prompt, 6 tiles keeps the context under 32k;
# lower it if you still overflow, raise it for sharper OCR on dense memes.
vllm serve OpenGVLab/InternVL2_5-26B-AWQ \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --trust-remote-code \
  --quantization awq \
  --dtype float16 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":5,"video":0}' \
  --mm-processor-kwargs '{"max_dynamic_patch":6}' \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.90
