#!/usr/bin/env bash
# Host Gemma 3 27B IT (google/gemma-3-27b-it, bf16) with vLLM as an
# OpenAI-compatible server for the meme2graph pipeline.
#
# The pipeline (agentflow/models.py) sends the meme plus up to 4 few-shot demo
# images as base64 data URIs, so --limit-mm-per-prompt must allow >= 5 images.
# It also sends response_format=json_schema; recent vLLM honours that natively.
#
# Memory notes:
#   - bf16 weights ~54 GB -> ~27 GB/GPU at TP=2, fits a 2x48 GB box with room.
#   - The blocker on 2 GPUs is the *sampler warmup*, not the weights: Gemma 3's
#     262K-token vocab makes the warmup batch's logits buffers large, and it is
#     sized by --max-num-seqs. Newer vLLM warms up more aggressively than it
#     used to, so keep --max-num-seqs small (this pipeline needs only ~4
#     concurrent requests anyway).
#   - "CUDA out of memory ... warming up sampler with N dummy requests" -> lower
#     --max-num-seqs further (2), add --enforce-eager, or drop
#     --gpu-memory-utilization to 0.80.
#   - Last resort for a tighter box: Google's QAT int4 build
#     google/gemma-3-27b-it-qat-q4_0-gguf (~15 GB) with
#     `--tokenizer google/gemma-3-27b-it --served-model-name google/gemma-3-27b-it`.
#
# Wiring (on the machine that runs run.py):
#   meme2graph/.env  ->  LOCAL_VLM_URL=http://<this-host>:8000/v1
#   config models block:  model_id: google/gemma-3-27b-it
#
# Gated repo: accept the license on HF, then `huggingface-cli login` or export
# HF_TOKEN before running.
#
# Usage:
#   bash experimental/vllm_gemma_27b.sh
#   HF_TOKEN=hf_xxx PORT=8000 TP=2 bash experimental/vllm_gemma_27b.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP=${TP:-2}
# Lower MAX_NUM_SEQS (2) if vLLM OOMs "warming up sampler".
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}

vllm serve google/gemma-3-27b-it \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --limit-mm-per-prompt '{"image":5,"video":0}' \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
