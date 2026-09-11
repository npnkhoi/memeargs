#!/usr/bin/env bash
# Host Mistral Small 3.2 24B (mistralai/Mistral-Small-3.2-24B-Instruct-2506) with
# vLLM as an OpenAI-compatible server for the meme2graph pipeline.
#
# The pipeline (agentflow/models.py) sends the meme plus up to 4 few-shot demo
# images as base64 data URIs, so --limit-mm-per-prompt must allow >= 5 images.
# It also sends response_format=json_schema; recent vLLM honours that natively.
#
# Wiring (on the machine that runs run.py):
#   meme2graph/.env  ->  LOCAL_VLM_URL=http://<this-host>:8000/v1
#   config models block:  model_id: mistralai/Mistral-Small-3.2-24B-Instruct-2506
#
# Gated repo: run `huggingface-cli login` or export HF_TOKEN first.
#
# Usage:
#   bash experimental/vllm_mistral_24b.sh
#   HF_TOKEN=hf_xxx PORT=8000 TP=2 bash experimental/vllm_mistral_24b.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TP=${TP:-2}

# Mistral 3.2 ships in Mistral's own checkpoint format; the three --*-format
# flags tell vLLM to read it. Drop them only for a Transformers-format mirror.
vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":5,"video":0}' \
  --gpu-memory-utilization 0.90
