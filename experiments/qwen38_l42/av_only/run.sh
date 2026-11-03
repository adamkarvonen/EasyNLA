#!/usr/bin/env bash
# AV-only RL: the AR stays the published (Qwen3.6) one; only the AV updates. Same batch and hyperparameters as
# cotrain/, 20 optimizer steps (40 rollout batches), held-out eval every 5 optimizer steps on the cotrain/ eval prompts.
set -euo pipefail
cd "$(dirname "$0")/../../.."
NLA_GPUS=B200:4 modal run --detach experiments/qwen38_l42/modal_app.py::rl --name q38_l42_av_only \
  --n-gpus 4 --num-steps 40 --batch-prompts 252 --grad-accum 2 --eval-every 10 --eval-n-prompts 128 --save-every 20 \
  --no-train-critic
