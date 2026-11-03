#!/usr/bin/env bash
# AR-only: the AV stays the published one; the AR is trained on its on-policy explanations, exactly as in cotrain/ but
# with no AV update (so no policy gradient and no weight sync). Same batch and hyperparameters, 20 optimizer steps
# (40 rollout batches), held-out eval every 5 optimizer steps on the cotrain/ eval prompts.
set -euo pipefail
cd "$(dirname "$0")/../../.."
NLA_GPUS=B200:4 modal run --detach experiments/qwen38_l42/modal_app.py::rl --name q38_l42_ar_only \
  --n-gpus 4 --num-steps 40 --batch-prompts 252 --grad-accum 2 --eval-every 10 --eval-n-prompts 128 --save-every 20 \
  --no-train-actor
