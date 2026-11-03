#!/usr/bin/env bash
# Co-trained RL (EasyNLA's default: AV and AR both update), 100 optimizer steps on 4xB200: 252 prompts x 8 samples x
# grad-accum 2 = 504 prompts per update, as the 3.6 run. Held-out eval every 10 optimizer steps on 128 prompts.
# Ran as q38_l42_rl100_4x (fork 4cc0d56/ac04489), OOMed after the first save and was resumed from iter_000050 on
# ccb48d8 (the second command).
set -euo pipefail
cd "$(dirname "$0")/../../.."
APP=experiments/qwen38_l42/modal_app.py::rl
ARGS=(--n-gpus 4 --num-steps 200 --batch-prompts 252 --grad-accum 2 --eval-every 20 --eval-n-prompts 128 --save-every 50)
if [[ "${1:-}" == resume ]]; then
  NLA_GPUS=B200:4 modal run --detach $APP --name q38_l42_rl100_4x_r1 --resume q38_l42_rl100_4x/iter_000050 "${ARGS[@]}"
else
  NLA_GPUS=B200:4 modal run --detach $APP --name q38_l42_rl100_4x "${ARGS[@]}"
fi
