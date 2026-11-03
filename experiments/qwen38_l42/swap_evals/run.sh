#!/usr/bin/env bash
# Score each AV under each AR (--eval-only, no update) on 512 held-out prompts (the first 128 are the in-run eval set),
# one 1xB200 container per pair. Usage: run.sh <av> <ar>, each "start" (the published 3.6 NLA's) or a path under
# /data/runs, e.g.  run.sh q38_l42_rl100_4x/iter_000200 start   or   run.sh start q38_l42_rl100_4x/critic_latest
set -euo pipefail
cd "$(dirname "$0")/../../.."
av=${1:?av}; ar=${2:?ar}
tag() { [[ $1 == start ]] && echo start || echo "${1%%/*}"; }
args=(--name "q38_l42_eval_av-$(tag "$av")_ar-$(tag "$ar")" --eval-only --eval-every 1 --eval-n-prompts 512)
[[ $av == start ]] || args+=(--av-lora "$av")
[[ $ar == start ]] || args+=(--ar "$ar")
NLA_GPUS=B200 modal run experiments/qwen38_l42/modal_app.py::rl "${args[@]}"
