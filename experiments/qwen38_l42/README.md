# Qwen3.8-27B layer-42 NLA

Adapts the Qwen3.6-27B NLA ([ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl), layer 42)
to Qwen3.8-27B activations with `nla.train_rl_vllm`, starting from its RL step-600 AV adapter and AR (the published
matched pair). All arms share `modal_app.py`, the data, and the 3.6 run's hyperparameters; each subfolder's `run.sh`
is the exact launch command.

| folder | AV | AR | what it tests |
|---|---|---|---|
| `cotrain/` | trained | trained | EasyNLA's default RL (100 optimizer steps) |
| `av_only/` | trained | fixed (published) | can the explanations themselves improve on Qwen3.8? |
| `ar_only/` | fixed (published) | trained | how much of `cotrain/` is the AR adapting? |
| `swap_evals/` | - | - | each final AV scored under each AR (`--eval-only`) |

## Setup

- **Data** (`modal_app.py::download`, `::extract`): 20k FineFineWeb docs (equal share of its 67 domains, the 3.6 NLA's
  corpus), Qwen3.8 layer-42 activations at 10 positions/doc, RL rows built with the 3.6 NLA's own prompt templates
  (`/data/q38_L42_20k/rl_shuf.parquet` on the `adam-nla-finetune` Modal volume). 5% of docs are held out
  (doc-hash split); the eval set is their first N rows.
- **Engine**: vLLM 0.19 serves the original Qwen3.6-27B snapshot, then the first weight sync pushes the full AV
  (`--vllm-name-map qwen3_5_wrapper`, text-only, Triton attention on B200). Prompts use the chat template with
  thinking off, as the 3.6 NLA's card prescribes.
- **Batch**: 4xB200, 252 prompts x 8 samples per rollout batch, `--grad-accum 2`: 504 prompts per optimizer step, as
  the 3.6 run. Updates stay on-policy (weights sync at the optimizer step). `--num-steps`, `--eval-every` and
  `--save-every` count rollout batches, i.e. half optimizer steps.
- **Eval**: held-out FVE = 1 - MSE / MSE of predicting the eval set's mean activation (normalized activations), one
  sample per prompt at temperature 1, scored by the run's live AR. The baseline depends on the eval set, so compare
  FVEs only at the same `--eval-n-prompts`. The in-run eval at batch N runs after that batch; with grad-accum 2 and an even N
  no optimizer step has happened in it yet, so eval@2k scores the weights after k optimizer steps.

## Results

### cotrain (`q38_l42_rl100_4x`)

Held-out FVE (128 prompts) by optimizer step:

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 |
|---|---|---|---|---|---|---|---|---|---|---|
| FVE % | 65.5 | 68.6 | 69.7 | 69.8 | 70.3 | 70.2 | 70.6 | 70.7 | 70.9 | 71.0 |

Parse rate 99-100% and KL to the start 0.79-0.88 throughout. The run OOMed on rank 0 in the first batch after its
first save (batch 50, allocator fragmentation, 180 MB short) and was resumed from `iter_000050` after commit `ccb48d8`
(cache release after eval/save, vLLM memory 0.38 -> 0.36). ~354 s/batch before the fix, ~420-490 s after.
Final: `runs/q38_l42_rl100_4x/iter_000200` (AV) and `critic_latest` (AR, step 200).

### av_only, ar_only, swap_evals

Pending.
