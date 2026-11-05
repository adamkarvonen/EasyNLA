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

Also here:
- `FINDINGS.md`: the short write-up (fine-tuning barely matters, and the gain is the AR's).
- `inference.py`: text -> activation -> explanation -> reconstruction.
- `MODEL_CARD.md`: the card of the released pair,
  [adamkarvonen/qwen3.8-27b-nla-L42](https://huggingface.co/adamkarvonen/qwen3.8-27b-nla-L42). It is published by
  `modal_app.py::release_check` (stage and test) and `::release_upload`.
- `BRINGUP.md`: the bring-up log (shims, thinking-prompt bug, smoke runs), moved from motivated-reasoning.

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

### swap_evals

Held-out FVE (%) on 512 prompts (eval-set baseline MSE 0.4414), `--eval-only`, commit `c3edbc3`:

| AV \ AR | published (start) | cotrain final (step 100) | ar_only final (step 20) |
|---|---|---|---|
| published (start) | 65.1 | 69.0 | 68.1 |
| cotrain final (step 100) | 65.2 | 69.6 | 68.2 |
| av_only final (step 20) | 65.1 | 68.6 | 67.3 |

The cotrain gain is almost all the AR's: the trained AR adds 3.9-4.4 points under either AV, and the trained AV adds
0.1-0.6 under either AR. The AV differences are within the noise of single-sample evals on 512 prompts (roughly
+-0.6 for a difference, estimated, not measured).

### ar_only (`q38_l42_ar_only`)

Held-out FVE (128 prompts) by optimizer step: 0: 66.6, 5: 68.8, 10: 68.8, 15: 69.3% (cotrain: 68.6 at 10, 69.7 at 20).
Its final AR (step 20) on the 512-prompt swap set: 68.1% under the published AV, 68.2% under the cotrain final AV:
+3.0 points over the published AR in 20 steps, against +3.9-4.4 for the cotrain AR in 100. Training the AR alone tracks
co-training step for step. ~176 s/batch (no policy update or weight sync).

### av_only (`q38_l42_av_only`)

Held-out FVE (128 prompts) by optimizer step, AR fixed at the published one: 0: 66.0, 5: 65.5, 10: 66.4, 15: 66.2%.
KL to the start 0.76-0.80, parse rate 99-100%. ~360-420 s/batch. Its final AV (step 20) on the 512-prompt swap set:
65.1% under the published AR, the same as the published AV; 68.6 and 67.3% under the two trained ARs, 0.4-0.8 below
the published AV. RL on the AV alone doesn't improve FVE in 20 steps. The small deficit under the trained ARs is
consistent with those ARs having been fit to the published AV's outputs, and is near the eval noise.

## Conclusion

See FINDINGS.md. In short: the gain is all the AR's, and the AV doesn't improve whether trained alone or with the AR.
Modal cost: $984 in all, of which $730 is the cotrain run (breakdown in MODEL_CARD.md).
