# Bring-up log: Qwen3.6 NLA -> Qwen3.8 (2026-11-01 to 11-03)

Moved here from `motivated-reasoning/nla_finetune/README.md` (commit `96b93e5` there, which also holds the superseded
code it describes: `easynla_shim.py`, `easynla_vllm_shim.py`, `apply_lens_patch.py` and the first `modal_app.py`).
Those shims became the fork's own changes (`--vllm-name-map qwen3_5_wrapper`, `nla/utils/prompts.py`, the
merged-AR loader in `nla/models.py`) and `modal_app.py` here. Results: README.md and FINDINGS.md.

**Goal (Adam, 2026-11-01).** An NLA for Qwen3.8-27B that is good enough for the paper. Instead of training one from
scratch, continue the RL of an existing Qwen3.6-27B NLA on Qwen3.8 activations.

## Why this starting point

- **Checkpoint:** `ceselder/qwen3.6-27b-nla-rl`, layer 42, 78% held-out FVE on Qwen3.6, trained with EasyNLA.
  - Without fine-tuning it transfers to Qwen3.8: 64% FVE on web text and 44% on loan transcripts
    (motivated-reasoning `loan_rl/nla_transfer/README.md`).
  - The native Qwen3.8 NLA (`gereon/qwen3.8-27b-nla-L42`) gets 65% and 34% on the same sets.
- **Reconstructor co-training removes part of the drop.** EasyNLA keeps training the reconstructor (the AR, "critic")
  during RL. About 6 of the 13 points lost on 3.8 come from the 3.6 reconstructor landing in 3.6's space, and
  co-training addresses that.
- **The verbalizer stays on Qwen3.6 weights (`av_base`)** and reads 3.8 activations; the transfer study did the same.
  The 3.8 NLA card reports that the 3.6 verbalizer gives no parseable output when run on 3.8 weights.
- **Multi-layer NLAs exist but aren't usable yet.** `agu18dec/qwen3.6-27b-all-layer-nla-rl2000` covers all 64 layers
  and `camilablank/qwen3.6-27b-multilayer-nla-rl600` covers the 11-layer ladder. Both were trained with an
  unpublished EasyNLA fork, and we have asked the authors for it.

## How it works

Code: [ceselder/EasyNLA](https://github.com/ceselder/EasyNLA) @ `d23cba0` (EasyNLA plus Qwen3.6 support), unpatched,
installed in the Modal image. `easynla_shim.py` holds the only glue:

- **Extraction (`Q38Extractor`).** EasyNLA's `HFExtractor`, but the model loads with `AutoModelForImageTextToText`,
  the class loan_rl's transfer study used. The vector is the output of decoder block 42 (= HF `hidden_states[43]`),
  as in the 3.6 NLA's data.
- **Reconstructor (`load_merged_critic`).** The published reconstructor is a merged 43-block backbone saved under
  multimodal key names. EasyNLA's HF RL path rebuilds its reconstructor from the raw base model, which would discard
  the trained weights. The shim loads the merged backbone in its place.

Pipeline (`modal_app.py`, app `adam-nla-finetune`; volumes `adam-mr-hf-cache` and `adam-nla-finetune`):

1. **`download`.** Fetches the NLA repo files, plus 20k FineFineWeb docs taken evenly from its 67 domains (the dataset
   is stored by domain, so a plain prefix would be all aerospace).
2. **`extract`.** EasyNLA datagen:
   - stage 0: Qwen3.8 layer-42 activations, 10 positions per doc
   - stage 1: split 0.25 / 0.25 / 0.5, as in the 3.6 run
   - stage 3: build the RL rows with the 3.6 NLA's own prompt templates
   - shuffle

   Stage 2 (Claude explanations) is skipped, because RL rows don't need them. The run asserts that the new sidecar's
   marker token, its neighbours and the actor prompt equal the 3.6 NLA's.
3. **`prep_ckpts`.** Builds two warm-start files:
   - `ref_zero/`: the step-600 verbalizer adapter with `lora_B` set to 0, so the KL reference is `av_base`, as in the
     3.6 run.
   - `ar_lora/`: a fresh r64 LoRA plus the published value head, in the format EasyNLA loads. Its `lora_B` is 0, so
     step 0 is exactly the published reconstructor.
4. **`rl`.** `train_rl_self_contained`, resumed from verbalizer adapter `iter_000600` plus the published reconstructor.
   - The published reconstructor was saved at step 600 (`saved_at_step.txt`), so 600 is the matched pair. loan_rl
     used iter_800.
   - Hyperparameters are the 3.6 run's `run_config.yaml`:
     - lr 1e-4, critic lr 8e-5
     - group size 8, KL 0.01 (k3)
     - length penalty 0.01 per token past 192
     - LoRA r64 / α16, rsLoRA
     - 256 new tokens
   - Batch size differs: theirs was 504.

## vLLM path (`rl_vllm`, from 2026-11-02)

EasyNLA's `train_rl_vllm`, the backend the 3.6 run used: vLLM (with vllm-lens injecting the activation) generates the
rollouts, and the HF actor is re-merged and pushed into the engine after every step. One engine (TP 1) sits beside the
trainer and the AR on each GPU; more GPUs = data-parallel ranks under torchrun, as in the 3.6 run (6xB200). B200 only:
actor + AR + engine need ~178 GB.

- **Image:** vllm 0.19.0 + vllm-lens 1.1.0 (EasyNLA's pins), then transformers 5.5.4 for the HF Qwen3.6 actor and AR
  (vLLM 0.19 asks for <5 but ships its own qwen3_5 config). `apply_lens_patch.py` applies EasyNLA's vllm-lens patch
  hunks one by one: the EasyNLA patcher refuses on a fresh install because hunk 9's anchor is created by hunks 7-8.
- **`easynla_vllm_shim.py`** stands in for the 3.6 run's unpublished `vllm_name_map: qwen3_5_wrapper`:
  - The engine loads the original Qwen3.6-27B snapshot (multimodal wrapper; vLLM 0.19's text-only class can't read
    av_base's `model.language_model.*` names). The first sync pushes all 851 actor tensors, so the engine serves
    av_base + the resumed LoRA exactly; later syncs push the 496 LoRA-merged tensors.
  - Weight names: HF `model.layers.*` -> wrapper `model.language_model.layers.*`.
  - Syncs merge and push one layer at a time (EasyNLA builds a second full copy of the merged weights first: OOM).
  - Text only (`limit_mm_per_prompt` 0: the vision encoder's B200 flash-attn path fails to import) and Triton
    attention (FlashInfer JIT-compiles on B200 and there is no nvcc in the image).
  - AR loads through `easynla_shim.load_merged_critic`, as in the HF path; LoRA targets as there.
- **Settings that differ from the 3.6 run, for memory:** vllm gpu mem 0.38 (0.4 there), logp micro-batch 2 (8),
  critic micro-batch 2 (4).
- `VLLM_ALLOW_INSECURE_SERIALIZATION=1`: the sync and steer checks send pickled callables from the trainer to its own
  engine in the same container (EasyNLA's scripts set it too). Adam approved, 2026-11-02.

**Agreement with the HF path (vsmoke9, 16 prompts x 8):** step-0 train FVE 53.5% (HF 55.3%) and parse rate 18% (18%).
vLLM vs HF logprob of the sampled tokens: mean |diff| 0.020-0.027, 0-5 of 128 rollouts above EasyNLA's 0.1 mask
threshold. Injection verified on 128/128 rollouts every step.

## The thinking-prompt bug (found 2026-11-02)

EasyNLA renders the AV prompt with the chat template's default, which for Qwen3.x is thinking ON: the prompt ends in
an open `<think>`. The 3.6 NLA's card says to use `enable_thinking=False`. `parse_diag` (64 transfer-study vectors per
cell, 3.6 AV iter_000600): thinking on parses on 25-29% of outputs (greedy or temperature 1, 3.6 or 3.8 activations;
the failures are good explanations missing the opening `<explanation>` tag); thinking off parses on 100% in all four
cells. So 3.8 activations never broke the format, and every run before `fork_smoke_dp2` used the wrong prompt: their
step-0 FVE was scored on the parsed minority, and their first steps relearned the tag. Fixed in the fork
(`nla/utils/prompts.py`, also used by AV SFT).

## Known risks

- **Cold Adam.** The 3.6 run's optimizer state isn't published, so Adam restarts with no moment history. EasyNLA's
  code warns this can collapse late-stage policies (entropy death). Watch entropy and FVE in the first steps; if it
  collapses, lower the lr.
- **Our own name map.** The vLLM path relies on `easynla_vllm_shim.py` in place of the 3.6 run's lost code. Its checks:
  the vLLM-vs-HF logprob gap, the steer log, and step-0 FVE equal to the HF path's.
- **The reconstructor is also the ruler.** The headline metric uses the co-trained reconstructor. Also report FVE
  under the frozen starting reconstructor, so the verbalizer's own gain is visible (as on the 3.8 NLA card).

## Runs

| run | data | steps | result |
|---|---|---|---|
| smoke1 | 1k docs (5,120 RL rows) | 5 | OOM on 1×H200 at first generation (actor + AR + optimizer); wiring checks passed |
| smoke2 | same | 5 | 2×H200, device_map auto: rollouts and AR scoring ran, then OOM in the policy update (logp micro-batch 8) |
| smoke3 | same | 5 | logp micro-batch 2 + gradient checkpointing: completes, checkpoint `/data/runs/smoke3/iter_000005`. ~1,800 s/step for 512 rollouts, ~85% of it generation (HF `generate`, ~65 tok/s). Train FVE 55 → 58 → 66 → 65 → 68% over steps 0–4; held-out eval after one update 51% (59% parsed). Parse rate 18 → 65 → 99 → 100 → 100%. The step-0 failures are not truncation (0% cut off, mean 184 tokens): the `<explanation>` tags are missing; raw texts weren't logged. Entropy 1.67 → 1.52; KL to `av_base` 0.76 → 1.32 at step 4 |
| vsmoke1-8 | 1k docs | 3 | vLLM path bring-up on 1xB200: LLM override, vision encoder, first-sync OOM, serialization flag, FlashInfer JIT, update OOMs (each fixed above) |
| vsmoke9 | 1k docs | 3 | vLLM path, 1xB200, 16 prompts x 8: works. 63-68 s/step after step 0 (generation 15 s, policy update 36 s, AR 6-10 s, sync 0.5 s). Train FVE 53.5 -> 50.8 -> 62.6%, parse rate 18 -> 72 -> 99%; held-out eval after one update 56.8% (64% parsed) |
| vsmoke_dp2 | 1k docs | 2 | 2xB200 data-parallel, 32 prompts: runs, but parse rate 17 -> 14% after one update (vs 18 -> 72% on 1 GPU) and held-out eval 11% parsed: checked further |
| vsmoke_dp2d | 1k docs | 4 | 2xB200, 32 prompts, `--assert-param-sync` (trainable tensors only; EasyNLA's full-model check OOMs): all 992 trainable tensors identical across ranks every step. Parse rate 17-19 -> 34-44 -> 98-100 -> 96-100%, train FVE 41-54 -> 62-64% at step 3; held-out eval after one update 44% (39% parsed). ~62 s/step (16 prompts per GPU): generation 14 s, policy update 35 s, AR 9 s. The dp2 single-step dip was noise |
| parse_diag | transfer vectors | - | thinking on: 25-29% parsed; thinking off: 100% (see above) |
| fork_smoke_dp2 | 1k docs | 3 | First run from the fork (thinking off), 2xB200, 32 prompts: parse 100% every step; param spread 0; logprob gap 0.021-0.023, 0 masked; steer 128/128. Train FVE 63-67% at step 0, 59-64% at step 2 (KL 0.75 -> 0.79-0.84); held-out after one update 63.2% |
| q38_l42_rl100 | 20k docs | 100 | 8xB200 never scheduled (no capacity); not run |
| q38_l42_rl100_4x | 20k docs | 200 batches = 100 optimizer steps | 4xB200, 252 prompts x 8 x grad-accum 2 (= 504 per update, as the 3.6 run), eval every 20 batches (128 held-out prompts), save every 50, fork `ccb48d8` (launched on `4cc0d56`/`ac04489`). Held-out FVE by optimizer step: 0: 65.5, 10: 68.6, 20: 69.7, 30: 69.8, 40: 70.3, 50: 70.2, 60: 70.6, 70: 70.7, 80: 70.9, 90: 71.0%. Parse 99-100% throughout, KL 0.79-0.88. OOMed on rank 0 at batch 50 right after the first save (fragmentation, 180 MB short); fixed (empty_cache after eval/save, vLLM mem 0.38 -> 0.36) and resumed as `_r1` from iter_000050 (optimizer, data cursor, co-trained AR restored). ~354 s/batch before, ~420-490 s after (smaller KV cache). Cost ~$800 Modal |
| q38_l42_final_frozenAR | 128 held-out prompts | eval only | Final AV (iter_000200) under the published (starting) AR, fork `557f95f`: held-out FVE 66.6% (start AV: 65.5%). **Invalid**: with `--num-steps 1` and no grad-accum, eval@0 runs after one cold-Adam update of AV and AR on 8 prompts; redone with `--eval-only` in the fork's `experiments/qwen38_l42/swap_evals/` |
| q38_l42_startAV_finalAR | 128 held-out prompts | eval only | Starting AV (published iter_000600) under the run's final co-trained AR (critic_latest @ 200), frozen, fork `3dcd234`: held-out FVE 71.1% (final AV under it: ~71.0% at batch 180). **Invalid** for the same reason (one AV update before eval@0) |
