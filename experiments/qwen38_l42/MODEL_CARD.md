---
license: other
base_model: ceselder/qwen3.6-27b-nla-rl
library_name: peft
tags:
- interpretability
- natural-language-autoencoder
- nla
- qwen3.8
pipeline_tag: text-generation
---

# Qwen3.8-27B layer-42 Natural Language Autoencoder

A [Natural Language Autoencoder](https://transformer-circuits.pub/2026/nla/) (NLA) for **Qwen3.8-27B**
**layer-42** residual-stream activations. It was built by continuing the RL of Celeste's Qwen3.6-27B NLA
([ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl)) on Qwen3.8 activations for 100
optimizer steps.

- **Verbalizer (AV)**: reads an activation and writes an `<explanation>`.
- **Reconstructor (AR)**: maps the explanation back to an activation.

**Held-out FVE: 69.6%**, against 65.1% for the original 3.6 NLA on the same Qwen3.8 activations (512 held-out
FineFineWeb prompts, one sample each at temperature 1).

> **Read this first: the gain is the reconstructor's.** Scoring each verbalizer under each reconstructor shows that
> the fine-tuned verbalizer's explanations are no better than the original's (65.2 vs 65.1% under the original AR,
> 69.6 vs 69.0% under this AR). The fine-tuned AR is what gains, because it maps explanations into Qwen3.8's
> activation space rather than Qwen3.6's. Use this AR whenever you report FVE on Qwen3.8. For reading explanations,
> the original 3.6 verbalizer is as good. Full write-up:
> [FINDINGS.md](https://github.com/adamkarvonen/EasyNLA/blob/main/experiments/qwen38_l42/FINDINGS.md).

## Contents

| path | what |
|---|---|
| `av_adapter/` | AV LoRA (r64, alpha 16, rsLoRA, all modules), RL step 100. It attaches to **`av_base/` from [ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl)** (Qwen3.6-27B with the warm-start AV LoRA merged), not to Qwen3.8 and not to plain Qwen3.6. |
| `ar_reconstructor/` | AR: the first 43 blocks of the model with the LoRA merged in (`model.safetensors`), plus `value_head.safetensors`. Co-trained, RL step 100. |
| `nla_meta.yaml` | Extraction config (Qwen/Qwen3.8-27B, layer 42), injection token ids, and the AV and AR prompt templates. They are identical to the 3.6 NLA's. |
| `inference.py` | Inference: text -> activation -> explanation -> reconstruction -> FVE. |

## How to use

The verbalizer runs on **Qwen3.6** weights (`av_base` + this adapter) and reads activations of **Qwen3.8**.

- The activation is the output of decoder block 42 of Qwen/Qwen3.8-27B (HF `hidden_states[43]`).
- It is injected at the marker token `㈜` (id 158983) after layer 1, added with its norm matched to the residual (as in
  Karvonen et al.).
- Use the chat template with `enable_thinking=False`. With thinking on, most outputs lose the opening
  `<explanation>` tag.
- Sample at temperature 1.

The injection hook, prompt construction and AR head are the training code's own, so the easiest way to run the NLA is
from the repo that trained it:

```bash
git clone https://github.com/adamkarvonen/EasyNLA && cd EasyNLA
pip install -e . "transformers>=5.5" peft flash-linear-attention causal-conv1d   # fast DeltaNet kernels
python experiments/qwen38_l42/inference.py --text "The Eiffel Tower is located in the city of" --n-samples 4
```

`inference.py` loads Qwen3.8 (activation), the AV and the AR one at a time (~55, 55 and 35 GB in bf16), so a single
80 GB GPU is enough. From Python:

```python
import sys; sys.path.insert(0, "experiments/qwen38_l42")
import inference as nla
from huggingface_hub import snapshot_download

nla_dir = snapshot_download("adamkarvonen/qwen3.8-27b-nla-L42")
av_base_dir = snapshot_download("ceselder/qwen3.6-27b-nla-rl", allow_patterns=["av_base/*"])

[(acts, token_ids)], _ = nla.extract(["Some text whose last token you want explained"])   # acts: [T, 5120]
vectors = [acts[-1]]                                                    # any [5120] layer-42 activations
samples, cfg, tok = nla.verbalize(vectors, nla_dir, av_base_dir, n_samples=4)
expls = [e for _, e in samples[0] if e is not None]                     # (response, explanation) pairs
preds, scale = nla.reconstruct(expls, nla_dir, cfg, tok)
print([round(nla.fve(p, vectors[0], scale), 3) for p in preds])         # per-sample FVE
```

Per-sample FVE divides by the predict-the-mean MSE of the training activations (0.4441, in `inference.py`). Pass
`baseline=` to use your own eval set's variance, as the numbers below do.

**Check** (`modal_app.py::release_check`, which runs this code on the trainer's held-out eval rows):

- Activations re-extracted from the source text equal the stored training activations (cosine >= 0.9999 on 16
  rows).
- The AV prompt equals the training rows' prompt.
- 64/64 explanations parse.
- Held-out FVE is 70.7% on the first 64 eval prompts (that set's own baseline). The 512-prompt number below is 69.6%,
  and 64 prompts is noisy.

Without the `flash-linear-attention` and `causal-conv1d` kernels, transformers falls back to a slow torch DeltaNet
path. It works, but generation is slow.

## Results

Held-out FVE (%) on 512 prompts from held-out documents. The baseline is the eval set's own predict-the-mean MSE
(0.4414). Every verbalizer is scored under every reconstructor (eval only, no updates):

| AV \ AR | original 3.6 AR | **this AR** (co-trained) | AR trained alone (20 steps) |
|---|---|---|---|
| original 3.6 AV (`iter_000600`) | 65.1 | 69.0 | 68.1 |
| **this AV** (co-trained) | 65.2 | **69.6** | 68.2 |
| AV trained alone, AR frozen (20 steps) | 65.1 | 68.6 | 67.3 |

Differences under ~0.6 points are within the noise. In-run curve of the co-trained pair (128 held-out prompts) by
optimizer step: 0: 65.5, 10: 68.6, 20: 69.7, 40: 70.3, 60: 70.6, 80: 70.9, 90: 71.0%. Parse rate was 99-100% and KL
to the start 0.79-0.88 throughout.

FVE isn't comparable across models or eval sets. The 3.6 NLA's 78% is on Qwen3.6 activations of its own eval set.

## Training

Code: [adamkarvonen/EasyNLA](https://github.com/adamkarvonen/EasyNLA), a fork of
[ceselder/EasyNLA](https://github.com/ceselder/EasyNLA). Experiment folder:
[`experiments/qwen38_l42/`](https://github.com/adamkarvonen/EasyNLA/tree/main/experiments/qwen38_l42). Its README
covers all arms and `cotrain/run.sh` is the exact launch command. The run is `nla.train_rl_vllm` on Modal
(`modal_app.py`).

- **Start**: AV adapter `av_rl_adapters/iter_000600` and `ar_reconstructor` from ceselder/qwen3.6-27b-nla-rl. The
  published AR was saved at step 600, so this is the matched pair.
- **Data**:
  - 20k FineFineWeb documents, an equal share from each of its 67 domains (the 3.6 NLA's corpus).
  - Qwen3.8-27B layer-42 activations at 10 positions per document, about 100k RL rows.
  - 5% of documents are held out (crc32 doc-hash split); the eval set is their first N rows.
- **Algorithm**:
  - On-policy GRPO, group size 8, temperature 1, k3 KL to `av_base` (beta 0.01).
  - Length penalty of 0.01 per token past 192; 256 new tokens.
  - AV lr 1e-4, AR lr 8e-5, LoRA r64 / alpha 16 on both; AR co-trained.
  - These are the 3.6 run's hyperparameters.
- **Batch**: 504 prompts x 8 samples per optimizer step (252 per rollout batch, grad-accum 2), as the 3.6 run;
  100 optimizer steps.
- **Hardware**: 4x B200, one vLLM 0.19 engine (TP 1) per GPU beside the trainer, data-parallel under torchrun. The
  engine serves the original Qwen3.6-27B, and weight syncs push the AV.

### Modal cost

| item | cost |
|---|---|
| this NLA (co-trained run, 100 steps, 4x B200, ~25 h incl. one OOM restart) | $730 |
| ablations: AV-only (20 steps) $119, AR-only (20 steps) $59, swap evals $13 | $191 |
| data extraction (Qwen3.8 activations, 1x H200) | $6 |
| bring-up and smoke tests | $56 |
| **total** | **$984** |

4x B200 ran ~$29/h on Modal.

## Caveats

- **One layer (42) and one domain.** Tested only on FineFineWeb web text.
- **FVE is scored by the AR.** The AR is what improved here, so higher FVE does not mean better explanations (see the
  table).
- **Cold optimizer.** The 3.6 run's Adam state isn't published, so RL restarted with fresh moments. Entropy and parse
  rate stayed stable.
- **License.** Built on Qwen3.6/Qwen3.8 and on the 3.6 NLA's `av_base`; their terms apply.
