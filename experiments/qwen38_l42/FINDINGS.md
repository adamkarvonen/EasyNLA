# Fine-tuning the Qwen3.6-27B NLA for Qwen3.8-27B barely matters

*2026-11-04. Layer 42, FineFineWeb text. Numbers and run details are in README.md here.*

**Question.** The Qwen3.6-27B NLA ([ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl))
already transfers to Qwen3.8-27B activations. Does continuing its RL on Qwen3.8 activations give a better NLA?

**Answer: not much, and none of the gain is in the explanations.**

- **Transfer without fine-tuning.** Out of the box, the 3.6 NLA scores 65% held-out FVE on Qwen3.8 web activations.
  - That is 13 points below its 78% on Qwen3.6.
  - It is about equal to the natively trained Qwen3.8 NLA (`gereon/qwen3.8-27b-nla-L42`, 65%; the earlier transfer
    study in motivated-reasoning `loan_rl/nla_transfer/`).
- **100 RL steps of EasyNLA co-training** (504 prompts per step, the 3.6 run's recipe, ~$730 on 4xB200) raise this to
  69.6%. Most of the gain comes in the first 20 steps.
- **The gain is the reconstructor's (AR's), not the verbalizer's (AV's).** Each final AV was scored under each AR on
  the same 512 held-out prompts:

  | AV \ AR | original | co-trained | trained alone |
  |---|---|---|---|
  | original | 65.1 | 69.0 | 68.1 |
  | co-trained | 65.2 | 69.6 | 68.2 |
  | trained alone (AR frozen) | 65.1 | 68.6 | 67.3 |

  - Swapping in a trained AR adds 3-4.5 points under any AV.
  - Swapping in a trained AV adds 0-0.6 points under any AR. That is within the noise (about +-0.6).
  - Training the AV alone against the original AR is flat for 20 steps.
- **Why the AR gains.** The earlier transfer study attributed about 6 of the 13 lost points to the 3.6 AR mapping
  explanations into 3.6's activation space. AR training recovers most of those. The other 7-8 points are explanation
  content lost on 3.8 activations, and AV RL does not recover any of them.

**What this means for using it.**

- The explanations of the fine-tuned NLA are no better than the original's. For reading explanations of Qwen3.8
  activations, the original 3.6 NLA is as good.
- The fine-tuned AR is worth using whenever an FVE or reconstruction is reported, because it scores 3.8 activations
  ~4 points higher for the same explanations.
- Released pair: [adamkarvonen/qwen3.8-27b-nla-L42](https://huggingface.co/adamkarvonen/qwen3.8-27b-nla-L42) (the
  co-trained AV and AR).

**Limits.**

- One layer (42) and one text domain (FineFineWeb web text). The transfer study's loan transcripts were not re-run with
  the fine-tuned AR.
- Evals use one sample per prompt at temperature 1.
- The AV-only arm ran 20 optimizer steps, not 100. Co-training's AV also didn't move in 100 steps, so a longer AV-only
  run is unlikely to differ.
- FVE is scored by the AR itself. A better AR raises FVE without the explanations saying anything new; that is the
  pattern here.
