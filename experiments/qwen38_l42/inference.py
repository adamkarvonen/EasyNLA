"""Inference for the Qwen3.8-27B layer-42 NLA (README.md here; weights on the Hugging Face Hub).

text -> Qwen3.8 layer-42 activation -> AV explanation -> AR reconstruction -> FVE.

The three models load one at a time (each ~35-55 GB in bf16), so one 80 GB GPU is enough. Needs this repo installed
(`pip install -e .`, for the injection hook, prompt and AR code shared with training) and transformers >= 5.5.

    python experiments/qwen38_l42/inference.py --text "The Eiffel Tower is in" --n-samples 4
"""

import argparse
import gc

import torch
import torch.nn.functional as F
import transformers
from huggingface_hub import snapshot_download
from peft import PeftModel

from nla.config import load_nla_config
from nla.datagen.extractors import HFExtractor
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_RE, normalize_activation, resolve_target_scale
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook

SUBJECT = "Qwen/Qwen3.8-27B"
LAYER = 42
NLA_REPO = "adamkarvonen/qwen3.8-27b-nla-L42"  # AV adapter, AR, nla_meta.yaml
AV_BASE_REPO = "ceselder/qwen3.6-27b-nla-rl"  # av_base: Qwen3.6-27B with the warm-start AV LoRA merged in
# Predict-the-mean MSE of normalized layer-42 activations on the training rows (FVE = 1 - MSE / this).
FVE_BASELINE = 0.4441


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def extract(texts, layer=LAYER):
    """Layer-`layer` residual (output of decoder block `layer`) at every token of each text: list of [T, d] fp32."""
    ex = HFExtractor(SUBJECT, auto_class="AutoModelForImageTextToText", batch_size=4)
    out = [(r.hidden_states, r.token_ids) for r in ex.extract(texts, layer)]
    tok = ex.tokenizer
    del ex
    free_gpu()
    return out, tok


def verbalize(vectors, nla_dir, av_base_dir, n_samples=1, max_new_tokens=256, temperature=1.0):
    """Explanations for each activation: list (per vector) of n_samples (response, explanation-or-None)."""
    tok = transformers.AutoTokenizer.from_pretrained(f"{av_base_dir}/av_base")
    cfg = load_nla_config(nla_dir, tok)
    base = transformers.AutoModelForCausalLM.from_pretrained(f"{av_base_dir}/av_base", torch_dtype=torch.bfloat16,
                                                             device_map="cuda")
    av = PeftModel.from_pretrained(base, f"{nla_dir}/av_adapter").eval()
    vref = [None]
    register_karvonen_hook(av, vref, cfg.injection_token_id, cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    msgs = [{"role": "user", "content": cfg.actor_prompt_template.replace("{injection_char}", "<INJECT>")}]
    ids = tok.encode(build_prompt_text(msgs, cfg.injection_char, tok), add_special_tokens=False)
    pt = torch.tensor([ids] * n_samples, device="cuda")
    out = []
    for v in vectors:
        vref[0] = v.float().cuda().unsqueeze(0).expand(n_samples, -1)
        try:
            with torch.no_grad():
                seqs = av.generate(input_ids=pt, attention_mask=torch.ones_like(pt), max_new_tokens=max_new_tokens,
                                   do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                                   pad_token_id=tok.eos_token_id)
        finally:
            vref[0] = None
        resps = tok.batch_decode(seqs[:, pt.shape[1]:], skip_special_tokens=True)
        out.append([(r, (m.group(1).strip() if (m := EXPLANATION_RE.search(r)) else None)) for r in resps])
    del av, base
    free_gpu()
    return out, cfg, tok


def reconstruct(explanations, nla_dir, cfg, tok):
    """AR prediction (normalized, [d] fp32) for each explanation string."""
    scale = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    ar = NLACriticModel.from_pretrained(f"{nla_dir}/ar_reconstructor", torch_dtype=torch.bfloat16).cuda().eval()
    preds = []
    for e in explanations:
        x = torch.tensor([tok.encode(cfg.critic_prompt_template.format(explanation=e), add_special_tokens=False)],
                         device="cuda")
        with torch.no_grad():
            preds.append(normalize_activation(critic_predict(ar, x, None, scale), scale)[0].cpu())
    del ar
    free_gpu()
    return preds, scale


def fve(pred_n, gold, scale, baseline=FVE_BASELINE):
    gn = normalize_activation(gold.float().unsqueeze(0), scale)[0]
    return 1.0 - F.mse_loss(pred_n, gn).item() / baseline


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", required=True)
    p.add_argument("--position", type=int, default=-1, help="token index to explain (default: the last token)")
    p.add_argument("--n-samples", type=int, default=1, help="explanations sampled per activation (temperature 1)")
    p.add_argument("--nla-repo", default=NLA_REPO, help="Hub repo or local dir with av_adapter/, ar_reconstructor/")
    args = p.parse_args()

    nla_dir = snapshot_download(args.nla_repo)
    av_base_dir = snapshot_download(AV_BASE_REPO, allow_patterns=["av_base/*"])
    [(acts, token_ids)], subj_tok = extract([args.text])
    v = acts[args.position]
    print(f"activation at token {args.position} ({subj_tok.decode(token_ids[args.position])!r}), "
          f"norm {v.norm():.1f}")
    [samples], cfg, tok = verbalize([v], nla_dir, av_base_dir, n_samples=args.n_samples)
    expls = [e for _, e in samples if e is not None]
    preds, scale = reconstruct(expls, nla_dir, cfg, tok) if expls else ([], None)
    it = iter(preds)
    for i, (resp, e) in enumerate(samples):
        print(f"\n--- sample {i}")
        if e is None:
            print(f"(no <explanation> parsed)\n{resp}")
            continue
        print(e)
        print(f"FVE {100 * fve(next(it), v, scale):.1f}%")


if __name__ == "__main__":
    main()
