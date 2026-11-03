"""Modal app for the Qwen3.8-27B layer-42 NLA experiments (README.md here): continue the RL of the Qwen3.6-27B NLA
(ceselder/qwen3.6-27b-nla-rl) on Qwen3.8 activations.

The image installs THIS checkout of the repo, so a run uses exactly the committed code: commit, then launch.

- download, extract: the NLA repo files and 20k FineFineWeb docs -> Qwen3.8 layer-42 RL rows (once).
- check_ar: CPU check that the published AR loads exactly.
- rl: nla.train_rl_vllm from the published AV adapter iter_000600 and AR (saved at RL step 600, the matched pair),
  with the 3.6 run's hyperparameters; one vLLM engine (TP 1) beside the trainer on each GPU, N GPUs = N data-parallel
  ranks under torchrun. B200 only (actor + AR + engine ~178 GB per GPU). The arms in the subfolders are its flags.
"""

import os
from pathlib import Path

import modal

# The repo root, two levels up locally; in the container this file sits alone in /root (and the repo is /opt/EasyNLA).
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/EasyNLA")
NLA_REPO = "ceselder/qwen3.6-27b-nla-rl"
# The published AR was saved at RL step 600 (ar_reconstructor/saved_at_step.txt): resume the AV from the same step.
NLA_ADAPTER = "av_rl_adapters/iter_000600"
VLLM_MODEL = "Qwen/Qwen3.6-27B"  # what the engine loads before the first (full) weight sync
SUBJECT = "Qwen/Qwen3.8-27B"
LAYER = 42

# vllm 0.19.0 + vllm-lens 1.1.0: EasyNLA's pinned pair (scripts/install_vllm_lens.sh), then the repo's vllm-lens patch.
# vLLM 0.19 asks for transformers<5 but ships its own qwen3_5 config; the HF actor and AR (Qwen3_5ForCausalLM) need
# transformers 5, so transformers is upgraded after vLLM.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install("vllm==0.19.0", "vllm-lens==1.1.0", "peft==0.19.1", "bitsandbytes", "wandb", "accelerate",
                    "safetensors", "huggingface_hub[hf_xet]", "numpy", "pyyaml", "pyarrow", "datasets", "httpx",
                    "orjson", "anthropic", "tqdm", "flash-linear-attention==0.5.2")
    .run_commands("pip install transformers==5.5.4")
    .add_local_dir(REPO, "/opt/EasyNLA", copy=True,
                   ignore=[".git", "**/__pycache__", "wandb", "data", "ckpts", "*.parquet"])
    .run_commands("cd /opt/EasyNLA && pip install --no-deps -e . && python utils/patch_vllm_lens.py")
    .env({"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1"})
)
hf_cache = modal.Volume.from_name("adam-mr-hf-cache")
data = modal.Volume.from_name("adam-nla-finetune", create_if_missing=True)
app = modal.App("adam-nla-finetune", image=image)
VOLS = {"/hf": hf_cache, "/data": data}


def _nla_dir() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(NLA_REPO, allow_patterns=["av_base/*", f"{NLA_ADAPTER}/*", "ar_reconstructor/*",
                                                       "nla_meta.yaml", "run_config.yaml"])


def _run(cmd: list[str]) -> None:
    import subprocess
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(volumes=VOLS, timeout=3600, memory=32768, secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])])
def download(n_docs: int = 20000):
    """Online: the NLA repo files, the vLLM engine's base snapshot, and n_docs FineFineWeb docs (the 3.6 NLA's corpus).
    FineFineWeb is stored by domain, so the head of the stream is all one domain (aerospace): take an equal share from
    the start of each domain's first file instead."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, snapshot_download

    print(_nla_dir(), snapshot_download(VLLM_MODEL))
    out = f"/data/corpus/finefineweb_{n_docs // 1000}k_domains.parquet"
    if not os.path.exists(out):
        os.makedirs("/data/corpus", exist_ok=True)
        files = sorted(f for f in HfApi().list_repo_files("m-a-p/FineFineWeb", repo_type="dataset")
                       if f.endswith("_000000.jsonl"))
        per = -(-n_docs // len(files))
        rows = []
        for f in files:
            ds = load_dataset("m-a-p/FineFineWeb", data_files=f, split="train", streaming=True)
            for i, ex in enumerate(ds):
                if i == per:
                    break
                rows.append({"text": ex["text"], "domain": f.split("/")[0]})
        rows = rows[:n_docs]
        print(f"{len(files)} domains x {per} docs -> {len(rows)}")
        Dataset.from_list(rows).shuffle(seed=0).to_parquet(out)
    print(out, os.path.getsize(out) / 1e6, "MB")
    hf_cache.commit()
    data.commit()


@app.function(gpu="H200", volumes=VOLS, timeout=4 * 3600, memory=262144)
def extract(corpus: str = "/data/corpus/finefineweb_20k_domains.parquet", n_docs: int = 20000,
            out_dir: str = "/data/q38_L42_20k"):
    """EasyNLA datagen for RL rows on the subject model; asserts the marker tokens and actor prompt equal the 3.6 NLA's."""
    import sys

    import yaml

    meta36 = yaml.safe_load(open(f"{_nla_dir()}/nla_meta.yaml"))
    py = sys.executable
    storage = ["--storage-cls", "nla.datagen.storage.LocalStorage"]
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(f"{out_dir}/base.parquet.nla_meta.yaml"):
        _run([py, "-m", "nla.datagen.stage0_extract", "--base-model", SUBJECT, "--corpus", corpus,
              "--corpus-start", "0", "--corpus-length", str(n_docs), "--layer-index", str(LAYER),
              "--positions-per-doc", "10", "--chunk-size", "256", "--seed", "42", "--chunks-per-file", "4",
              "--extractor-kwargs",
              '{"batch_size": 16, "max_length": 2048, "auto_class": "AutoModelForImageTextToText"}',
              "--output", f"{out_dir}/base.parquet", *storage])
        data.commit()
    _run([py, "-m", "nla.datagen.stage1_split", "--base", f"{out_dir}/base.parquet", "--av-sft-frac", "0.25",
          "--ar-sft-frac", "0.25", "--rl-frac", "0.5", "--seed", "42", "--output-dir", f"{out_dir}/splits", *storage])
    _run([py, "-m", "nla.datagen.stage3_build", "--input", f"{out_dir}/splits/rl_raw.parquet", "--stage", "rl",
          "--output", f"{out_dir}/rl.parquet", "--actor-template", meta36["prompt_templates"]["actor"],
          "--critic-template", meta36["prompt_templates"]["critic"], "--keep-debug-metadata", *storage])
    _run([py, "-m", "nla.datagen.stage_shuffle", "--input", f"{out_dir}/rl.parquet",
          "--output", f"{out_dir}/rl_shuf.parquet", "--seed", "42", *storage])
    meta = yaml.safe_load(open(f"{out_dir}/rl_shuf.parquet.nla_meta.yaml"))
    for k in ("injection_char", "injection_token_id", "injection_left_neighbor_id", "injection_right_neighbor_id"):
        assert meta["tokens"][k] == meta36["tokens"][k], (k, meta["tokens"][k], meta36["tokens"][k])
    assert meta["prompt_templates"]["actor"] == meta36["prompt_templates"]["actor"]
    print(yaml.safe_dump({k: v for k, v in meta.items() if k != "prompt_templates"}))
    data.commit()


@app.function(volumes=VOLS, timeout=3600, memory=196608, cpu=8)
def check_ar():
    """CPU: NLACriticModel.from_pretrained on the published AR (a merged 43-block Qwen3_5ForCausalLM saved under
    model.language_model.* names) loads every backbone tensor of the checkpoint, and its value head."""
    import torch
    from safetensors.torch import load_file

    from nla.models import NLACriticModel

    path = f"{_nla_dir()}/ar_reconstructor"
    critic = NLACriticModel.from_pretrained(path, torch_dtype=torch.bfloat16)
    ours = critic.state_dict()
    raw = load_file(f"{path}/model.safetensors")
    n = 0
    for k, v in raw.items():
        k2 = "backbone." + k.replace("model.language_model.", "model.")
        assert k2 in ours, (k, k2, [x for x in ours if x.startswith("backbone.model.layers.0.")][:3])
        assert torch.equal(ours[k2].to(v.dtype), v), k
        n += 1
    head = load_file(f"{path}/value_head.safetensors")
    for k, v in head.items():
        assert torch.equal(ours[f"value_head.{k}"].to(v.dtype), v), k
    print(f"AR ok: {n} backbone tensors and {len(head)} value-head tensors equal the checkpoint; "
          f"{len(ours) - n - len(head)} other entries: {[k for k in ours if not k.startswith(('backbone.model.layers', 'value_head'))]}")


# memory: (request, limit) in MiB. Each DP rank stages actor + AR through CPU RAM at load, a burst past the request.
@app.function(gpu=os.environ.get("NLA_GPUS", "B200"), volumes=VOLS, timeout=24 * 3600, memory=(262144, 1048576),
              secrets=[modal.Secret.from_local_environ(["WANDB_API_KEY", "HF_TOKEN"])])
def rl(name: str, n_gpus: int = 1, num_steps: int = 5, batch_prompts: int = 16, grad_accum: int = 1, lr: float = 1e-4,
       eval_every: int = 5, eval_n_prompts: int = 64, save_every: int = 0, av_lora: str = "", ar: str = "",
       train_actor: bool = True, train_critic: bool = True, eval_only: bool = False, resume: str = "",
       rows: str = "/data/q38_L42_20k/rl_shuf.parquet", val_rows: int = 5000, vllm_gpu_mem: float = 0.36,
       critic_micro_batch: int = 2, wandb_project: str = "adam-nla-finetune", extra: str = ""):
    """One run of nla.train_rl_vllm. The 3.6 run's run_config.yaml hyperparameters, except, for memory on B200: vllm
    gpu mem 0.36 (0.4), logp micro-batch 2 (8), critic micro-batch 2 (4). NLA_GPUS (read when the app is built
    locally, e.g. B200:4) must match n_gpus. Paths are relative to /data/runs on the adam-nla-finetune volume:

    - av_lora: start the AV from this adapter (e.g. q38_l42_rl100_4x/iter_000200) instead of the published one.
    - ar: start (or score with) this AR (e.g. q38_l42_rl100_4x/critic_latest) instead of the published one.
    - train_actor / train_critic: --no-train-actor fixes the AV, --no-train-critic fixes the AR.
    - eval_only: score the AV under the AR on the eval set once (eval_n_prompts held-out prompts), no update.
    - resume: continue an interrupted run from its own iter_* (optimizer state, data cursor and critic_latest).
    """
    import shlex
    import shutil
    import sys

    from huggingface_hub import snapshot_download

    assert not (resume and (av_lora or ar)), "resume restores the run's own AV and AR"
    d = _nla_dir()
    vllm_model = snapshot_download(VLLM_MODEL)
    # Weight sync and the steer checks ship pickled callables from the trainer to its own engine in the same container
    # (EasyNLA's scripts set this too).
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    launch = ([sys.executable, "-m", "nla.train_rl_vllm"] if n_gpus == 1 else
              [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpus}",
               "-m", "nla.train_rl_vllm"])
    save_dir = f"/data/runs/{resume.split('/')[0]}" if resume else f"/data/runs/{name}"
    if resume:
        lora = f"/data/runs/{resume}"
    elif av_lora:
        # Copied out of its run dir: the trainer looks for optim_latest beside the LoRA and would resume that run's step.
        lora = "/tmp/av_lora"
        shutil.copytree(f"/data/runs/{av_lora}", lora)
    else:
        lora = f"{d}/{NLA_ADAPTER}"
    _run([*launch,
          "--av-ckpt", f"{d}/av_base", "--resume-from-lora", lora,
          "--ar-ckpt", f"/data/runs/{ar}" if ar else f"{d}/ar_reconstructor",
          "--ar-lora", "--ar-lora-r", "64", "--ar-lora-alpha", "16",
          "--train-actor" if train_actor else "--no-train-actor",
          "--train-critic" if train_critic else "--no-train-critic",
          *(["--eval-only"] if eval_only else []),
          "--rl-parquet", rows, "--sidecar", rows, "--save-dir", save_dir,
          "--num-steps", str(num_steps), "--batch-prompts", str(batch_prompts), "--grad-accum", str(grad_accum),
          "--group-size", "8", "--max-new-tokens", "256", "--temperature", "1.0", "--lr", str(lr), "--critic-lr", "8e-5",
          "--lora-r", "64", "--lora-alpha", "16", "--kl-beta", "0.01", "--kl-estimator", "k3",
          "--length-penalty", "0.01", "--length-threshold", "192",
          "--logp-micro-batch", "2", "--gradient-checkpointing", "--critic-micro-batch", str(critic_micro_batch),
          "--vllm-model", vllm_model, "--vllm-name-map", "qwen3_5_wrapper", "--vllm-attn-backend", "TRITON_ATTN",
          "--vllm-gpu-mem", str(vllm_gpu_mem), "--vllm-max-len", "1024", "--vllm-tp", "1", "--ipc-weight-sync",
          "--eval-every", str(eval_every), "--eval-n-prompts", str(eval_n_prompts), "--val-rows", str(val_rows),
          "--save-every", str(save_every or max(num_steps, 1)), "--wandb-project", wandb_project, "--wandb-name", name,
          *shlex.split(extra)])
    data.commit()
