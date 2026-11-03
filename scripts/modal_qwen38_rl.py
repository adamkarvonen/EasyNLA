"""Continue the RL of the Qwen3.6-27B NLA (ceselder/qwen3.6-27b-nla-rl, layer 42) on Qwen3.8-27B activations, on Modal.

The image installs THIS checkout of the repo (no git clone), so a run uses exactly the committed code: commit, then
launch. Stages:

1. download: the NLA repo files + a 20k-doc FineFineWeb sample (equal share from each of its 67 domains).
2. extract: stage0 (Qwen3.8 layer-42 activations, 10 positions/doc) -> stage1 split -> stage3 RL rows with the 3.6
   NLA's own prompt templates -> shuffle. No stage2: RL rows need no explanations.
3. rl: nla.train_rl_vllm resumed from the published AV adapter iter_000600 + the published AR (saved at RL step 600,
   so the matched pair), with the 3.6 run's hyperparameters. One vLLM engine (TP 1) beside the trainer on each GPU;
   N GPUs = N data-parallel ranks under torchrun. B200 only (actor + AR + engine ~178 GB per GPU).

    set -a && . .env && set +a          # HF_TOKEN, WANDB_API_KEY
    modal run scripts/modal_qwen38_rl.py::download
    modal run scripts/modal_qwen38_rl.py::extract
    NLA_GPUS=B200:8 modal run scripts/modal_qwen38_rl.py::rl --name q38_l42_rl --n-gpus 8 --num-steps 100 \
        --batch-prompts 504 --eval-every 10 --eval-n-prompts 128 --save-every 25

Resume an interrupted run from its last checkpoint (optimizer state and data cursor continue; the AR restarts from
the run's critic_latest with a fresh LoRA, as EasyNLA's --ar-lora resume does):

    NLA_GPUS=B200:4 modal run scripts/modal_qwen38_rl.py::rl --name <wandb name> --resume <run>/iter_000050 ...

Modal names (app, volumes) carry the adam- prefix of the workspace this was run in; change APP/VOLUME names to reuse.
"""

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
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


# memory: (request, limit) in MiB. Each DP rank stages actor + AR through CPU RAM at load, so 8 ranks can burst past
# the 512 GiB request.
@app.function(gpu=os.environ.get("NLA_GPUS", "B200"), volumes=VOLS, timeout=24 * 3600, memory=(524288, 1048576),
              secrets=[modal.Secret.from_local_environ(["WANDB_API_KEY", "HF_TOKEN"])])
def rl(name: str, num_steps: int = 5, batch_prompts: int = 16, lr: float = 1e-4, eval_every: int = 5,
       eval_n_prompts: int = 64, rows: str = "/data/q38_L42_20k/rl_shuf.parquet", val_rows: int = 5000,
       n_gpus: int = 1, save_every: int = 0, vllm_gpu_mem: float = 0.36, critic_micro_batch: int = 2,
       wandb_project: str = "adam-nla-finetune", resume: str = "", init_lora: str = "", train_critic: bool = True,
       extra: str = ""):
    """nla.train_rl_vllm resumed from the 3.6 NLA. The 3.6 run's run_config.yaml hyperparameters, except, for memory on
    B200: vllm gpu mem 0.36 (0.4), logp micro-batch 2 (8), critic micro-batch 2 (4). NLA_GPUS (read when the app is
    built locally, e.g. B200:8) must match n_gpus."""
    import shlex
    import sys

    from huggingface_hub import snapshot_download

    d = _nla_dir()
    vllm_model = snapshot_download(VLLM_MODEL)
    # Weight sync and the steer checks ship pickled callables from the trainer to its own engine in the same container
    # (EasyNLA's scripts set this too).
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    launch = ([sys.executable, "-m", "nla.train_rl_vllm"] if n_gpus == 1 else
              [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n_gpus}",
               "-m", "nla.train_rl_vllm"])
    # resume: continue an interrupted run from its own checkpoint (iter_* in its save dir: LoRA, optim_latest with the
    # step, critic_latest); otherwise start from the published AV adapter.
    resume_lora = f"/data/runs/{resume}" if resume else f"{d}/{NLA_ADAPTER}"
    if init_lora:
        # Start a FRESH run (step 0, published AR) from another run's adapter, e.g. to score a trained AV under the
        # frozen starting AR (--train-critic off, --num-steps 1, eval at step 0). Copied out of its run dir: the
        # trainer looks for optim_latest beside the LoRA and would otherwise resume that run's step.
        import shutil
        resume_lora = "/tmp/init_lora"
        shutil.copytree(f"/data/runs/{init_lora}", resume_lora)
    save_dir = f"/data/runs/{resume.split('/')[0]}" if resume else f"/data/runs/{name}"
    _run([*launch,
          "--av-ckpt", f"{d}/av_base", "--resume-from-lora", resume_lora,
          "--ar-ckpt", f"{d}/ar_reconstructor", "--ar-lora", "--ar-lora-r", "64", "--ar-lora-alpha", "16",
          *(["--train-critic"] if train_critic else []), "--rl-parquet", rows, "--sidecar", rows, "--save-dir", save_dir,
          "--num-steps", str(num_steps), "--batch-prompts", str(batch_prompts), "--group-size", "8",
          "--max-new-tokens", "256", "--temperature", "1.0", "--lr", str(lr), "--critic-lr", "8e-5",
          "--lora-r", "64", "--lora-alpha", "16", "--kl-beta", "0.01", "--kl-estimator", "k3",
          "--length-penalty", "0.01", "--length-threshold", "192",
          "--logp-micro-batch", "2", "--gradient-checkpointing", "--critic-micro-batch", str(critic_micro_batch),
          "--vllm-model", vllm_model, "--vllm-name-map", "qwen3_5_wrapper", "--vllm-attn-backend", "TRITON_ATTN",
          "--vllm-gpu-mem", str(vllm_gpu_mem), "--vllm-max-len", "1024", "--vllm-tp", "1", "--ipc-weight-sync",
          "--eval-every", str(eval_every), "--eval-n-prompts", str(eval_n_prompts), "--val-rows", str(val_rows),
          "--save-every", str(save_every or max(num_steps, 1)), "--wandb-project", wandb_project, "--wandb-name", name,
          *shlex.split(extra)])
    data.commit()
