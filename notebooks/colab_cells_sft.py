"""Colab cell sequence for Protocol One — Rejection-Sampling SFT (vanilla path).

This is the v2 rewrite after Unsloth+bitsandbytes proved unworkable on the
current Colab T4 image. Three separate bnb versions hit three independent
failures in succession (CUDA-13 lib missing, GLIBCXX too old, triton.ops
removed). The reliable path on T4 is plain transformers + peft + TRL in
fp16 — slower than 4-bit Unsloth but actually runs.

Numbers for Qwen 2.5-1.5B-Instruct on T4 (15 GB):
    Model fp16          ~ 3.0 GB
    LoRA adapters (r=16) ~30 MB
    Optimizer state     ~150 MB
    Activations + grads ~5-8 GB  (with gradient checkpointing)
    Headroom            ~3-5 GB
Total fits comfortably. ~2-3x slower than Unsloth 4-bit but bulletproof.

For larger models (3B / 7B) on HF compute (A100/H100), the same script
runs unchanged — just flip `MODEL_SIZE` in Cell 5. fp16 of a 7B model is
~14 GB which fits A100 40GB easily.

Pipeline recap:
    build_sft_dataset.py     transcripts -> belief graphs (matcher-filtered JSONL)
    SFTTrainer (vanilla TRL) supervised fine-tune Qwen on (transcript, belief)
    RFTEvalCallback          live env eval every N steps -> the climbing curve
    plotting.py              dark-themed plots into notebooks/figures/

Cells are Jupytext-style. Copy each block between `# %%` markers into one
Colab cell, in order. 17 cells total.
"""

# %% [Cell 1] -- Verify GPU
!nvidia-smi


# %% [Cell 2] -- Minimal install (NO Unsloth, NO bitsandbytes, NO torchcodec)
# These four packages are the entire training stack for the vanilla path.
# Pinning trl==0.22.2 + transformers==4.56.2 because they were validated
# against this codebase. Other versions may also work; don't drift unless
# you have a reason.
!pip install --upgrade -qqq pip
!pip install -q transformers==4.56.2 peft==0.13.2 "accelerate>=1.4.0" datasets==3.0.1 matplotlib
!pip install -q --no-deps trl==0.22.2
# accelerate must be >= 1.4 because transformers 4.56.2 calls
# Accelerator.unwrap_model(keep_torch_compile=False), and that kwarg was
# added in accelerate 1.4 (NOT 1.1 as I originally pinned).
print("OK vanilla install set installed (transformers + peft + trl + accelerate)")


# %% [Cell 3] -- Verify versions
import importlib.metadata as md
for pkg, want in [("transformers", "4.56.2"), ("trl", "0.22.2"),
                  ("peft", "0.13.2"), ("accelerate", ">=1.4.0"),
                  ("datasets", "3.0.1"), ("matplotlib", "any")]:
    try:
        got = md.version(pkg)
    except Exception:
        got = "MISSING"
    print(f"{pkg:14} = {got:12} (want {want})")


# %% [Cell 4] -- Clone repo + install editable
%cd /content
!rm -rf /content/repo
!git clone https://github.com/suhaniawasthi10/Protocol-RE.git /content/repo
%cd /content/repo
!pip install -q -e .
print("OK repo cloned + installed editable")


# %% [Cell 5] -- Build the SFT dataset (~3-5 min). Skip if data/sft.jsonl is
# already in the repo (it is, if you pushed it from your laptop).
import os
if os.path.exists("/content/repo/data/sft.jsonl"):
    n = sum(1 for _ in open("/content/repo/data/sft.jsonl"))
    print(f"OK using existing data/sft.jsonl ({n} examples) -- skipping build")
else:
    !python -m scripts.build_sft_dataset \
        --episodes 1500 --threshold 0.40 --mutation-prob 0.25 \
        --max-probes 12 --out data/sft.jsonl


# %% [Cell 6] -- Imports + hardware-aware config
import os, sys, json, torch
sys.path.insert(0, "/content/repo")

# === Hardware preset =========================================================
# T4 free  : MODEL_SIZE = "1.5B"   ~3 GB fp16, fits 15 GB easily
# L4/A100  : MODEL_SIZE = "3B"     ~6 GB fp16, fits 24 GB
# A100/H100: MODEL_SIZE = "7B"    ~14 GB fp16, needs 40 GB+
MODEL_SIZE = "1.5B"
# =============================================================================

_PRESETS = {
    "1.5B": dict(model="Qwen/Qwen2.5-1.5B-Instruct", per_device_bs=2,
                 grad_accum=4, max_seq_len=4096, lr=2e-4),
    "3B":   dict(model="Qwen/Qwen2.5-3B-Instruct",   per_device_bs=2,
                 grad_accum=4, max_seq_len=4096, lr=1.5e-4),
    "7B":   dict(model="Qwen/Qwen2.5-7B-Instruct",   per_device_bs=1,
                 grad_accum=8, max_seq_len=4096, lr=1e-4),
}
CFG = _PRESETS[MODEL_SIZE]

SMOKE_STEPS     = 30        # pre-flight: confirm loss is going down
FULL_STEPS      = 200       # full run -- ~20-30 min on T4 for 1.5B
EVAL_EVERY      = 25        # how often the live env-eval callback fires
EVAL_N_EPISODES = 6         # episodes per eval pass
RUN_ID          = f"sft_{MODEL_SIZE.lower()}"

print(f"OK preset = {MODEL_SIZE} | model = {CFG['model']}")


# %% [Cell 7] -- Load model + LoRA (vanilla transformers, fp16)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

tokenizer = AutoTokenizer.from_pretrained(CFG["model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CFG["model"],
    torch_dtype = torch.float16,
    device_map  = "cuda",
)
# Gradient checkpointing slashes activation memory (~3x) for SFT.
# enable_input_require_grads is needed for grad-ckpt + LoRA to coexist
# (the input embedding outputs need requires_grad=True so gradients flow).
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.0,
    bias           = "none",
    task_type      = "CAUSAL_LM",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("OK model + LoRA ready (vanilla transformers, fp16)")


# %% [Cell 8] -- Load dataset, pre-format with chat template, configure SFTTrainer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from notebooks.sft_callbacks import RFTEvalCallback

ds = load_dataset("json", data_files="/content/repo/data/sft.jsonl", split="train")
print(f"OK loaded {len(ds)} examples")

# Pre-apply Qwen's chat template so SFTTrainer reads a flat 'text' column.
# This is more portable across TRL versions than relying on auto-detection
# of the messages field.
def to_text(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
    }

ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])
print(f"OK formatted; sample text length = {len(ds[0]['text'])} chars")

sft_args = SFTConfig(
    output_dir                  = f"/content/repo/checkpoints/{RUN_ID}",
    per_device_train_batch_size = CFG["per_device_bs"],
    gradient_accumulation_steps = CFG["grad_accum"],
    max_length                  = CFG["max_seq_len"],   # was max_seq_length pre-TRL-0.18
    learning_rate               = CFG["lr"],
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    max_steps                   = SMOKE_STEPS,
    logging_steps               = 1,
    save_steps                  = max(50, FULL_STEPS),
    save_total_limit            = 2,
    report_to                   = "none",
    fp16                        = True,
    bf16                        = False,
    seed                        = 42,
    packing                     = False,
    dataset_text_field          = "text",
)

eval_cb = RFTEvalCallback(
    eval_every_steps = EVAL_EVERY,
    n_episodes       = EVAL_N_EPISODES,
    mutation_prob    = 0.0,
    seed             = 1234,
    max_new_tokens   = 1500,
    max_probes       = 12,
)
eval_cb.bind(model, tokenizer)

trainer = SFTTrainer(
    model            = model,
    train_dataset    = ds,
    args             = sft_args,
    processing_class = tokenizer,
    callbacks        = [eval_cb],
)
print("OK SFTTrainer ready")


# %% [Cell 9] -- BASELINE eval (BEFORE training) -- the "before" point
from notebooks.sft_eval import evaluate

print("Running baseline eval (8 episodes, base spec)...")
model.eval()
baseline_summary = evaluate(model, tokenizer, n_episodes=8, mutation_prob=0.0,
                            seed=999, max_new_tokens=1500, progress=True)
print(f"\nBaseline mean reward: {baseline_summary.mean_reward:.3f}  "
      f"parse_rate: {baseline_summary.parse_rate:.2f}")
model.train()


# %% [Cell 10] -- Smoke train (~3-5 min). Inspect the loss before going further.
trainer.train()


# %% [Cell 11] -- Full train (~20-30 min on T4 with 1.5B fp16)
# Self-contained: rebuilds sft_args + trainer from scratch so it's safe to
# run even if a partial kernel restart wiped Cell 6 / Cell 8 state. Requires
# only `model` and `tokenizer` in the namespace (from Cell 7).
import gc, torch, sys
sys.path.insert(0, "/content/repo")

try:
    _ = model; _ = tokenizer
except NameError:
    raise RuntimeError("model/tokenizer missing -- re-run Cell 7 first.")

from accelerate import Accelerator
if not getattr(Accelerator, "_protocol_one_patched", False):
    _orig_unwrap = Accelerator.unwrap_model
    def _patched_unwrap(self, model, *args, **kwargs):
        kwargs.pop('keep_torch_compile', None)
        return _orig_unwrap(self, model, *args, **kwargs)
    Accelerator.unwrap_model = _patched_unwrap
    Accelerator._protocol_one_patched = True

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from notebooks.sft_callbacks import RFTEvalCallback

ds = load_dataset("json", data_files="/content/repo/data/sft.jsonl", split="train")
def to_text(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)}
ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

RUN_ID, FULL_STEPS, EVAL_EVERY, EVAL_N_EPISODES = "sft_1.5b", 200, 25, 6
sft_args = SFTConfig(
    output_dir = f"/content/repo/checkpoints/{RUN_ID}",
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    max_length = 4096,
    learning_rate = 2e-4,
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.05,
    max_steps = FULL_STEPS,
    logging_steps = 1,
    save_steps = max(50, FULL_STEPS),
    save_total_limit = 2,
    report_to = "none",
    fp16 = True, bf16 = False, seed = 42,
    packing = False, dataset_text_field = "text",
)
eval_cb = RFTEvalCallback(
    eval_every_steps=EVAL_EVERY, n_episodes=EVAL_N_EPISODES,
    mutation_prob=0.0, seed=1234, max_new_tokens=1500, max_probes=12,
)
eval_cb.bind(model, tokenizer)
trainer = SFTTrainer(
    model=model, train_dataset=ds, args=sft_args,
    processing_class=tokenizer, callbacks=[eval_cb],
)

gc.collect(); torch.cuda.empty_cache()
trainer.train(resume_from_checkpoint=False)


# %% [Cell 12] -- TRAINED eval (after training) -- base spec
print("Running trained eval (12 episodes, base spec)...")
model.eval()
trained_summary = evaluate(model, tokenizer, n_episodes=12, mutation_prob=0.0,
                           seed=999, max_new_tokens=1500, progress=True)
print(f"\nTrained mean reward: {trained_summary.mean_reward:.3f}  "
      f"parse_rate: {trained_summary.parse_rate:.2f}")


# %% [Cell 13] -- TRAINED eval -- with mutations (generalization test)
print("Running mutation-generalization eval (12 episodes, mutation_prob=0.5)...")
mut_summary = evaluate(model, tokenizer, n_episodes=12, mutation_prob=0.5,
                       seed=2024, max_new_tokens=1500, progress=True)
print(f"\nTrained mean reward (mutated): {mut_summary.mean_reward:.3f}  "
      f"parse_rate: {mut_summary.parse_rate:.2f}")
model.train()


# %% [Cell 14] -- Generate all the eye-pleasing plots
from notebooks import plotting as P

os.makedirs("/content/repo/notebooks/figures", exist_ok=True)
fig_dir = "/content/repo/notebooks/figures"

print(P.plot_training_loss(trainer.state.log_history,
                           f"{fig_dir}/loss_curve.png"))
print(P.plot_eval_reward_curve(trainer.state.log_history,
                               f"{fig_dir}/reward_curve.png",
                               baseline=baseline_summary.mean_reward))
print(P.plot_component_breakdown(trainer.state.log_history,
                                 f"{fig_dir}/component_breakdown.png"))
print(P.plot_baseline_vs_trained(baseline_summary, trained_summary,
                                 f"{fig_dir}/baseline_vs_trained.png"))
print(P.plot_mutation_generalization(trained_summary, mut_summary,
                                     f"{fig_dir}/mutation_generalization.png"))
print(P.plot_dashboard(trainer.state.log_history, trainer.state.log_history,
                       baseline_summary, trained_summary,
                       f"{fig_dir}/dashboard.png",
                       baseline_reward=baseline_summary.mean_reward))
print("OK all figures written to notebooks/figures/")


# %% [Cell 15] -- Save numeric artifacts (used by the README and pitch deck)
import time
results = {
    "run_id":   RUN_ID,
    "model":    CFG["model"],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config":   {"smoke_steps": SMOKE_STEPS, "full_steps": FULL_STEPS,
                 "lr": CFG["lr"], "batch": CFG["per_device_bs"],
                 "grad_accum": CFG["grad_accum"]},
    "baseline": {"mean_reward": baseline_summary.mean_reward,
                 "std_reward":  baseline_summary.std_reward,
                 "parse_rate":  baseline_summary.parse_rate,
                 "components":  baseline_summary.component_means},
    "trained":  {"mean_reward": trained_summary.mean_reward,
                 "std_reward":  trained_summary.std_reward,
                 "parse_rate":  trained_summary.parse_rate,
                 "components":  trained_summary.component_means,
                 "by_variant":  trained_summary.by_variant},
    "trained_mutated": {"mean_reward": mut_summary.mean_reward,
                        "std_reward":  mut_summary.std_reward,
                        "by_variant":  mut_summary.by_variant},
}
out = f"/content/repo/notebooks/figures/results_{RUN_ID}.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"OK wrote {out}")
print(json.dumps(results, indent=2))


# %% [Cell 16] -- Save LoRA adapter for the inference demo
# Adapter is ~30 MB; merge into base model at inference time via merge_and_unload().
adapter_dir = f"/content/repo/checkpoints/{RUN_ID}_lora"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"OK LoRA adapter + tokenizer saved to {adapter_dir}")


# %% [Cell 17] -- Commit + push artifacts to GitHub
!cd /content/repo && git config user.email "you@example.com" && git config user.name "Suhani"
!cd /content/repo && git add notebooks/figures data/sft.jsonl notebooks/sft_eval.py notebooks/sft_callbacks.py notebooks/plotting.py notebooks/colab_cells_sft.py scripts/build_sft_dataset.py
!cd /content/repo && git commit -m "RFT pipeline (vanilla fp16): dataset, eval, plots, run artifacts"
!cd /content/repo && git push origin main
