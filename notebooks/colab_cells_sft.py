"""Colab cell sequence for Protocol One — Rejection-Sampling SFT training.

This is the working pivot from broken multi-turn GRPO. We build a clean
(probe_transcript -> belief_graph_json) dataset from live env rollouts,
then SFT Qwen via Unsloth + TRL's SFTTrainer (which, unlike GRPOTrainer's
environment_factory, actually does what it claims).

Why SFT-with-rejection-sampling is defensible:
    - The official Hackathon Help Guide §3 prescribes SFT-first when the
      base model can't reliably produce successful rollouts (FAQ #16, #45).
    - The env is in the data-generation loop; the matcher is the verifier.
      That satisfies the "training loop connects to your environment"
      criterion -- just not via gradient backprop through the env.
    - Llama 2 used the same recipe (rejection-sampling fine-tuning, RFT).
      Academically known as STaR / expert iteration.

Cells are Jupytext-style. Copy each block between `# %%` markers into its
own Colab cell, in order.

Hardware presets at the top of Cell 7:
    T4 free  : MODEL_SIZE = "1.5B"  (Qwen 2.5-1.5B-Instruct, 4-bit)
    A100/L4  : MODEL_SIZE = "3B"    (Qwen 2.5-3B-Instruct,   4-bit)
    Larger   : MODEL_SIZE = "7B"    (Qwen 2.5-7B-Instruct,   4-bit) -- needs HF compute
"""

# %% [Cell 1] -- Verify GPU
!nvidia-smi


# %% [Cell 2] -- Canonical pinned install (same pins as the GRPO notebook;
# T4-tested by Unsloth and known-stable. Don't drift.)
import os
!pip install --upgrade -qqq uv

if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    try:
        import numpy, PIL
        _numpy = f"numpy=={numpy.__version__}"
        _pil = f"pillow=={PIL.__version__}"
    except Exception:
        _numpy, _pil = "numpy", "pillow"
    !uv pip install -qqq --upgrade {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
!uv pip install -qqq matplotlib
!uv pip install transformers==4.56.2
!uv pip install --no-deps trl==0.22.2
print("OK install set installed")


# %% [Cell 3] -- Remove torchcodec (ABI mismatch)
!pip uninstall -y torchcodec
print("OK torchcodec removed")


# %% [Cell 4] -- Verify versions
import importlib.metadata as md
for pkg, want in [("transformers", "4.56.2"), ("trl", "0.22.2"),
                  ("unsloth", "any"), ("matplotlib", "any")]:
    try:
        got = md.version(pkg)
    except Exception:
        got = "MISSING"
    print(f"{pkg:14} = {got:12} (want {want})")


# %% [Cell 5] -- Clone repo + install
%cd /content
!rm -rf /content/repo
!git clone https://github.com/suhaniawasthi10/Protocol-RE.git /content/repo
%cd /content/repo
!pip install -q -e .
print("OK repo cloned + installed editable")


# %% [Cell 6] -- Build the SFT dataset (~3-5 min for 1500 episodes)
# Mix base spec + Designer mutations so the model sees diverse transcripts.
# Threshold 0.40 keeps the top half of trajectories; calibrate up/down based
# on the score distribution printed at the end of this cell.
!python -m scripts.build_sft_dataset \
    --episodes 1500 \
    --threshold 0.40 \
    --mutation-prob 0.25 \
    --max-probes 12 \
    --out data/sft.jsonl


# %% [Cell 7] -- Imports + hardware-aware config
import os, sys, json
sys.path.insert(0, "/content/repo")

# === Hardware preset =========================================================
# Pick ONE: "1.5B" for T4 free, "3B" for L4/A100, "7B" for HF compute.
MODEL_SIZE = "1.5B"
# =============================================================================

_PRESETS = {
    "1.5B": dict(model="unsloth/Qwen2.5-1.5B-Instruct", per_device_bs=2,
                 grad_accum=4, max_seq_len=4096, lr=2e-4),
    "3B":   dict(model="unsloth/Qwen2.5-3B-Instruct",   per_device_bs=2,
                 grad_accum=4, max_seq_len=4096, lr=1.5e-4),
    "7B":   dict(model="unsloth/Qwen2.5-7B-Instruct",   per_device_bs=1,
                 grad_accum=8, max_seq_len=4096, lr=1e-4),
}
CFG = _PRESETS[MODEL_SIZE]

SMOKE_STEPS  = 30          # pre-flight: confirm loss is going down
FULL_STEPS   = 200         # full run -- ~10-25 min on T4
EVAL_EVERY   = 25          # how often the live env-eval callback fires
EVAL_N_EPISODES = 6        # episodes per eval pass (keep small -- generation is slow)
RUN_ID       = f"sft_{MODEL_SIZE.lower()}"

print(f"OK preset = {MODEL_SIZE} | model = {CFG['model']}")


# %% [Cell 8] -- Load model + LoRA via Unsloth (4-bit)
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = CFG["model"],
    max_seq_length  = CFG["max_seq_len"],
    load_in_4bit    = True,
    dtype           = None,
)
model = FastLanguageModel.get_peft_model(
    model, r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha    = 32,
    lora_dropout  = 0.0,
    bias          = "none",
    use_gradient_checkpointing = "unsloth",
    random_state  = 42,
)
print("OK model + LoRA ready")


# %% [Cell 9] -- Load dataset, configure SFTTrainer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from notebooks.sft_callbacks import RFTEvalCallback

ds = load_dataset("json", data_files="/content/repo/data/sft.jsonl", split="train")
print(f"OK loaded {len(ds)} examples; columns = {ds.column_names}")
print(f"   sample message roles: {[m['role'] for m in ds[0]['messages']]}")

sft_args = SFTConfig(
    output_dir                  = f"/content/repo/checkpoints/{RUN_ID}",
    per_device_train_batch_size = CFG["per_device_bs"],
    gradient_accumulation_steps = CFG["grad_accum"],
    max_seq_length              = CFG["max_seq_len"],
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


# %% [Cell 10] -- BASELINE eval (BEFORE training) -- the "before" curve point
from notebooks.sft_eval import evaluate
FastLanguageModel.for_inference(model)
print("Running baseline eval (8 episodes, base spec)...")
baseline_summary = evaluate(model, tokenizer, n_episodes=8, mutation_prob=0.0,
                            seed=999, max_new_tokens=1500, progress=True)
print(f"\nBaseline mean reward: {baseline_summary.mean_reward:.3f}  "
      f"parse_rate: {baseline_summary.parse_rate:.2f}")
FastLanguageModel.for_training(model)


# %% [Cell 11] -- Smoke train (~3-5 min). Inspect the loss before going further.
trainer.train()


# %% [Cell 12] -- Full train (~15-25 min on T4 with 1.5B)
import gc, torch
gc.collect(); torch.cuda.empty_cache()
sft_args.max_steps = FULL_STEPS
trainer.args = sft_args
trainer.train(resume_from_checkpoint=False)


# %% [Cell 13] -- TRAINED eval (after training) -- base spec
FastLanguageModel.for_inference(model)
print("Running trained eval (12 episodes, base spec)...")
trained_summary = evaluate(model, tokenizer, n_episodes=12, mutation_prob=0.0,
                           seed=999, max_new_tokens=1500, progress=True)
print(f"\nTrained mean reward: {trained_summary.mean_reward:.3f}  "
      f"parse_rate: {trained_summary.parse_rate:.2f}")


# %% [Cell 14] -- TRAINED eval -- with mutations (generalization test)
print("Running mutation-generalization eval (12 episodes, mutation_prob=0.5)...")
mut_summary = evaluate(model, tokenizer, n_episodes=12, mutation_prob=0.5,
                       seed=2024, max_new_tokens=1500, progress=True)
print(f"\nTrained mean reward (mutated): {mut_summary.mean_reward:.3f}  "
      f"parse_rate: {mut_summary.parse_rate:.2f}")
FastLanguageModel.for_training(model)


# %% [Cell 15] -- Generate all the eye-pleasing plots
import os
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


# %% [Cell 16] -- Save numeric artifacts (used by the README and pitch deck)
import json, time
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


# %% [Cell 17] -- Commit + push artifacts to GitHub
!cd /content/repo && git config user.email "you@example.com" && git config user.name "Suhani"
!cd /content/repo && git add notebooks/figures data/sft.jsonl notebooks/sft_eval.py notebooks/sft_callbacks.py notebooks/plotting.py notebooks/colab_cells_sft.py scripts/build_sft_dataset.py
!cd /content/repo && git commit -m "RFT pipeline: dataset, eval, plots, and run artifacts"
!cd /content/repo && git push origin main


# %% [Cell 18] -- (optional) Save the merged LoRA weights for the inference demo
# The pitch demo loads this checkpoint to drive the LIVE env via the OpenEnv
# WebSocket -- different code path from training, but uses the same model.
model.save_pretrained_merged(
    f"/content/repo/checkpoints/{RUN_ID}_merged",
    tokenizer,
    save_method = "merged_4bit_forced",   # safe path per Unsloth QLoRA warning
)
print("OK merged checkpoint saved")
