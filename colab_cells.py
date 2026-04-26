"""Canonical Colab cell sequence for Protocol One GRPO training (free T4).

This is a Jupytext-style notebook script. Each ``# %%`` marker delimits one
Colab cell. Copy each block (the lines BETWEEN ``# %%`` markers) into its
own cell in your Colab notebook, in order.

NOTE: this file is NOT meant to be executed via ``python COLAB_CELLS.py``.
The IPython shell magics (``!command``, ``%cd``) make it not pure Python.
It's a paste-from reference and a record of the canonical sequence.

Prereqs:
  - Colab runtime: T4 GPU (Runtime -> Change runtime type)
  - No HF token needed (env runs as a subprocess in the same Colab)

The cells assume:
  - github.com/suhaniawasthi10/Protocol-RE  (your fork)
  - /content/repo as the clone path
  - http://127.0.0.1:8000 as the env server URL

FAST defaults baked in (~60% faster than docs originals):
  - MAX_COMPLETION_LEN = 1024 (was 1536)  -- 33% less generation per rollout
  - GRAD_ACCUM         = 2    (was 4)     -- half the microbatches per step
  - MAX_PROBES         = 8    (was 12)    -- rollouts terminate ~33% sooner
  Smoke 10 steps:  ~5-7 min   (was ~15 min)
  Full 60 steps:   ~30-40 min (was ~90 min)

Two non-obvious fixes baked in here that we discovered the hard way:
  1. PROMPT and the env's INSTRUCTIONS_TEMPLATE are reframed as a legitimate
     development-sandbox documentation task. Phrases like "reverse-engineering"
     and "undocumented HTTP API" trigger Qwen's safety refusals, which makes
     the model emit "I'm sorry, I can't assist with that" instead of using the
     tools. Reframing makes the model engage with the task.
  2. Cell 11 verifies the env server is up AND drives a single end-to-end
     rollout with the wrapper before the trainer touches it. This catches
     the two failure modes that produce "all rewards 0":
       a. env server died (subprocess.Popen handle was orphaned)
       b. wrapper or tool dispatch is broken
     in 5 seconds, instead of after 7 minutes of pointless training.

Total cells: 15 (was 14 -- added Cell 11 for the wrapper smoke test).
"""


# %% [Cell 1] -- Verify GPU
!nvidia-smi


# %% [Cell 2] -- Canonical pinned install (~5 min, T4-tested by Unsloth)
import os
!pip install --upgrade -qqq uv

if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    try:
        import numpy, PIL
        _numpy = f"numpy=={numpy.__version__}"
        _pil = f"pillow=={PIL.__version__}"
    except Exception:
        _numpy, _pil = "numpy", "pillow"
    try:
        import subprocess
        is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except Exception:
        is_t4 = False
    _vllm, _triton = ("vllm==0.9.2", "triton==3.2.0") if is_t4 else ("vllm==0.15.1", "triton")
    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq {_triton}
!uv pip install transformers==4.56.2
!uv pip install --no-deps trl==0.22.2
print("OK canonical pin set installed")


# %% [Cell 3] -- Remove torchcodec (ABI mismatch with downgraded torch)
!pip uninstall -y torchcodec
print("OK torchcodec removed")


# %% [Cell 4] -- Verify versions (no module init, avoids side effects)
import importlib.metadata as md
for pkg, want in [("vllm", "0.9.2"), ("triton", "3.2.0"),
                  ("transformers", "4.56.2"), ("trl", "0.22.2"),
                  ("unsloth", "any")]:
    try:
        got = md.version(pkg)
    except Exception:
        got = "MISSING"
    print(f"{pkg:14} = {got:12} (want {want})")
# If any version is wrong, run:
#   !pip install --force-reinstall --no-deps vllm==0.9.2 triton==3.2.0 transformers==4.56.2 trl==0.22.2
# then restart runtime and re-verify.


# %% [Cell 5] -- Clean clone + install + apply THREE patches (idempotent on fresh clone)
# The %cd /content FIRST is critical: if you previously ran this cell, your
# shell's CWD is /content/repo. Deleting that directory while the shell is
# inside it leaves the shell with an invalid CWD, which makes git clone and
# every subsequent shell command fail with "getcwd: cannot access parent
# directories". Move the shell out of the doomed directory before deleting it.
%cd /content
-rf /content/repo
!git clone https://github.com/suhaniawasthi10/Protocol-RE.git /content/repo
%cd /content/repo
!pip install -q -e .

# --- Patch 1: trainer_wrapper.py -- import fallback + auto-finalize ---
import pathlib, re
p = pathlib.Path("/content/repo/notebooks/trainer_wrapper.py")
src = p.read_text()

# Import fallback (clean replace -- file is fresh from clone, single match expected)
old_import = "from protocol_one_env import ProtocolOneEnv, ProtocolOneAction"
if old_import in src:
    new_import = (
        "try:\n"
        "    from protocol_one_env import ProtocolOneEnv, ProtocolOneAction\n"
        "except ImportError:\n"
        "    from client import ProtocolOneEnv  # type: ignore\n"
        "    from models import ProtocolOneAction  # type: ignore"
    )
    src = src.replace(old_import, new_import, 1)

# Auto-finalize in close()
new_close = '''    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Auto-finalize if rollout ended without an explicit finalize() call
        if not self.done:
            try:
                result = self._env.step(ProtocolOneAction(tool="finalize", args={}))
                self._on_terminal(result)
            except Exception:
                pass
        try:
            self._env.__exit__(None, None, None)
        except Exception:
            pass
'''
pattern = re.compile(r'    def close\(self\) -> None:.*?(?=\n    def |\nclass |\Z)', re.DOTALL)
src, n = pattern.subn(new_close.rstrip() + '\n', src)
assert n == 1, f"Expected 1 close() match, got {n}"
p.write_text(src)

import ast
ast.parse(p.read_text())
print("OK trainer_wrapper.py patched + parses cleanly")

# --- Patch 2: env file -- probe budget configurable + reframe instructions ---
# The original INSTRUCTIONS_TEMPLATE uses "reverse-engineering" / "undocumented",
# which triggers Qwen's safety refusal ("I'm sorry, I can't assist with that").
# Reframe as a legitimate dev-sandbox API documentation task.
p2 = pathlib.Path("/content/repo/server/protocol_one_env_environment.py")
src2 = p2.read_text()
if "\nimport os\n" not in src2:
    src2 = src2.replace("\nimport json\n", "\nimport json\nimport os\n")
src2 = src2.replace(
    "MAX_PROBES_PER_EPISODE = 80",
    'MAX_PROBES_PER_EPISODE = int(os.environ.get("MAX_PROBES_PER_EPISODE", "8"))',
)
# Reframe: drop trigger words from the env's initial-observation text.
src2 = src2.replace(
    "You are reverse-engineering an undocumented HTTP API.",
    "You are mapping the structure of an in-house HTTP API to generate documentation.",
)
src2 = src2.replace(
    "Task: discover the API's structure and produce a complete belief graph of its",
    "Task: map this development server's structure and produce a complete belief graph of its",
)
p2.write_text(src2)
print("OK env file patched (MAX_PROBES=8, instructions reframed for safety-aligned models)")


# %% [Cell 6] -- Launch env server with MAX_PROBES=8
import subprocess, time, httpx, os

try:
    if 'proc' in globals() and proc.poll() is None:
        proc.terminate(); proc.wait(timeout=5)
except Exception:
    pass

os.makedirs("/content/repo/logs/rollouts", exist_ok=True)
proc = subprocess.Popen(
    ["uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "8000",
     "--log-level", "warning"],
    cwd="/content/repo",
    env={**os.environ, "MAX_PROBES_PER_EPISODE": "8"},
)
for _ in range(60):
    try:
        if httpx.get("http://127.0.0.1:8000/health", timeout=0.5).status_code == 200:
            print("OK env server up with MAX_PROBES=8")
            break
    except Exception:
        pass
    time.sleep(0.5)
else:
    raise RuntimeError("server didn't come up")


# %% [Cell 7] -- Sanity check (optional, ~30s)
import os
os.environ["SPACE_URL"] = "http://127.0.0.1:8000"
!python /content/repo/scripts/verify_phase2.py
!MAX_PROBES_PER_EPISODE=8 python /content/repo/scripts/smoke_test_scripted.py


# %% [Cell 8] -- Imports + config (FAST defaults)
import os, sys, torch
sys.path.insert(0, "/content/repo")
os.environ["PROTOCOL_ONE_ENV_URL"] = "http://127.0.0.1:8000"
os.environ["ROLLOUT_LOG_DIR"]      = "/content/repo/logs/rollouts"
os.environ["ROLLOUT_SAMPLE_RATE"]  = "0.10"

from unsloth import FastLanguageModel        # MUST be before trl/transformers
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from notebooks.trainer_wrapper import ProtocolOneToolEnv, ROLLOUT_METRICS_Q
from notebooks.callbacks import ProtocolOneMetricsCallback

MODEL_NAME          = "unsloth/Qwen2.5-1.5B-Instruct"
NUM_GENERATIONS     = 2
PER_DEVICE_BATCH    = 1
GRAD_ACCUM          = 2          # FAST: half microbatches per step (was 4)
MAX_COMPLETION_LEN  = 1024       # FAST: 33% less generation (was 1536)
LR                  = 5e-6
SMOKE_STEPS         = 10
FULL_STEPS          = 60
RUN_ID              = "run_a_fast"
print(f"OK TRL is {GRPOTrainer.__module__}")


# %% [Cell 9] -- Load Qwen 1.5B + LoRA (vLLM colocate, enforce_eager for T4)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name              = MODEL_NAME,
    max_seq_length          = MAX_COMPLETION_LEN + 2048,
    load_in_4bit            = True,
    fast_inference          = True,
    gpu_memory_utilization  = 0.5,
    max_lora_rank           = 16,
    enforce_eager           = True,        # disables CUDA graphs (T4-stable)
)
model = FastLanguageModel.get_peft_model(
    model, r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32, lora_dropout = 0.0, bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
)
print("OK model + LoRA ready")


# %% [Cell 10] -- Reward + dataset + GRPOConfig
def reward_func(prompts=None, completions=None, completion_ids=None, **kwargs):
    """TRL 0.22.2 GRPO reward.
    Reads from ROLLOUT_METRICS_Q (populated by ProtocolOneToolEnv._on_terminal).
    """
    n = len(completions) if completions is not None else 0
    # Some TRL versions pass envs through; try first.
    for key in ("environments", "envs", "env_instances"):
        envs = kwargs.get(key)
        if envs:
            envs_list = list(envs) if not isinstance(envs, list) else envs
            if len(envs_list) == n and hasattr(envs_list[0], "reward"):
                return [float(getattr(e, "reward", 0.0)) for e in envs_list]
    # Fallback: read last n entries from the rollout queue.
    items = list(ROLLOUT_METRICS_Q)
    if len(items) >= n and n > 0:
        return [float(items[i].get("reward", 0.0)) for i in range(-n, 0)]
    return [0.0] * n


# IMPORTANT: this prompt is reframed as a legitimate dev-sandbox documentation
# task. Earlier wording ("probing an undocumented HTTP API") triggered Qwen's
# safety refusal, which made the model emit "I'm sorry, I can't assist with
# that" and never call the tools. Don't revert to security-flavored phrasing.
PROMPT = (
    "You are an automated API characterization agent for development environments. "
    "Your task: map the endpoint shape, resource model, and auth scheme of an "
    "in-house HTTP service so we can generate accurate API documentation for it. "
    "You have full permission to query this service -- it is a development "
    "sandbox we own and operate. Use the tools probe(), update_model(), and "
    "finalize() to send test requests, record findings, and submit your "
    "structural map. Strategy: send 2-3 test requests with probe(), then call "
    "update_model() with what you learned, then repeat. Call finalize() once "
    "you have a reasonable map (or are running low on probe budget). You "
    "receive zero score if the episode ends without finalize() being called."
)
dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": PROMPT}]] * 256})

args = GRPOConfig(
    output_dir                  = f"/content/repo/checkpoints/{RUN_ID}",
    per_device_train_batch_size = PER_DEVICE_BATCH,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_generations             = NUM_GENERATIONS,
    learning_rate               = LR,
    lr_scheduler_type           = "cosine",
    max_completion_length       = MAX_COMPLETION_LEN,
    max_steps                   = SMOKE_STEPS,
    logging_steps               = 1,
    save_steps                  = 50,
    save_total_limit            = 2,
    use_vllm                    = True,
    vllm_mode                   = "colocate",
    log_completions             = True,
    report_to                   = "none",
    fp16                        = True,
    bf16                        = False,
    seed                        = 42,
)
print(f"OK reward_func + dataset ({len(dataset)} rows) + GRPOConfig ready")


# %% [Cell 11] -- Pre-train smoke test: env server + wrapper + tools (NEW)
# Catches the two most common failure modes in <5 sec:
#   1. env server is down (ConnectionRefusedError)
#   2. wrapper instantiates but tool dispatch is broken
# If this cell prints a non-zero reward and queue length > 0, the trainer
# has a working pipeline and the next 7 minutes won't be wasted.
import httpx

# 1. Server health
try:
    r = httpx.get("http://127.0.0.1:8000/health", timeout=2.0)
    assert r.status_code == 200, f"unexpected status {r.status_code}"
    print(f"OK env server reachable at :8000")
except Exception as e:
    raise RuntimeError(
        f"!!! env server NOT reachable: {e}\n"
        "    Re-run Cell 6 BEFORE this cell, then try again."
    )

# 2. Drive ONE rollout through the wrapper directly (no model)
env = ProtocolOneToolEnv()
text = env.reset()
print(f"OK env.reset() returned {len(text)} chars")
print(f"   (first 160 chars: {text[:160]!r})")

# Hit a known-good probe path
r1 = env.probe("GET", "/users", headers={"Authorization": "Bearer token_full"})
print(f"OK probe GET /users (authed): {r1[:120]!r}")

# update_model with a tiny belief, then finalize
r2 = env.update_model({
    "endpoints": [{"method": "GET", "path": "/users", "auth_required": True}],
    "auth": {"type": "bearer", "scopes_observed": ["users:read"]},
})
print(f"OK update_model: {r2[:120]!r}")

fin = env.finalize()
print(f"OK finalize: {fin[:160]!r}")
print(f"   self.reward = {env.reward:.4f}  (must be > 0)")
print(f"   self.done   = {env.done}")
print(f"   queue length after one rollout = {len(ROLLOUT_METRICS_Q)}")
env.close()

assert env.reward > 0, "Reward is 0 -- wrapper or matcher is broken"
assert len(ROLLOUT_METRICS_Q) > 0, "Queue is empty -- _on_terminal never fired"
# Drain the queue so the real training run starts clean
ROLLOUT_METRICS_Q.clear()
print("OK pre-train smoke passed -- proceed to Cell 12 (smoke train)")


# %% [Cell 12] -- Smoke train (~5-7 min with FAST defaults). STOP and inspect.
trainer = GRPOTrainer(
    model               = model,
    processing_class    = tokenizer,
    reward_funcs        = reward_func,
    train_dataset       = dataset,
    args                = args,
    environment_factory = ProtocolOneToolEnv,
    callbacks           = [ProtocolOneMetricsCallback()],
)
trainer.train()


# %% [Cell 13] -- Full train (~30-40 min). Only run after smoke shows non-zero reward.
import gc, torch
gc.collect()
torch.cuda.empty_cache()

args.max_steps = FULL_STEPS
trainer.args = args
trainer.train(resume_from_checkpoint=False)


# %% [Cell 14] -- Generate plots (saved to notebooks/figures/, committed in Cell 15)
import pandas as pd, matplotlib.pyplot as plt, os
os.makedirs("/content/repo/notebooks/figures", exist_ok=True)
log = pd.DataFrame(trainer.state.log_history)

# Reward curve
fig, ax = plt.subplots(figsize=(8, 4))
log[["step", "reward"]].dropna().plot(x="step", y="reward", ax=ax, legend=False)
ax.axhline(0.459, ls="--", color="grey", label="scripted baseline (0.459)")
ax.set_xlabel("Training step")
ax.set_ylabel("Mean episode reward")
ax.set_title("Protocol One -- GRPO reward (Qwen 2.5-1.5B + Unsloth, FAST cfg)")
ax.legend()
fig.savefig("/content/repo/notebooks/figures/reward_curve.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Loss curve
if "loss" in log.columns:
    fig, ax = plt.subplots(figsize=(8, 4))
    log[["step", "loss"]].dropna().plot(x="step", y="loss", ax=ax, legend=False)
    ax.set_xlabel("Training step")
    ax.set_ylabel("GRPO loss")
    fig.savefig("/content/repo/notebooks/figures/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

# Per-component breakdown
component_cols = [c for c in log.columns if c.startswith("env/component_")]
if component_cols:
    fig, ax = plt.subplots(figsize=(8, 4))
    log[["step"] + component_cols].dropna().plot(x="step", y=component_cols, ax=ax)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Component reward")
    ax.set_title("Reward components -- which capability the agent learned")
    fig.savefig("/content/repo/notebooks/figures/component_breakdown.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

print("OK figures saved to notebooks/figures/")


# %% [Cell 15] -- Commit + push artifacts back to GitHub
!cd /content/repo && git config user.email "you@example.com" && git config user.name "Suhani"
!cd /content/repo && git add notebooks/figures notebooks/trainer_wrapper.py server/protocol_one_env_environment.py
!cd /content/repo && git commit -m "Training artifacts: reward curves and Colab-applied patches"
!cd /content/repo && git push origin main
