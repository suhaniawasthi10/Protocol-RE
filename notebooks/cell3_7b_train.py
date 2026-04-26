import gc
gc.collect(); torch.cuda.empty_cache()

# Apply accelerate compatibility patch
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

ds = load_dataset("json", data_files="/home/user/Protocol-RE/data/sft.jsonl", split="train")

def to_text(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])
print(f"OK {len(ds)} examples formatted")

sft_args = SFTConfig(
    output_dir = f"/home/user/Protocol-RE/checkpoints/{RUN_ID}",
    per_device_train_batch_size = CFG["per_device_bs"],
    gradient_accumulation_steps = CFG["grad_accum"],
    max_length = CFG["max_seq_len"],
    learning_rate = CFG["lr"],
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.05,
    max_steps = FULL_STEPS,
    logging_steps = 1,
    save_steps = max(50, FULL_STEPS),
    save_total_limit = 2,
    report_to = "none",
    fp16 = True,
    bf16 = False,
    seed = 42,
    packing = False,
    dataset_text_field = "text",
)

eval_cb = RFTEvalCallback(
    eval_every_steps = EVAL_EVERY,
    n_episodes = EVAL_N_EPISODES,
    mutation_prob = 0.0,
    seed = 1234,
    max_new_tokens = 1500,
    max_probes = 12,
)
eval_cb.bind(model, tokenizer)

trainer = SFTTrainer(
    model = model,
    train_dataset = ds,
    args = sft_args,
    processing_class = tokenizer,
    callbacks = [eval_cb],
)

print("OK trainer ready, starting full train (200 steps, ~70 min)...")
trainer.train(resume_from_checkpoint=False)
