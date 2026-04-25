# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""End-to-end evaluation of an SFT-trained model against the live env.

Used by:
    - notebooks.sft_callbacks.RFTEvalCallback (during training, periodic eval)
    - colab_cells_sft.py (post-training baseline vs trained comparison)

Pipeline per episode:
    1. Roll out fresh probe transcript via the same scripted strategies used
       to build the SFT dataset (so eval distribution matches train).
    2. Format the transcript into the SFT prompt (system + user message).
    3. model.generate() the belief graph completion.
    4. Parse the JSON; tolerate fenced blocks and trailing prose.
    5. score(belief, spec) via the matcher; return reward + breakdown.

Mutation handling: when mutation_prob > 0, episodes drawn from the same
Designer mutation taxonomy used in training. Pass mutation_prob=0 for a
clean base-spec eval.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

# Same upstream pieces used by the dataset builder so eval and train are
# definitionally aligned.
from scripts.build_sft_dataset import (  # type: ignore[import-not-found]
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    execute_probes,
    format_transcript,
    pick_probes,
)
from server.designer import Designer
from server.matcher import score
from server.protocol_server import MockProtocolServer
from server.spec import SPEC


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class EpisodeResult:
    reward: float
    breakdown: dict[str, float]
    spec_variant: str
    parsed_ok: bool
    n_probes: int
    endpoints_found: int
    endpoints_total: int
    false_claims: int
    raw_completion: str = ""


@dataclass
class EvalSummary:
    n: int
    mean_reward: float
    std_reward: float
    parse_rate: float
    component_means: dict[str, float] = field(default_factory=dict)
    by_variant: dict[str, float] = field(default_factory=dict)
    episodes: list[EpisodeResult] = field(default_factory=list)


def parse_belief_graph(text: str) -> dict | None:
    """Try hard to extract a JSON object from a model completion."""
    if not text:
        return None
    # 1. Direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2. Fenced ```json ... ```
    m = _FENCE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 3. First {...} substring
    m = _OBJECT_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _build_prompt_messages(transcript: str, n_probes: int) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(n=n_probes, transcript=transcript)},
    ]


def run_one_eval_episode(
    model: Any,
    tokenizer: Any,
    rng: random.Random,
    mutation_prob: float = 0.0,
    max_probes: int = 12,
    max_new_tokens: int = 1500,
    temperature: float = 0.0,
) -> EpisodeResult:
    """Run a single (probe -> generate -> score) cycle."""
    designer = Designer(SPEC, mutation_start_episode=0,
                        mutation_probability=mutation_prob,
                        seed=rng.randint(0, 2**31))
    spec_used = designer.maybe_mutate()
    spec_variant = (designer.last_mutation_log or {}).get("type", "base")

    server = MockProtocolServer(spec_used)
    client = TestClient(server.app)
    probes = pick_probes(rng, max_probes)
    observed = execute_probes(client, probes)
    transcript = format_transcript(observed)

    messages = _build_prompt_messages(transcript, len(observed))
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    import torch
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion_ids = out[0][inputs.shape[1]:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

    belief = parse_belief_graph(completion)
    parsed_ok = belief is not None
    result = score(belief or {}, spec_used)

    return EpisodeResult(
        reward=result.total,
        breakdown=dict(result.breakdown),
        spec_variant=spec_variant,
        parsed_ok=parsed_ok,
        n_probes=len(observed),
        endpoints_found=result.endpoints_found,
        endpoints_total=result.endpoints_total,
        false_claims=result.false_claims,
        raw_completion=completion[:2000],
    )


def evaluate(
    model: Any,
    tokenizer: Any,
    n_episodes: int = 16,
    mutation_prob: float = 0.0,
    seed: int = 1234,
    max_probes: int = 12,
    max_new_tokens: int = 1500,
    temperature: float = 0.0,
    progress: bool = False,
) -> EvalSummary:
    """Run n_episodes and aggregate."""
    rng = random.Random(seed)
    episodes: list[EpisodeResult] = []
    for i in range(n_episodes):
        ep = run_one_eval_episode(
            model, tokenizer, rng,
            mutation_prob=mutation_prob,
            max_probes=max_probes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        episodes.append(ep)
        if progress:
            print(f"  [{i + 1}/{n_episodes}] variant={ep.spec_variant:<22} "
                  f"reward={ep.reward:.3f}  parsed={ep.parsed_ok}")

    rewards = [e.reward for e in episodes]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards) if rewards else 0.0
    parse_rate = sum(1 for e in episodes if e.parsed_ok) / len(episodes) if episodes else 0.0

    # Per-component means
    component_means: dict[str, float] = {}
    if episodes:
        keys = set().union(*(e.breakdown.keys() for e in episodes))
        for k in keys:
            vals = [e.breakdown.get(k, 0.0) for e in episodes]
            component_means[k] = sum(vals) / len(vals)

    # Per-variant mean reward
    by_variant: dict[str, list[float]] = {}
    for e in episodes:
        by_variant.setdefault(e.spec_variant, []).append(e.reward)
    variant_means = {k: sum(v) / len(v) for k, v in by_variant.items()}

    return EvalSummary(
        n=len(episodes),
        mean_reward=mean,
        std_reward=var ** 0.5,
        parse_rate=parse_rate,
        component_means=component_means,
        by_variant=variant_means,
        episodes=episodes,
    )
