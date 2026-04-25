# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""TrainerCallback for live evaluation against the env during SFT.

The training-loss curve from SFT is monotonic and uninformative for our
storytelling: cross-entropy on filtered transcripts goes down. What we
actually want to plot is *matcher reward improving over training* on
held-out probe transcripts, which is the metric that proves the model
generalized.

This callback runs ``sft_eval.evaluate`` every ``eval_every_steps`` and
writes the result into ``state.log_history`` under ``eval/*`` keys, so
the same plotting code that reads training loss can also read eval reward.
"""
from __future__ import annotations

from typing import Any

from transformers import TrainerCallback

from notebooks.sft_eval import evaluate


class RFTEvalCallback(TrainerCallback):
    """Periodic env-based evaluation during SFT training.

    Args:
        eval_every_steps: how often (in optimizer steps) to run an eval pass.
        n_episodes: episodes per eval pass. Keep small (8-16) -- this runs
            on the live model in inference mode and adds wall time per call.
        mutation_prob: probability of designer mutation per eval episode.
            Set 0.0 for clean base-spec eval, > 0.0 to test generalization.
        seed: deterministic eval set; same seed each call so the curve
            measures *the model improving*, not eval-set lottery.
    """

    def __init__(
        self,
        eval_every_steps: int = 25,
        n_episodes: int = 8,
        mutation_prob: float = 0.0,
        seed: int = 1234,
        max_new_tokens: int = 1500,
        max_probes: int = 12,
    ):
        self.eval_every_steps = eval_every_steps
        self.n_episodes = n_episodes
        self.mutation_prob = mutation_prob
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.max_probes = max_probes
        self._tokenizer = None
        self._model = None
        self._history: list[dict] = []

    def bind(self, model: Any, tokenizer: Any) -> None:
        """Required: call this once before trainer.train() so we hold
        references to the live model + tokenizer (Trainer sometimes wraps
        them in DataParallel and the kwargs in on_step_end are inconsistent
        across TRL/transformers versions)."""
        self._model = model
        self._tokenizer = tokenizer

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._model is None or self._tokenizer is None:
            return
        if state.global_step == 0:
            return
        if state.global_step % self.eval_every_steps != 0:
            return
        self._run_and_log(state)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        # Always log a final eval point so the curve has the endpoint.
        if self._model is None or self._tokenizer is None:
            return
        self._run_and_log(state)

    def _run_and_log(self, state: Any) -> None:
        was_training = self._model.training
        self._model.eval()
        try:
            summary = evaluate(
                self._model,
                self._tokenizer,
                n_episodes=self.n_episodes,
                mutation_prob=self.mutation_prob,
                seed=self.seed,
                max_new_tokens=self.max_new_tokens,
                max_probes=self.max_probes,
            )
        finally:
            if was_training:
                self._model.train()

        entry = {
            "step": state.global_step,
            "eval/reward_mean": summary.mean_reward,
            "eval/reward_std": summary.std_reward,
            "eval/parse_rate": summary.parse_rate,
        }
        for k, v in summary.component_means.items():
            entry[f"eval/component_{k}"] = v
        for k, v in summary.by_variant.items():
            entry[f"eval/variant_{k}"] = v
        state.log_history.append(entry)
        self._history.append(entry)

        bd = ", ".join(f"{k}={v:.2f}" for k, v in sorted(summary.component_means.items()))
        print(
            f"  [eval @ step {state.global_step}]  "
            f"reward = {summary.mean_reward:.3f} ± {summary.std_reward:.3f}  "
            f"parse_rate = {summary.parse_rate:.2f}  ({bd})"
        )

    @property
    def history(self) -> list[dict]:
        return list(self._history)
