"""TRL ``TrainerCallback`` that surfaces Protocol One per-rollout metrics.

The trainer wrapper (``notebooks.trainer_wrapper``) pushes a metrics dict
to ``ROLLOUT_METRICS_Q`` at every rollout's terminal step. This callback
drains the queue at each TRL logging step, computes window averages, and
writes them under ``env/*`` keys so they show up alongside ``train/*`` in
Trackio / wandb / TensorBoard. The per-component values are what tell the
storytelling-relevant story of *which* capability the agent improves on.
"""
from __future__ import annotations

from typing import Any

from transformers import TrainerCallback

from notebooks.trainer_wrapper import ROLLOUT_METRICS_Q


_COMPONENT_KEYS = (
    "endpoints_discovered",
    "endpoint_details",
    "resources",
    "state_machines",
    "auth",
    "penalty",
)


def _avg(items: list[dict], key: str) -> float | None:
    vals = [i.get(key) for i in items if i.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


class ProtocolOneMetricsCallback(TrainerCallback):
    """Logs Protocol-One-specific rollout metrics every logging step."""

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or not ROLLOUT_METRICS_Q:
            return
        items = list(ROLLOUT_METRICS_Q)
        ROLLOUT_METRICS_Q.clear()
        n = len(items)
        if n == 0:
            return

        logs["env/rollouts_in_window"] = n
        avg_probes = _avg(items, "probes_used")
        if avg_probes is not None:
            logs["env/avg_probes"] = avg_probes
        logs["env/finalize_rate"] = sum(1 for i in items if i.get("finalized")) / n
        for k in ("endpoints_in_belief", "scopes_in_belief",
                  "endpoints_found", "false_claims"):
            v = _avg(items, k)
            if v is not None:
                logs[f"env/avg_{k}"] = v

        # Per-component breakdown averages — the story of *which* capability
        # the agent is improving on.
        for ck in _COMPONENT_KEYS:
            vals = []
            for i in items:
                bd = i.get("breakdown") or {}
                if ck in bd:
                    vals.append(bd[ck])
            if vals:
                logs[f"env/component_{ck}"] = sum(vals) / len(vals)

        # Phase-4 mutation tracking. Silent unless mutations are on, so a
        # clean Phase-3 run produces no env/mutation_* keys.
        with_mut = sum(1 for i in items if i.get("mutation_type"))
        if with_mut:
            logs["env/mutation_rate"] = with_mut / n
            mut_types = [i.get("mutation_type") for i in items if i.get("mutation_type")]
            for mt in set(mut_types):
                logs[f"env/mutation_{mt}"] = mut_types.count(mt) / n
