# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""Dark-theme plotting for Protocol One results.

Palette and aesthetic match the split-panel viz mocked in
`docs/07_PHASE_5_DEMO.md`. Plots are saved as high-DPI PNGs into
`notebooks/figures/` and embedded in the README.

Visual contract:
    - Dark background, light grid, light text -- so the curve is the
      brightest thing on the page.
    - Baseline reference is a dashed grey horizontal -- any value above
      the dashed line is "model beat the scripted heuristic baseline."
    - Mutations get yellow vertical markers; eval points after a mutation
      that recover above baseline are the demo moment.
    - Component breakdown is a small-multiples panel: one tiny axes per
      reward component, so the reader can see *which* capability moved.
    - Baseline-vs-trained comparison is a single dual-bar with deltas
      annotated above each pair.
"""
from __future__ import annotations

import os
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


# ---------------------------------------------------------------------------
# Palette (kept consistent with the HTML viz in 07_PHASE_5_DEMO.md)
# ---------------------------------------------------------------------------

PALETTE = {
    "bg":           "#0f1117",   # deep navy
    "panel":        "#161a23",   # slightly lighter card
    "grid":         "#262b36",
    "text":         "#e6e6e6",
    "muted":        "#9097a8",
    "verified":     "#3dd68c",   # success green
    "partial":      "#f2c84b",   # warning yellow
    "false":        "#e44d4d",   # error red
    "accent":       "#5cabff",   # cool blue (trained model)
    "accent_warm":  "#c084fc",   # soft purple (alt series)
    "baseline":     "#9097a8",
}


def apply_dark_style() -> None:
    """Apply the project palette globally. Call once before plotting."""
    rcParams.update({
        "figure.facecolor":   PALETTE["bg"],
        "axes.facecolor":     PALETTE["bg"],
        "savefig.facecolor":  PALETTE["bg"],
        "axes.edgecolor":     PALETTE["grid"],
        "axes.labelcolor":    PALETTE["text"],
        "axes.titlecolor":    PALETTE["text"],
        "axes.titleweight":   "semibold",
        "axes.titlesize":     12,
        "axes.labelsize":     10,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         PALETTE["grid"],
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,
        "grid.alpha":         0.7,
        "xtick.color":        PALETTE["muted"],
        "ytick.color":        PALETTE["muted"],
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "text.color":         PALETTE["text"],
        "legend.facecolor":   PALETTE["panel"],
        "legend.edgecolor":   PALETTE["grid"],
        "legend.labelcolor":  PALETTE["text"],
        "legend.fontsize":    9,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        "figure.dpi":         110,
        "savefig.dpi":        160,
        "savefig.bbox":       "tight",
    })


def _save(fig: plt.Figure, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_training_loss(log_history: list[dict], out_path: str,
                       title: str = "SFT training loss — Qwen 2.5 + LoRA") -> str:
    """Cross-entropy curve from `trainer.state.log_history`."""
    apply_dark_style()
    rows = [(r["step"], r["loss"]) for r in log_history if "loss" in r and "step" in r]
    if not rows:
        rows = [(r.get("step", 0), r["loss"]) for r in log_history if "loss" in r]
    rows.sort()
    if not rows:
        raise ValueError("log_history has no 'loss' rows to plot")
    xs, ys = zip(*rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(xs, ys, color=PALETTE["accent"], lw=1.8, label="train loss")
    ax.fill_between(xs, ys, min(ys) * 0.9, color=PALETTE["accent"], alpha=0.10)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    return _save(fig, out_path)


def plot_eval_reward_curve(
    eval_history: list[dict],
    out_path: str,
    baseline: float | None = None,
    title: str = "Held-out matcher reward over training",
    mutation_steps: Iterable[int] = (),
) -> str:
    """Reward-on-eval-set curve. baseline is a dashed horizontal reference."""
    apply_dark_style()
    rows = [(r["step"], r["eval/reward_mean"], r.get("eval/reward_std", 0.0))
            for r in eval_history if "eval/reward_mean" in r]
    rows.sort()
    if not rows:
        raise ValueError("eval_history has no 'eval/reward_mean' rows")
    steps, means, stds = zip(*rows)
    means = np.array(means); stds = np.array(stds)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.fill_between(steps, means - stds, means + stds, color=PALETTE["accent"], alpha=0.18,
                    label="±1 std (eval episodes)")
    ax.plot(steps, means, color=PALETTE["accent"], lw=2.2, marker="o",
            markersize=5, label="mean reward")

    if baseline is not None:
        ax.axhline(baseline, ls="--", lw=1.2, color=PALETTE["baseline"],
                   label=f"untrained baseline ({baseline:.2f})")

    for ms in mutation_steps:
        ax.axvline(ms, ls=":", lw=1.0, color=PALETTE["partial"], alpha=0.85)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean matcher reward (held-out episodes)")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right")
    return _save(fig, out_path)


def plot_component_breakdown(
    eval_history: list[dict],
    out_path: str,
    title: str = "Reward components — which capability the agent learned",
) -> str:
    """Small multiples: one panel per matcher component."""
    apply_dark_style()
    component_keys = sorted({k for r in eval_history for k in r if k.startswith("eval/component_")})
    if not component_keys:
        raise ValueError("eval_history has no eval/component_* keys")

    n = len(component_keys)
    cols = 3
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(11, 2.6 * rows_n), sharex=True)
    axes = np.atleast_2d(axes).flatten()

    colors = [PALETTE["verified"], PALETTE["accent"], PALETTE["partial"],
              PALETTE["accent_warm"], PALETTE["false"], PALETTE["muted"]]

    for idx, key in enumerate(component_keys):
        ax = axes[idx]
        rows = [(r["step"], r[key]) for r in eval_history if key in r and "step" in r]
        rows.sort()
        if not rows:
            ax.set_visible(False); continue
        xs, ys = zip(*rows)
        c = colors[idx % len(colors)]
        ax.plot(xs, ys, color=c, lw=1.8, marker="o", markersize=3)
        ax.fill_between(xs, ys, color=c, alpha=0.12)
        nice_name = key.replace("eval/component_", "").replace("_", " ")
        ax.set_title(nice_name, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=8)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, color=PALETTE["text"], fontsize=12, y=1.01, weight="semibold")
    fig.text(0.5, -0.02, "Training step", ha="center", color=PALETTE["muted"])
    fig.tight_layout()
    return _save(fig, out_path)


def plot_baseline_vs_trained(
    baseline_summary: Any,
    trained_summary: Any,
    out_path: str,
    title: str = "Baseline vs SFT-trained — held-out matcher score",
) -> str:
    """Grouped bar chart over reward components + total."""
    apply_dark_style()
    components = sorted(set(baseline_summary.component_means) | set(trained_summary.component_means))
    labels = ["total"] + [c.replace("_", " ") for c in components]
    base = [baseline_summary.mean_reward] + [baseline_summary.component_means.get(c, 0.0) for c in components]
    trained = [trained_summary.mean_reward] + [trained_summary.component_means.get(c, 0.0) for c in components]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.6))
    b1 = ax.bar(x - w / 2, base, w, color=PALETTE["baseline"], label="untrained baseline",
                edgecolor=PALETTE["bg"], linewidth=0.8)
    b2 = ax.bar(x + w / 2, trained, w, color=PALETTE["accent"], label="SFT trained",
                edgecolor=PALETTE["bg"], linewidth=0.8)

    for x_i, (b, t, lab) in enumerate(zip(base, trained, labels)):
        delta = t - b
        # For "penalty", lower is better, so flip the polarity for color/sign.
        is_better = (delta < 0) if "penalty" in lab else (delta >= 0)
        sign = "+" if delta >= 0 else ""
        color = PALETTE["verified"] if is_better else PALETTE["false"]
        ax.text(x_i, max(b, t) + 0.025, f"{sign}{delta:.2f}",
                ha="center", color=color, fontsize=9, weight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", color=PALETTE["text"])
    ax.set_ylabel("Score (0 - 1)")
    ax.set_ylim(0, max(max(base + trained) + 0.18, 1.0))
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, out_path)


def plot_mutation_generalization(
    base_summary: Any,
    mutation_summary: Any,
    out_path: str,
    title: str = "Generalization to held-out mutations",
) -> str:
    """Per-variant reward bars: base spec vs each mutation type."""
    apply_dark_style()
    variants = sorted(set(base_summary.by_variant) | set(mutation_summary.by_variant))
    base_v = [base_summary.by_variant.get(v, np.nan) for v in variants]
    mut_v = [mutation_summary.by_variant.get(v, np.nan) for v in variants]

    x = np.arange(len(variants))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - w / 2, base_v, w, color=PALETTE["accent"], label="base-spec eval",
           edgecolor=PALETTE["bg"], linewidth=0.8)
    ax.bar(x + w / 2, mut_v, w, color=PALETTE["partial"], label="mutated-spec eval",
           edgecolor=PALETTE["bg"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=9, color=PALETTE["text"])
    ax.set_ylabel("Mean matcher reward")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, out_path)


def plot_dataset_calibration(
    score_buckets: dict[str, int],
    threshold: float,
    out_path: str,
    title: str = "SFT dataset score distribution",
) -> str:
    """Bar histogram of episode scores from build_sft_dataset (--stats-only)."""
    apply_dark_style()
    keys = sorted(score_buckets.keys(), key=float)
    vals = [score_buckets[k] for k in keys]
    colors = [PALETTE["false"] if float(k) < threshold else PALETTE["verified"] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(keys, vals, color=colors, edgecolor=PALETTE["bg"], linewidth=0.8)
    ax.axvline(x=str(round(threshold, 1)), ls="--", color=PALETTE["partial"],
               lw=1.2, label=f"keep threshold = {threshold:.2f}")
    ax.set_xlabel("Matcher score (binned)")
    ax.set_ylabel("Episode count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    return _save(fig, out_path)


def plot_dashboard(
    log_history: list[dict],
    eval_history: list[dict],
    baseline_summary: Any,
    trained_summary: Any,
    out_path: str,
    baseline_reward: float | None = None,
) -> str:
    """One eye-pleasing 2x2 hero image for the README."""
    apply_dark_style()
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # (0,0) Training loss
    ax1 = fig.add_subplot(gs[0, 0])
    rows = sorted([(r["step"], r["loss"]) for r in log_history if "loss" in r])
    if rows:
        xs, ys = zip(*rows)
        ax1.plot(xs, ys, color=PALETTE["accent"], lw=1.8)
        ax1.fill_between(xs, ys, color=PALETTE["accent"], alpha=0.12)
    ax1.set_title("SFT training loss")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Cross-entropy")

    # (0,1) Eval reward curve
    ax2 = fig.add_subplot(gs[0, 1])
    rows = sorted([(r["step"], r["eval/reward_mean"]) for r in eval_history
                   if "eval/reward_mean" in r])
    if rows:
        xs, ys = zip(*rows)
        ax2.plot(xs, ys, color=PALETTE["verified"], lw=2.2, marker="o", markersize=5)
        ax2.fill_between(xs, ys, color=PALETTE["verified"], alpha=0.15)
    if baseline_reward is not None:
        ax2.axhline(baseline_reward, ls="--", color=PALETTE["baseline"],
                    label=f"baseline {baseline_reward:.2f}")
        ax2.legend(loc="lower right")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Held-out reward over training")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Matcher reward")

    # (1,0) Baseline vs trained bars
    ax3 = fig.add_subplot(gs[1, 0])
    components = sorted(set(baseline_summary.component_means) | set(trained_summary.component_means))
    labels = ["total"] + [c.replace("_", " ")[:12] for c in components]
    base = [baseline_summary.mean_reward] + [baseline_summary.component_means.get(c, 0.0) for c in components]
    trained = [trained_summary.mean_reward] + [trained_summary.component_means.get(c, 0.0) for c in components]
    x = np.arange(len(labels)); w = 0.38
    ax3.bar(x - w / 2, base, w, color=PALETTE["baseline"], label="baseline")
    ax3.bar(x + w / 2, trained, w, color=PALETTE["accent"], label="trained")
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, max(max(base + trained) + 0.1, 1.0))
    ax3.set_title("Baseline vs trained (per component)")
    ax3.legend(loc="upper right")

    # (1,1) Per-variant generalization
    ax4 = fig.add_subplot(gs[1, 1])
    variants = sorted(set(baseline_summary.by_variant) | set(trained_summary.by_variant))
    base_v = [baseline_summary.by_variant.get(v, 0.0) for v in variants]
    train_v = [trained_summary.by_variant.get(v, 0.0) for v in variants]
    x = np.arange(len(variants)); w = 0.38
    ax4.bar(x - w / 2, base_v, w, color=PALETTE["baseline"], label="baseline")
    ax4.bar(x + w / 2, train_v, w, color=PALETTE["accent"], label="trained")
    ax4.set_xticks(x)
    ax4.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=8)
    ax4.set_ylabel("Mean reward")
    ax4.set_ylim(0, 1.0)
    ax4.set_title("Generalization across spec variants")
    ax4.legend(loc="upper right")

    fig.suptitle("Protocol One — SFT training results",
                 color=PALETTE["text"], fontsize=14, weight="semibold", y=0.995)
    return _save(fig, out_path)
