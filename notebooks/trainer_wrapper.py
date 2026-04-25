"""TRL ``environment_factory`` wrapper for the Protocol One environment.

Each ``ProtocolOneToolEnv`` instance owns one OpenEnv WebSocket session
against the Protocol One server (a local subprocess in Colab, or an
HF Space in production). TRL spawns ``num_generations`` instances per
training step.

The model's tool calls (``probe`` / ``update_model`` / ``finalize``) are
dispatched to ``ProtocolOneAction(tool=..., args=...)`` over the OpenEnv
client. The reward function reads ``env.reward`` after the rollout ends.

Reward design: terminal-only. ``self.reward`` stays at 0.0 until
``finalize()`` (or probe-budget exhaustion) sets it. This matches the
env's terminal-only reward and prevents shaping reward hacks.

Telemetry:
  - ``ROLLOUT_METRICS_Q``: bounded module-level deque drained by
    ``ProtocolOneMetricsCallback`` into TRL logs every logging step.
  - Per-rollout JSON snapshots under ``logs/rollouts/``, sampled at
    ``ROLLOUT_SAMPLE_RATE`` (default 0.05). Used to build the demo viz.

Configuration via env vars (all optional):
  PROTOCOL_ONE_ENV_URL  base URL of the env server (default localhost:8000)
  ROLLOUT_SAMPLE_RATE   probability of dumping a rollout snapshot (default 0.05)
  ROLLOUT_LOG_DIR       directory for snapshot dumps (default logs/rollouts)
"""
from __future__ import annotations

import collections
import json
import os
import random
import uuid
from typing import Any

try:
    # Preferred: the package-installed import path (`pip install -e .`).
    from protocol_one_env import ProtocolOneEnv, ProtocolOneAction
except ImportError:
    # Fallback: bare-module imports from the repo root. This handles
    # environments where the editable install of a `package-dir = {"x": "."}`
    # layout doesn't register the package on sys.path (notably in Colab),
    # but the repo root itself is on sys.path.
    from client import ProtocolOneEnv  # type: ignore
    from models import ProtocolOneAction  # type: ignore


ENV_URL = os.environ.get("PROTOCOL_ONE_ENV_URL", "http://127.0.0.1:8000")
ROLLOUT_SAMPLE_RATE = float(os.environ.get("ROLLOUT_SAMPLE_RATE", "0.05"))
ROLLOUT_DIR = os.environ.get("ROLLOUT_LOG_DIR", "logs/rollouts")

# Drained by notebooks.callbacks.ProtocolOneMetricsCallback. Bounded so a
# stuck or absent callback can't blow memory across a long run.
ROLLOUT_METRICS_Q: "collections.deque[dict[str, Any]]" = collections.deque(maxlen=4096)


class ProtocolOneToolEnv:
    """One-rollout TRL environment-factory class.

    TRL constructs one of these per parallel generation, calls ``reset()``,
    routes the model's tool calls to ``probe`` / ``update_model`` /
    ``finalize``, then reads ``env.reward`` for the reward function.
    """

    def __init__(self) -> None:
        self._env = ProtocolOneEnv(base_url=ENV_URL).sync()
        self._env.__enter__()
        self.reward: float = 0.0
        self.done: bool = False
        self._probe_log: list[dict[str, Any]] = []
        self._final_text: str = ""
        self._breakdown: dict[str, float] | None = None
        self._closed: bool = False

    # --- Lifecycle -----------------------------------------------------

    def reset(self, **kwargs: Any) -> str:
        """Start a fresh episode and return the initial instructions text."""
        result = self._env.reset()
        self.reward = 0.0
        self.done = False
        self._probe_log.clear()
        self._final_text = ""
        self._breakdown = None
        return result.observation.text or ""

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Auto-finalize if the rollout ended without an explicit finalize() call
        # (e.g., the model emitted EOS early or hit max_completion_length before
        # calling the finalize tool). The env scores whatever belief graph it
        # accumulated — a tiny but non-zero reward beats no reward at all,
        # because GRPO needs a relative signal within each generation group.
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

    def __del__(self) -> None:
        # Best-effort: never raise from __del__.
        try:
            self.close()
        except Exception:
            pass

    # --- Tool methods (introspected by TRL) ----------------------------
    # The docstrings below ARE the tool descriptions the model sees.
    # Keep them specific and example-rich; the model often copies the
    # examples verbatim into its first probes.

    def probe(
        self,
        method: str,
        path: str,
        headers: dict | None = None,
        body: dict | None = None,
    ) -> str:
        """Send an HTTP request to the undocumented API and observe the response.

        Args:
            method: HTTP method. One of: GET, POST, PUT, PATCH, DELETE.
            path: URL path including any query string. Examples:
                '/users', '/users/u_alice', '/users?limit=10', '/auth/whoami'.
            headers: Optional headers dict. To authenticate, pass
                {"Authorization": "Bearer <token>"}. The starting token is
                given to you in the system prompt; other tokens may exist
                with different scopes — discovering them is part of the task.
            body: Optional request body dict for POST/PUT/PATCH. Omit (or
                pass None) for GET/DELETE. Example: {"email": "x@y.com",
                "role": "admin"}.

        Returns:
            Response text including HTTP status code and response body
            (truncated if very long). Status codes are highly informative:
              200/201  - success;
              401      - missing or invalid auth (try a Bearer token);
              403      - valid auth but wrong scope (response usually names the required scope);
              404      - endpoint or resource not found;
              422      - malformed request (validation error);
              409      - state-machine conflict (e.g. publishing an archived doc);
              410      - already deleted (idempotent DELETE on second call).
        """
        if self.done:
            return "Episode already ended; reset() to start a new one."
        result = self._env.step(ProtocolOneAction(
            tool="probe",
            args={
                "method": str(method).upper(),
                "path": path,
                "headers": headers or {},
                "body": body,
            },
        ))
        text = result.observation.text or ""
        self._probe_log.append({
            "method": str(method).upper(),
            "path": path,
            "had_auth": bool((headers or {}).get("Authorization")),
            "first_line": text.split("\n", 1)[0][:200],
        })
        if result.done:
            self._on_terminal(result)
        return text[:3000]

    def update_model(self, delta: dict) -> str:
        """Merge new findings into your belief graph (incremental).

        Call this whenever you've learned something new from probes. Multiple
        calls are merged on the server; you do not need to repeat earlier
        findings each time.

        Args:
            delta: Partial belief graph in this exact shape (all top-level
                keys are optional; include only what's new):
                {
                  "endpoints": [
                    {"method": "GET", "path": "/users",
                     "auth_required": true, "auth_scope": "users:read",
                     "params": [{"name": "limit", "type": "int", "location": "query"}],
                     "responses": {"200": {"shape": "list<User>"},
                                   "401": {"shape": "error"}}}
                  ],
                  "resources": [
                    {"name": "User",
                     "fields": [{"name": "id", "type": "string"},
                                {"name": "email", "type": "string"}],
                     "state_machine": {
                        "states": ["active", "suspended"],
                        "transitions": [{"from": "active", "to": "suspended"}]
                     }}
                  ],
                  "auth": {"type": "bearer", "scopes_observed": ["users:read"]}
                }
              Path placeholders should use {id} (e.g. '/users/{id}').

        Returns:
            Confirmation string with current belief-graph statistics
            (counts of endpoints, resources, auth scopes observed).
        """
        if self.done:
            return "Episode already ended; reset() to start a new one."
        result = self._env.step(ProtocolOneAction(
            tool="update_model",
            args={"delta": delta or {}},
        ))
        if result.done:
            self._on_terminal(result)
        return (result.observation.text or "")[:1000]

    def finalize(self) -> str:
        """Submit your belief graph as final and end the episode.

        Call this exactly once, when your belief graph is as complete as
        you can make it within the probe budget. You receive zero reward
        if the episode ends without you calling finalize.

        Returns:
            Final reward summary including per-component breakdown.
        """
        if self.done:
            return f"Already finalized. Final reward: {self.reward:.3f}"
        result = self._env.step(ProtocolOneAction(tool="finalize", args={}))
        self._on_terminal(result)
        return (result.observation.text or "")[:2000]

    # --- Internal ------------------------------------------------------

    def _on_terminal(self, result: Any) -> None:
        self.reward = float(result.reward or 0.0)
        self.done = True
        obs = result.observation
        self._final_text = obs.text or ""
        md = getattr(obs, "metadata", {}) or {}
        self._breakdown = md.get("breakdown")
        bg_stats = getattr(obs, "belief_graph_stats", {}) or {}

        ROLLOUT_METRICS_Q.append({
            "reward": self.reward,
            "probes_used": getattr(obs, "probes_used", len(self._probe_log)),
            "endpoints_in_belief": bg_stats.get("endpoints", 0),
            "scopes_in_belief": bg_stats.get("auth_scopes_observed", 0),
            "finalized": True,
            "breakdown": self._breakdown or {},
            "endpoints_found": md.get("endpoints_found"),
            "endpoints_total": md.get("endpoints_total"),
            "false_claims": md.get("false_claims"),
            "mutation_type": (md.get("mutation_log") or {}).get("type")
                              if md.get("mutation_log") else None,
        })

        # Sample a rollout snapshot for the viz / qualitative analysis.
        # Never let snapshot I/O bubble up and break a rollout.
        if random.random() < ROLLOUT_SAMPLE_RATE:
            try:
                os.makedirs(ROLLOUT_DIR, exist_ok=True)
                fname = os.path.join(ROLLOUT_DIR, f"rollout_{uuid.uuid4().hex[:10]}.json")
                with open(fname, "w") as f:
                    json.dump({
                        "reward": self.reward,
                        "breakdown": self._breakdown,
                        "probes": self._probe_log,
                        "final_text": self._final_text[:2000],
                        "mutation_log": md.get("mutation_log"),
                    }, f)
            except Exception:
                pass
