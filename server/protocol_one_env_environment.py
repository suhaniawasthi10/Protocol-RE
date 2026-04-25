# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Protocol One OpenEnv Environment.

One instance == one training rollout. On reset(), we:
  1. Ask the Designer for a (possibly mutated) copy of SPEC.
  2. Build a fresh MockProtocolServer off that spec, wrap in a TestClient.
  3. Reset belief_graph, probe_history, probe count.
  4. Return the initial observation with tool docs and a starter token.

Each step() receives a ProtocolOneAction with a tool discriminator and
dispatches to probe / update_model / finalize. Reward is only set at
terminal step — we deliberately avoid per-probe rewards to prevent the
model from reward-hacking by spamming update_model.
"""

from __future__ import annotations

import json
import os
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ProtocolOneAction, ProtocolOneObservation
except ImportError:
    from models import ProtocolOneAction, ProtocolOneObservation

from .designer import Designer
from .matcher import score
from .protocol_server import MockProtocolServer, create_server
from .spec import INITIAL_TOKEN, SPEC


MAX_PROBES_PER_EPISODE = int(os.environ.get("MAX_PROBES_PER_EPISODE", "12"))
RESPONSE_TRUNCATE_CHARS = 2000


INSTRUCTIONS_TEMPLATE = """You are reverse-engineering an undocumented HTTP API.

Task: discover the API's structure and produce a complete belief graph of its \
endpoints, resources, authentication, and state transitions.

Base URL: (implicit; the probe tool handles it — paths start with /)
Starting auth token: Bearer {token}

Additional tokens with other scopes may exist. Discovering them is part of the \
task. You have {max_probes} probes; unauthenticated probes are cheap ways to \
discover endpoints (401 vs 404 is informative).

Tools:
1. probe(method, path, headers, body): send an HTTP request, get back status + body
2. update_model(delta): merge findings into your belief graph (incremental)
3. finalize(final_belief_graph=None): commit and end the episode (reward is computed here)

Belief-graph shape (emit via update_model):
{{
  "endpoints": [
    {{"method": "GET", "path": "/users",
      "auth_required": true, "auth_scope": "users:read",
      "params": [{{"name": "limit", "type": "int", "location": "query"}}],
      "responses": {{"200": {{"shape": "list<User>"}}, "401": {{"shape": "error"}}}}}}
  ],
  "resources": [
    {{"name": "User",
      "fields": [{{"name": "id", "type": "string"}}, {{"name": "email", "type": "string"}}],
      "state_machine": {{"states": ["active", "suspended"],
                         "transitions": [{{"from": "active", "to": "suspended",
                                            "via": "POST /users/{{id}}/suspend"}}]}}}}
  ],
  "auth": {{"type": "bearer", "scopes_observed": ["users:read"]}}
}}

You are scored on:
- Endpoints correctly identified (method + path)
- Correct auth requirements and scopes
- Correct params and response codes
- Correct resource fields and state-machine transitions

False claims reduce your score. When uncertain, omit rather than guess. Call \
finalize() before the probe budget runs out; you get zero reward if you never finalize.

Begin probing."""


class ProtocolOneEnvironment(Environment):
    """OpenEnv environment for the Protocol One task."""

    # Each WebSocket session gets its own Environment instance; state never leaks
    # between rollouts, so we can safely run many concurrent sessions for GRPO.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # Persistent across episodes (mutation counter lives in the designer)
        self.designer = Designer(SPEC)

        # Refreshed per-episode in reset()
        self.current_spec: dict = SPEC
        self.server: MockProtocolServer | None = None
        self.client: TestClient | None = None
        self.belief_graph: dict[str, Any] = {"endpoints": [], "resources": [], "auth": {}}
        self.probe_history: list[dict[str, Any]] = []
        self.probes_used: int = 0
        self.done: bool = False
        self.last_reward: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv contract
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> ProtocolOneObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_spec = self.designer.maybe_mutate()
        self.server = create_server(self.current_spec)
        self.client = TestClient(self.server.app)
        self.belief_graph = {"endpoints": [], "resources": [], "auth": {}}
        self.probe_history = []
        self.probes_used = 0
        self.done = False
        self.last_reward = 0.0

        return ProtocolOneObservation(
            text=INSTRUCTIONS_TEMPLATE.format(
                token=INITIAL_TOKEN,
                max_probes=MAX_PROBES_PER_EPISODE,
            ),
            probes_used=0,
            probes_remaining=MAX_PROBES_PER_EPISODE,
            belief_graph_stats=self._stats(),
            done=False,
            reward=None,
            metadata={
                "episode_id": self._state.episode_id,
                "mutation_log": self.designer.last_mutation_log,
            },
        )

    def step(self, action: ProtocolOneAction, **kwargs: Any) -> ProtocolOneObservation:  # type: ignore[override]
        # Defensive: if a client steps without having called reset() first
        # (e.g., in a stateless HTTP smoke test), initialise episode state
        # on demand rather than crashing.
        if self.client is None:
            self.reset()

        if self.done:
            return self._obs("Episode already ended.", done=True)

        self._state.step_count += 1

        tool = action.tool
        args = action.args or {}

        if tool == "probe":
            return self._handle_probe(args)
        if tool == "update_model":
            return self._handle_update(args)
        if tool == "finalize":
            return self._handle_finalize(args)
        return self._obs(
            f"Invalid tool: {tool!r}. Must be 'probe', 'update_model', or 'finalize'.",
            done=False,
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_probe(self, args: dict[str, Any]) -> ProtocolOneObservation:
        if self.probes_used >= MAX_PROBES_PER_EPISODE:
            return self._end_episode("Probe budget exhausted.")

        method = str(args.get("method", "GET")).upper()
        path = str(args.get("path", "/"))
        headers = args.get("headers") or {}
        body = args.get("body")

        if not isinstance(headers, dict):
            headers = {}

        self.probes_used += 1
        assert self.client is not None

        try:
            if body is not None and method in ("GET", "DELETE"):
                # Silently drop body for methods that don't use it; saves the
                # agent from confusing 400s.
                body = None
            if body is not None and not isinstance(body, (dict, list, str)):
                body = None
            resp = self.client.request(
                method=method,
                url=path,
                headers={str(k): str(v) for k, v in headers.items()},
                json=body if body is not None else None,
            )
            resp_body = resp.text
            if len(resp_body) > RESPONSE_TRUNCATE_CHARS:
                resp_body = resp_body[:RESPONSE_TRUNCATE_CHARS] + "… [truncated]"
            text = (
                f"[Probe {self.probes_used}/{MAX_PROBES_PER_EPISODE}] "
                f"HTTP {resp.status_code}\nBody: {resp_body}"
            )
        except Exception as e:
            text = f"[Probe {self.probes_used}/{MAX_PROBES_PER_EPISODE}] Request error: {e}"

        self.probe_history.append({"method": method, "path": path, "response_text": text})
        return self._obs(text, done=False)

    def _handle_update(self, args: dict[str, Any]) -> ProtocolOneObservation:
        delta = args.get("delta")
        if delta is None:
            # Be forgiving: sometimes the agent emits the belief graph at
            # the top level instead of wrapping in {"delta": ...}.
            delta = args
        if isinstance(delta, str):
            # Forgive stringified JSON
            try:
                delta = json.loads(delta)
            except Exception:
                return self._obs(
                    f"update_model: could not parse delta as JSON. "
                    f"Expected {{'endpoints': [...], 'resources': [...], 'auth': {{...}}}}",
                    done=False,
                )
        if not isinstance(delta, dict):
            return self._obs(
                "update_model: delta must be a JSON object with endpoints/resources/auth keys.",
                done=False,
            )

        self._merge_delta(delta)
        stats = self._stats()
        return self._obs(
            f"Belief graph updated. Stats: {stats}. Call finalize() when done.",
            done=False,
        )

    def _handle_finalize(self, args: dict[str, Any]) -> ProtocolOneObservation:
        final = args.get("final_belief_graph")
        if isinstance(final, dict):
            # Replace, not merge — agent is committing a final answer.
            self.belief_graph = {
                "endpoints": final.get("endpoints", []),
                "resources": final.get("resources", []),
                "auth": final.get("auth", {}),
            }
        return self._end_episode("Finalized.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _merge_delta(self, delta: dict) -> None:
        # endpoints: upsert by (method, path) key
        if isinstance(delta.get("endpoints"), list):
            by_key: dict[tuple[str, str], dict] = {}
            for e in self.belief_graph["endpoints"]:
                by_key[(str(e.get("method", "")).upper(), str(e.get("path", "")))] = e
            for new_e in delta["endpoints"]:
                if not isinstance(new_e, dict):
                    continue
                key = (str(new_e.get("method", "")).upper(), str(new_e.get("path", "")))
                if key in by_key:
                    by_key[key] = {**by_key[key], **new_e}
                else:
                    by_key[key] = new_e
            self.belief_graph["endpoints"] = list(by_key.values())

        # resources: upsert by name
        if isinstance(delta.get("resources"), list):
            by_name: dict[str, dict] = {}
            for r in self.belief_graph["resources"]:
                by_name[str(r.get("name", ""))] = r
            for new_r in delta["resources"]:
                if not isinstance(new_r, dict):
                    continue
                name = str(new_r.get("name", ""))
                if name in by_name:
                    by_name[name] = {**by_name[name], **new_r}
                else:
                    by_name[name] = new_r
            self.belief_graph["resources"] = list(by_name.values())

        # auth: shallow merge, dedup scopes_observed
        if isinstance(delta.get("auth"), dict):
            new_auth = {**self.belief_graph.get("auth", {}), **delta["auth"]}
            if isinstance(new_auth.get("scopes_observed"), list):
                new_auth["scopes_observed"] = sorted({
                    s for s in new_auth["scopes_observed"] if isinstance(s, str)
                })
            self.belief_graph["auth"] = new_auth

    def _stats(self) -> dict[str, int]:
        auth = self.belief_graph.get("auth", {}) or {}
        scopes = auth.get("scopes_observed", []) or []
        return {
            "endpoints": len(self.belief_graph.get("endpoints", [])),
            "resources": len(self.belief_graph.get("resources", [])),
            "auth_scopes_observed": len(scopes) if isinstance(scopes, list) else 0,
        }

    def _obs(
        self,
        text: str,
        done: bool,
        reward: float | None = None,
        extra_metadata: dict | None = None,
    ) -> ProtocolOneObservation:
        metadata: dict[str, Any] = {"episode_id": self._state.episode_id}
        if extra_metadata:
            metadata.update(extra_metadata)
        return ProtocolOneObservation(
            text=text,
            probes_used=self.probes_used,
            probes_remaining=max(0, MAX_PROBES_PER_EPISODE - self.probes_used),
            belief_graph_stats=self._stats(),
            done=done,
            reward=reward,
            metadata=metadata,
        )

    def _end_episode(self, reason: str) -> ProtocolOneObservation:
        result = score(self.belief_graph, self.current_spec)
        self.last_reward = result.total
        self.done = True
        breakdown_str = ", ".join(f"{k}={v:.2f}" for k, v in sorted(result.breakdown.items()))
        text = (
            f"{reason}\n"
            f"Final reward: {result.total:.3f}  (endpoints_found={result.endpoints_found}"
            f"/{result.endpoints_total}, false_claims={result.false_claims})\n"
            f"Breakdown: {breakdown_str}"
        )
        return self._obs(
            text,
            done=True,
            reward=result.total,
            extra_metadata={
                "breakdown": result.breakdown,
                "endpoints_found": result.endpoints_found,
                "endpoints_total": result.endpoints_total,
                "false_claims": result.false_claims,
                "mutation_log": self.designer.last_mutation_log,
            },
        )
