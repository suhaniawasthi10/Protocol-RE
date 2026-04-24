"""Between-episode protocol mutator (scripted, not RL-trained).

Phase 1-3: disabled by default (MUTATION_PROBABILITY=0.0), so the deployed
pre-hackathon Space runs a clean baseline protocol.

Phase 4 (hackathon day 1): flip MUTATION_PROBABILITY up. The designer is
what produces the "dip-and-recover" reward curve that sells the demo.

Five mutation types from 06_PHASE_4_DESIGNER.md:
  - rename_field: rename a field on a resource
  - deprecate_endpoint: remove an endpoint entirely
  - swap_error_code: permute the error codes for one endpoint
  - tighten_auth_scope: bump an endpoint's scope requirement to admin
  - shift_state_transition: add a previously-forbidden transition

Mutations operate on a DEEP COPY of the spec dict so the base SPEC is never
permanently modified.
"""

from __future__ import annotations

import copy
import os
import random


class Designer:
    MUTATION_TYPES = (
        "rename_field",
        "deprecate_endpoint",
        "swap_error_code",
        "tighten_auth_scope",
        "shift_state_transition",
    )

    def __init__(
        self,
        base_spec: dict,
        mutation_start_episode: int | None = None,
        mutation_probability: float | None = None,
        seed: int = 42,
    ):
        """
        Args:
            base_spec: the canonical SPEC; never mutated in place.
            mutation_start_episode: mutations start being applied from this
                episode index onward (cooldown). If None, read from
                MUTATION_START_EPISODE env var (default 50).
            mutation_probability: chance per episode of applying a mutation
                once the cooldown has passed. If None, read from
                MUTATION_PROBABILITY env var (default 0.0 — disabled).
            seed: rng seed for reproducibility.
        """
        self.base_spec = copy.deepcopy(base_spec)
        if mutation_start_episode is None:
            mutation_start_episode = int(os.environ.get("MUTATION_START_EPISODE", "50"))
        if mutation_probability is None:
            mutation_probability = float(os.environ.get("MUTATION_PROBABILITY", "0.0"))
        self.mutation_start_episode = mutation_start_episode
        self.mutation_probability = mutation_probability
        self.episodes_seen = 0
        self.rng = random.Random(seed)
        self.last_mutation_log: dict | None = None

    # ------------------------------------------------------------------

    def maybe_mutate(self, base_spec: dict | None = None) -> dict:
        """Called by env.reset() once per episode. Returns a (possibly mutated) spec."""
        spec_src = base_spec if base_spec is not None else self.base_spec
        self.episodes_seen += 1
        if self.episodes_seen < self.mutation_start_episode:
            self.last_mutation_log = None
            return copy.deepcopy(spec_src)
        if self.rng.random() > self.mutation_probability:
            self.last_mutation_log = None
            return copy.deepcopy(spec_src)
        return self._apply_mutation(spec_src)

    def _apply_mutation(self, base: dict) -> dict:
        spec = copy.deepcopy(base)
        mutation_type = self.rng.choice(self.MUTATION_TYPES)
        method = getattr(self, f"_mutate_{mutation_type}")
        log = method(spec)
        self.last_mutation_log = {"type": mutation_type, **log}
        return spec

    # ------------------------------------------------------------------
    # Individual mutations — each returns a dict that becomes part of
    # last_mutation_log for offline analysis.
    # ------------------------------------------------------------------

    def _mutate_rename_field(self, spec: dict) -> dict:
        resource = self.rng.choice(spec["resources"])
        renameable = [f for f in resource["fields"] if f["name"] != "id"]
        if not renameable:
            return {"applied": False, "reason": "no_renameable_fields"}
        field = self.rng.choice(renameable)
        old = field["name"]
        new = f"{old}_v2"
        field["name"] = new
        return {"applied": True, "resource": resource["name"], "old": old, "new": new}

    def _mutate_deprecate_endpoint(self, spec: dict) -> dict:
        # Keep core endpoints alive so the task stays learnable.
        protected = {"list_users", "get_user", "health"}
        candidates = [e for e in spec["endpoints"] if e["id"] not in protected]
        if not candidates:
            return {"applied": False, "reason": "no_deprecatable_endpoints"}
        target = self.rng.choice(candidates)
        spec["endpoints"].remove(target)
        return {"applied": True, "endpoint_id": target["id"], "path": target["path"]}

    def _mutate_swap_error_code(self, spec: dict) -> dict:
        candidates = [
            e for e in spec["endpoints"]
            if sum(1 for k in e.get("responses", {}) if k.startswith(("4", "5"))) >= 2
        ]
        if not candidates:
            return {"applied": False, "reason": "no_swap_candidates"}
        ep = self.rng.choice(candidates)
        error_codes = [k for k in ep["responses"] if k.startswith(("4", "5"))]
        c1, c2 = self.rng.sample(error_codes, 2)
        ep["responses"][c1], ep["responses"][c2] = ep["responses"][c2], ep["responses"][c1]
        return {"applied": True, "endpoint_id": ep["id"], "swapped": [c1, c2]}

    def _mutate_tighten_auth_scope(self, spec: dict) -> dict:
        candidates = [e for e in spec["endpoints"]
                      if e.get("auth_scope") and e.get("auth_scope") != "admin"]
        if not candidates:
            return {"applied": False, "reason": "no_scoped_endpoints"}
        ep = self.rng.choice(candidates)
        old = ep["auth_scope"]
        ep["auth_scope"] = "admin"
        return {"applied": True, "endpoint_id": ep["id"], "old_scope": old, "new_scope": "admin"}

    def _mutate_shift_state_transition(self, spec: dict) -> dict:
        # Only consider resources that actually have a currently-forbidden
        # transition available. A 2-state machine with both directions already
        # defined has nothing to mutate, so it gets filtered out here.
        eligible = []
        for r in spec["resources"]:
            sm = r.get("state_machine")
            if not sm:
                continue
            existing = {(t["from"], t["to"]) for t in sm.get("transitions", [])}
            states = sm["states"]
            candidates = [(a, b) for a in states for b in states
                          if a != b and (a, b) not in existing]
            if candidates:
                eligible.append((r, sm, candidates))
        if not eligible:
            return {"applied": False, "reason": "no_new_transitions_possible"}
        resource, sm, candidates = self.rng.choice(eligible)
        a, b = self.rng.choice(candidates)
        new_transition = {"from": a, "to": b,
                          "via": f"POST /{resource['name'].lower()}s/{{id}}/custom_{a}_to_{b}"}
        sm["transitions"].append(new_transition)
        return {"applied": True, "resource": resource["name"],
                "new_transition": new_transition}
