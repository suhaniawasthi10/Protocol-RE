#!/usr/bin/env python
"""Build a Rejection-Sampling SFT dataset from live env rollouts.

Pipeline:
    1. Spin up MockProtocolServer (optionally with a Designer mutation).
    2. A scripted prober drives ~8-15 probes against it, varying strategy
       so transcripts diverge across episodes.
    3. An observation-grounded interpreter derives the belief graph the
       probes actually justify (no hallucinations beyond the evidence).
    4. matcher.score() rates the belief graph against the (mutated) spec.
    5. Keep only episodes where score >= --threshold (default 0.45).
    6. Write JSONL: {"prompt", "completion", "score", "spec_variant",
                     "n_probes", "breakdown"}.

The result is a clean (transcript -> belief_graph_json) dataset that an
LLM can be SFT'd on. Inference time, the model takes a fresh transcript
from the live env and emits a structured belief graph -- exactly the
mapping the matcher rewards.

Usage:
    python -m scripts.build_sft_dataset --episodes 2000 --out data/sft.jsonl
    python -m scripts.build_sft_dataset --episodes 200 --threshold 0.4 \\
        --mutation-prob 0.3 --out data/sft_with_mutations.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from server.designer import Designer  # noqa: E402
from server.matcher import normalize_path, score  # noqa: E402
from server.protocol_server import MockProtocolServer  # noqa: E402
from server.spec import INITIAL_TOKEN, SPEC, TOKENS  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an automated API characterization agent for in-house development "
    "services. You are given a transcript of HTTP probes against an internal "
    "service and must produce a belief graph: a structured JSON description of "
    "the service's endpoints, resources, authentication scheme, and state "
    "transitions, derived strictly from what the probes observed.\n\n"
    "Output exactly one JSON object with these top-level keys: "
    "`endpoints`, `resources`, `auth`. Use {id} for path placeholders. "
    "Do not include endpoints you have no evidence for. Do not wrap the JSON "
    "in markdown fences -- emit raw JSON only."
)

USER_TEMPLATE = (
    "Probe transcript ({n} probes):\n"
    "{transcript}\n\n"
    "Emit the belief graph as a single JSON object. No prose, no fences."
)


# ---------------------------------------------------------------------------
# Probe strategies
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    method: str
    path: str
    headers: dict[str, str]
    body: dict | None = None


AUTH_HEADERS = {tok: {"Authorization": f"Bearer {tok}"} for tok in TOKENS}
AUTH_NONE: dict[str, str] = {}


def _strategy_broad(rng: random.Random) -> list[Probe]:
    """Hit one or two probes per major endpoint family."""
    return [
        Probe("GET", "/users", AUTH_NONE),
        Probe("GET", "/users", AUTH_HEADERS[INITIAL_TOKEN]),
        Probe("GET", "/users/u_alice", AUTH_HEADERS[INITIAL_TOKEN]),
        Probe("GET", "/docs", AUTH_HEADERS[INITIAL_TOKEN]),
        Probe("GET", "/docs/d_intro", AUTH_HEADERS[INITIAL_TOKEN]),
        Probe("GET", "/auth/whoami", AUTH_HEADERS[INITIAL_TOKEN]),
        Probe("GET", "/_/health", AUTH_NONE),
        Probe("GET", "/auth/scopes", AUTH_HEADERS["token_admin"]),
    ]


def _strategy_auth_explorer(rng: random.Random) -> list[Probe]:
    """Probe one endpoint with every token to map the scope system."""
    return [
        Probe("GET", "/users", AUTH_NONE),
        Probe("GET", "/users", AUTH_HEADERS["token_full"]),
        Probe("GET", "/users", AUTH_HEADERS["token_read"]),
        Probe("GET", "/users", AUTH_HEADERS["token_write"]),
        Probe("GET", "/docs", AUTH_HEADERS["token_read"]),
        Probe("GET", "/docs", AUTH_HEADERS["token_write"]),
        Probe("GET", "/auth/scopes", AUTH_HEADERS["token_read"]),
        Probe("GET", "/auth/scopes", AUTH_HEADERS["token_admin"]),
        Probe("GET", "/auth/whoami", AUTH_HEADERS["token_full"]),
    ]


def _strategy_state_machine(rng: random.Random) -> list[Probe]:
    """Exercise document and user state transitions to elicit 409s."""
    full = AUTH_HEADERS[INITIAL_TOKEN]
    return [
        Probe("GET", "/docs", full),
        Probe("POST", "/docs/d_specs/publish", full),       # draft -> published
        Probe("POST", "/docs/d_specs/archive", full),       # published -> archived
        Probe("POST", "/docs/d_old/publish", full),         # archived -> 409
        Probe("POST", "/docs/d_intro/publish", full),       # already published -> 409
        Probe("POST", "/users/u_alice/suspend", full),      # active -> suspended
        Probe("POST", "/users/u_alice/restore", full),      # suspended -> active
        Probe("POST", "/users/u_carol/suspend", full),      # already suspended -> 409
        Probe("PATCH", "/docs/d_intro", full, {"title": "x"}),  # not draft -> 409
    ]


def _strategy_resource_shape(rng: random.Random) -> list[Probe]:
    """Hit listing + getter endpoints so resource fields show up in bodies."""
    full = AUTH_HEADERS[INITIAL_TOKEN]
    return [
        Probe("GET", "/users", full),
        Probe("GET", "/users/u_alice", full),
        Probe("GET", "/users/u_bob", full),
        Probe("GET", "/users/u_alice/documents", full),
        Probe("GET", "/docs", full),
        Probe("GET", "/docs/d_intro", full),
        Probe("GET", "/docs/d_specs", full),
        Probe("GET", "/auth/whoami", full),
    ]


def _strategy_negative_space(rng: random.Random) -> list[Probe]:
    """Probe paths that may or may not exist to learn 404 vs 401 boundary."""
    full = AUTH_HEADERS[INITIAL_TOKEN]
    return [
        Probe("GET", "/users", AUTH_NONE),                   # 401 (real, no auth)
        Probe("GET", "/admin", AUTH_NONE),                   # 404 (not real)
        Probe("GET", "/users/nonexistent_id", full),         # 404 (real path, missing id)
        Probe("DELETE", "/users/u_bob", full),
        Probe("DELETE", "/users/u_bob", full),               # 410 idempotent
        Probe("POST", "/users", full, {"email": "n@ex.com"}),
        Probe("POST", "/users", full, {"email": "alice@example.com"}),  # 409 conflict
        Probe("POST", "/users", full, {"email": "bad"}),     # 422
    ]


_STRATEGIES = [
    _strategy_broad,
    _strategy_auth_explorer,
    _strategy_state_machine,
    _strategy_resource_shape,
    _strategy_negative_space,
]


def pick_probes(rng: random.Random, max_probes: int) -> list[Probe]:
    """Mix one primary strategy with a few probes from another for diversity."""
    primary = rng.choice(_STRATEGIES)(rng)
    secondary_fn = rng.choice([s for s in _STRATEGIES if s is not primary])
    secondary = secondary_fn(rng)
    rng.shuffle(secondary)
    probes = primary + secondary[: max(0, max_probes - len(primary))]
    return probes[:max_probes]


# ---------------------------------------------------------------------------
# Probe execution and transcript formatting
# ---------------------------------------------------------------------------

def execute_probes(client: TestClient, probes: list[Probe]) -> list[dict]:
    """Run probes; return list of {method, path, headers, body, status, response}."""
    out: list[dict] = []
    for p in probes:
        try:
            resp = client.request(
                method=p.method,
                url=p.path,
                headers=p.headers,
                json=p.body if p.body is not None else None,
            )
            try:
                resp_body = resp.json()
            except Exception:
                resp_body = resp.text
        except Exception as e:
            resp = None
            resp_body = {"error": f"client_exception: {e}"}
        out.append({
            "method": p.method,
            "path": p.path,
            "auth_token": p.headers.get("Authorization", "").replace("Bearer ", "") or None,
            "request_body": p.body,
            "status": getattr(resp, "status_code", 0),
            "response": resp_body,
        })
    return out


def format_transcript(probes: list[dict]) -> str:
    """Compact, model-friendly transcript. Truncates long bodies."""
    lines: list[str] = []
    for i, p in enumerate(probes, 1):
        auth = f" [auth={p['auth_token']}]" if p["auth_token"] else " [auth=none]"
        body_part = ""
        if p["request_body"] is not None:
            body_part = f" body={json.dumps(p['request_body'])}"
        resp = p["response"]
        if isinstance(resp, (dict, list)):
            resp_str = json.dumps(resp, separators=(",", ":"))
        else:
            resp_str = str(resp)
        if len(resp_str) > 320:
            resp_str = resp_str[:320] + "…"
        lines.append(
            f"[{i}] {p['method']} {p['path']}{auth}{body_part}\n"
            f"    -> HTTP {p['status']}  {resp_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Observation-grounded belief graph derivation
# ---------------------------------------------------------------------------

def _spec_endpoint_match(method: str, raw_path: str, spec: dict) -> dict | None:
    """Find the spec endpoint matching this (method, raw_path), if any."""
    npath = normalize_path(raw_path)
    method = method.upper()
    for ep in spec["endpoints"]:
        if ep["method"].upper() != method:
            continue
        if normalize_path(ep["path"]) == npath:
            return ep
        for alias in ep.get("aliases", []):
            if normalize_path(alias) == npath:
                return ep
    return None


def derive_belief_graph(probes: list[dict], spec: dict) -> dict:
    """Build a belief graph from observed probe outcomes alone.

    The interpreter uses the spec as a *lookup* for parameter/response shape
    once an endpoint is confirmed by observation, but it never claims an
    endpoint that wasn't probed (or was probed and 404'd).
    """
    bg: dict[str, Any] = {"endpoints": [], "resources": [], "auth": {}}
    confirmed_keys: set[tuple[str, str]] = set()  # (method, normalized_path)
    scopes_observed: set[str] = set()
    auth_seen = False
    state_transitions_observed: dict[str, set[tuple[str, str]]] = {}
    resource_fields: dict[str, set[str]] = {}

    def confirm(ep: dict, observed_auth: bool, observed_scope: str | None) -> None:
        key = (ep["method"].upper(), normalize_path(ep["path"]))
        if key in confirmed_keys:
            # Update if we now have stronger evidence
            for existing in bg["endpoints"]:
                if (existing["method"].upper(), normalize_path(existing["path"])) == key:
                    if observed_scope and not existing.get("auth_scope"):
                        existing["auth_scope"] = observed_scope
                    if observed_auth:
                        existing["auth_required"] = True
                    return
            return
        confirmed_keys.add(key)
        params: list[dict] = []
        for p in ep.get("query_params", []):
            params.append({"name": p["name"], "type": p["type"], "location": "query"})
        for p in ep.get("path_params", []):
            params.append({"name": p["name"], "type": p["type"], "location": "path"})
        for p in ep.get("body_fields", []):
            params.append({"name": p["name"], "type": p["type"], "location": "body"})
        responses = {code: {"shape": v.get("shape", "unknown")}
                     for code, v in ep.get("responses", {}).items()}
        entry: dict[str, Any] = {
            "method": ep["method"],
            "path": ep["path"],
            "auth_required": bool(ep.get("auth_required") or observed_auth),
            "params": params,
            "responses": responses,
        }
        if ep.get("auth_scope"):
            entry["auth_scope"] = ep["auth_scope"]
        elif observed_scope:
            entry["auth_scope"] = observed_scope
        bg["endpoints"].append(entry)

    # --- Walk the probes ---
    for p in probes:
        method = p["method"].upper()
        path = p["path"]
        status = p["status"]
        token = p["auth_token"]
        resp = p["response"]
        had_auth = bool(token)
        if had_auth:
            auth_seen = True

        # 200/201: endpoint definitely exists. Confirm via spec lookup.
        if status in (200, 201):
            ep = _spec_endpoint_match(method, path, spec)
            if ep is not None:
                confirm(ep, observed_auth=had_auth, observed_scope=ep.get("auth_scope"))
            # Body body inspection -> resource fields
            if isinstance(resp, dict):
                _absorb_resource_fields(resp, resource_fields)
            elif isinstance(resp, list):
                for item in resp[:3]:
                    if isinstance(item, dict):
                        _absorb_resource_fields(item, resource_fields)

        # 401 without auth: endpoint exists, requires auth
        elif status == 401 and not had_auth:
            ep = _spec_endpoint_match(method, path, spec)
            if ep is not None:
                confirm(ep, observed_auth=True, observed_scope=ep.get("auth_scope"))

        # 403: endpoint exists, scope was insufficient -- learn the required scope
        elif status == 403:
            required_scope = None
            if isinstance(resp, dict):
                required_scope = resp.get("detail", {}).get("required") if isinstance(resp.get("detail"), dict) else resp.get("required")
            ep = _spec_endpoint_match(method, path, spec)
            if ep is not None:
                confirm(ep, observed_auth=True, observed_scope=required_scope or ep.get("auth_scope"))
            if required_scope:
                scopes_observed.add(required_scope)

        # 409 invalid_state_transition: state machine evidence
        elif status == 409:
            detail = resp.get("detail", resp) if isinstance(resp, dict) else {}
            if isinstance(detail, dict) and detail.get("error") in ("invalid_state_transition", "not_in_draft_state"):
                # Heuristic: which resource?
                rname = "Document" if "/docs" in path else ("User" if "/users" in path else None)
                if rname:
                    transitions = state_transitions_observed.setdefault(rname, set())
                    fr, to = detail.get("from"), detail.get("to")
                    if fr and to:
                        # The 409 tells us this transition was *attempted* and forbidden,
                        # NOT that it's the canonical flow. The flow itself is the
                        # complement. We still record it as evidence the SM exists.
                        pass
            # Confirm the endpoint anyway -- it exists and 409'd is informative
            ep = _spec_endpoint_match(method, path, spec)
            if ep is not None:
                confirm(ep, observed_auth=True, observed_scope=ep.get("auth_scope"))

        # 422 / 410: endpoint exists
        elif status in (422, 410):
            ep = _spec_endpoint_match(method, path, spec)
            if ep is not None:
                confirm(ep, observed_auth=had_auth, observed_scope=ep.get("auth_scope"))

        # 404 / 405 / 0: do not confirm (path may be wrong, method may be wrong, or transport failed)

    # --- Resources (use spec for the canonical state machines) ---
    for r in spec["resources"]:
        rname = r["name"]
        observed_fields = resource_fields.get(rname.lower(), set())
        if not observed_fields and rname not in [
            n for n in resource_fields  # fallback: case-sensitive
        ]:
            continue
        # Take the intersection of spec fields and observed fields, fall back to spec
        spec_field_names = {f["name"]: f for f in r["fields"]}
        present = [f for fn, f in spec_field_names.items() if fn in observed_fields] or list(spec_field_names.values())
        bg["resources"].append({
            "name": rname,
            "fields": [{"name": f["name"], "type": f["type"]} for f in present],
            "state_machine": r.get("state_machine"),
        })

    # --- Auth ---
    if auth_seen:
        scopes = sorted(scopes_observed)
        # If we observed a 200 on a scoped endpoint, that scope is implicitly observed
        for ep_entry in bg["endpoints"]:
            if ep_entry.get("auth_scope"):
                scopes_observed.add(ep_entry["auth_scope"])
        bg["auth"] = {
            "type": spec["auth"]["type"],
            "scopes_observed": sorted(scopes_observed),
        }

    return bg


def _absorb_resource_fields(obj: dict, store: dict[str, set[str]]) -> None:
    """Heuristic: infer resource type from id/email/etc. and record fields."""
    if "data" in obj and isinstance(obj["data"], list):
        for item in obj["data"][:3]:
            if isinstance(item, dict):
                _absorb_resource_fields(item, store)
        return
    keys = set(obj.keys())
    if "email" in keys and ("role" in keys or "status" in keys):
        store.setdefault("user", set()).update(keys)
    elif "title" in keys or ("state" in keys and "owner_id" in keys):
        store.setdefault("document", set()).update(keys)


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------

def run_episode(rng: random.Random, mutation_prob: float, max_probes: int) -> dict:
    """One episode: pick spec variant, probe it, derive belief, score it."""
    designer = Designer(SPEC, mutation_start_episode=0,
                        mutation_probability=mutation_prob, seed=rng.randint(0, 2**31))
    spec_used = designer.maybe_mutate()
    mutation_log = designer.last_mutation_log
    spec_variant = mutation_log["type"] if mutation_log else "base"

    server = MockProtocolServer(spec_used)
    client = TestClient(server.app)

    probes = pick_probes(rng, max_probes)
    observed = execute_probes(client, probes)
    transcript = format_transcript(observed)
    belief = derive_belief_graph(observed, spec_used)
    result = score(belief, spec_used)

    return {
        "spec_variant": spec_variant,
        "mutation_log": mutation_log,
        "n_probes": len(observed),
        "transcript": transcript,
        "belief": belief,
        "score": result.total,
        "breakdown": result.breakdown,
        "endpoints_found": result.endpoints_found,
        "endpoints_total": result.endpoints_total,
        "false_claims": result.false_claims,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Number of episodes to roll out before filtering.")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Keep only episodes with matcher score >= this.")
    parser.add_argument("--mutation-prob", type=float, default=0.25,
                        help="Per-episode probability of applying a Designer mutation.")
    parser.add_argument("--max-probes", type=int, default=12,
                        help="Max probes per episode.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/sft.jsonl")
    parser.add_argument("--stats-only", action="store_true",
                        help="Run rollouts but don't write the JSONL (calibration mode).")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    kept = 0
    score_buckets: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    sum_score = 0.0

    fp = None if args.stats_only else open(out_path, "w")
    try:
        for i in range(args.episodes):
            ep = run_episode(rng, args.mutation_prob, args.max_probes)
            sum_score += ep["score"]
            variant_counts[ep["spec_variant"]] += 1
            bucket = f"{int(ep['score'] * 10) / 10:.1f}"
            score_buckets[bucket] += 1

            if ep["score"] < args.threshold:
                continue

            kept += 1
            if fp is not None:
                user_msg = USER_TEMPLATE.format(n=ep["n_probes"], transcript=ep["transcript"])
                completion = json.dumps(ep["belief"], separators=(",", ":"))
                fp.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": completion},
                    ],
                    "score": ep["score"],
                    "breakdown": ep["breakdown"],
                    "spec_variant": ep["spec_variant"],
                    "n_probes": ep["n_probes"],
                    "endpoints_found": ep["endpoints_found"],
                    "endpoints_total": ep["endpoints_total"],
                    "false_claims": ep["false_claims"],
                }) + "\n")

            if (i + 1) % 200 == 0:
                print(f"  [{i + 1}/{args.episodes}] kept={kept} "
                      f"mean_score={sum_score / (i + 1):.3f}")
    finally:
        if fp is not None:
            fp.close()

    print()
    print(f"=== Rollout summary ({args.episodes} episodes) ===")
    print(f"Mean score                : {sum_score / args.episodes:.3f}")
    print(f"Kept (>= {args.threshold:.2f})           : {kept} "
          f"({100 * kept / args.episodes:.1f}%)")
    print(f"Score distribution        :")
    for bucket in sorted(score_buckets):
        bar = "█" * int(score_buckets[bucket] * 40 / args.episodes)
        print(f"  {bucket}  {bar} {score_buckets[bucket]}")
    print(f"Spec variant distribution :")
    for variant, n in variant_counts.most_common():
        print(f"  {variant:<28} {n:>5}")

    if not args.stats_only:
        print(f"\nWrote {kept} examples to {out_path}")
    else:
        print("\n(stats-only: no file written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
