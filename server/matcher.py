"""Reward function: score a belief graph against the hidden spec.

Pure, deterministic, no I/O. Six components weighted to total in [0, 1]:
  - 0.30 endpoints discovered (method+path matched)
  - 0.20 endpoint details correct (auth, params, responses)
  - 0.15 resource fields correct
  - 0.10 state-machine transitions correct
  - 0.10 auth model correct
  - 0.15 false-claim penalty (capped, subtracted)

Forgiving of malformed input — returns a zero-ish MatcherResult rather than
raising. The agent is allowed to emit garbage; we just don't reward it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Weights sum to 1.00 so a perfect belief graph can land near 1.0.
# Endpoint discovery dominates because that's the first thing an agent learns;
# auth gets a relatively heavy share because it's discoverable in 1-2 probes
# and gives early reward signal.
WEIGHTS = {
    "endpoints_discovered": 0.35,
    "endpoint_details": 0.25,
    "resources": 0.15,
    "state_machines": 0.10,
    "auth": 0.15,
}
# Per-item penalty intentionally small: the first correct endpoint is worth
# 0.35/18 ≈ 0.019, so anything above ~0.015 per false claim would make a
# single guess net-negative. We keep it well below that. Cap saturates the
# total penalty so spam can't drive reward arbitrarily down.
PENALTY_CAP = 0.15
PENALTY_PER_FALSE_ENDPOINT = 0.01
PENALTY_PER_FALSE_RESOURCE = 0.01
PENALTY_PER_FALSE_SCOPE = 0.01

_TYPE_SYNONYMS = {
    "integer": "int",
    "boolean": "bool",
    "datetime": "timestamp",
    "str": "string",
}


@dataclass
class MatcherResult:
    total: float = 0.0
    endpoints_found: int = 0
    endpoints_total: int = 0
    false_claims: int = 0
    breakdown: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

def normalize_path(p: Any) -> str:
    if not isinstance(p, str):
        return ""
    p = p.strip().lower()
    if not p:
        return ""
    p = p.rstrip("/")
    if not p.startswith("/"):
        p = "/" + p
    # Replace any {xxx} identifier with {id}
    p = re.sub(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", "{id}", p)
    return p


def normalize_method(m: Any) -> str:
    return m.upper().strip() if isinstance(m, str) else ""


def normalize_type(t: Any) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip().lower()
    return _TYPE_SYNONYMS.get(t, t)


def _ep_key(ep: dict) -> tuple[str, str]:
    return normalize_method(ep.get("method")), normalize_path(ep.get("path"))


def _spec_endpoint_keys(spec: dict) -> set[tuple[str, str]]:
    return {(e["method"].upper(), normalize_path(e["path"])) for e in spec["endpoints"]}


# ---------------------------------------------------------------------------
# Cleaning / coercion
# ---------------------------------------------------------------------------

def _coerce_belief_graph(bg: Any) -> dict:
    """Make `bg` look like a well-shaped belief graph; drop malformed entries."""
    if not isinstance(bg, dict):
        return {"endpoints": [], "resources": [], "auth": {}}

    endpoints_raw = bg.get("endpoints", [])
    resources_raw = bg.get("resources", [])
    auth_raw = bg.get("auth", {})

    endpoints = [
        e for e in (endpoints_raw if isinstance(endpoints_raw, list) else [])
        if isinstance(e, dict) and isinstance(e.get("method"), str) and isinstance(e.get("path"), str)
    ]
    resources = [
        r for r in (resources_raw if isinstance(resources_raw, list) else [])
        if isinstance(r, dict) and isinstance(r.get("name"), str)
    ]
    auth = auth_raw if isinstance(auth_raw, dict) else {}

    return {"endpoints": endpoints, "resources": resources, "auth": auth}


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def score_endpoints_discovered(bg: dict, spec: dict) -> tuple[float, int]:
    spec_keys = _spec_endpoint_keys(spec)
    belief_keys = {_ep_key(e) for e in bg["endpoints"]}
    belief_keys.discard(("", ""))
    correct = belief_keys & spec_keys
    return (len(correct) / max(len(spec_keys), 1)), len(correct)


def score_endpoint_details(bg: dict, spec: dict) -> float:
    spec_by_key = {(e["method"].upper(), normalize_path(e["path"])): e for e in spec["endpoints"]}
    total_weight = 0.0
    total_score = 0.0

    seen: set[tuple[str, str]] = set()
    for b_ep in bg["endpoints"]:
        key = _ep_key(b_ep)
        if key not in spec_by_key or key in seen:
            continue
        seen.add(key)
        s_ep = spec_by_key[key]

        # Auth required?
        if "auth_required" in b_ep:
            total_weight += 1.0
            if bool(b_ep.get("auth_required")) == bool(s_ep.get("auth_required", False)):
                total_score += 1.0

        # Auth scope (only if spec demands one)
        if s_ep.get("auth_scope"):
            total_weight += 1.0
            if b_ep.get("auth_scope") == s_ep["auth_scope"]:
                total_score += 1.0

        # Query params
        spec_qp = {p["name"] for p in s_ep.get("query_params", [])}
        if spec_qp:
            total_weight += 1.0
            belief_qp = {p["name"] for p in b_ep.get("params", [])
                         if isinstance(p, dict) and p.get("location") == "query" and isinstance(p.get("name"), str)}
            total_score += len(belief_qp & spec_qp) / len(spec_qp)

        # Body fields
        spec_body = {p["name"] for p in s_ep.get("body_fields", [])}
        if spec_body:
            total_weight += 1.0
            belief_body = {p["name"] for p in b_ep.get("params", [])
                           if isinstance(p, dict) and p.get("location") == "body" and isinstance(p.get("name"), str)}
            total_score += len(belief_body & spec_body) / len(spec_body)

        # Path params
        spec_pp = {p["name"] for p in s_ep.get("path_params", [])}
        if spec_pp:
            total_weight += 1.0
            belief_pp = {p["name"] for p in b_ep.get("params", [])
                         if isinstance(p, dict) and p.get("location") == "path" and isinstance(p.get("name"), str)}
            # Path placeholders normalize to {id} so we accept any name as matching the path-param slot.
            # If the agent declared at least as many path params as the spec, give full credit.
            if len(belief_pp) >= len(spec_pp):
                total_score += 1.0
            else:
                total_score += len(belief_pp & spec_pp) / len(spec_pp)

        # Response codes
        spec_codes = set(s_ep.get("responses", {}).keys())
        if spec_codes and isinstance(b_ep.get("responses"), dict):
            total_weight += 1.0
            belief_codes = {str(k) for k in b_ep["responses"].keys()}
            total_score += len(belief_codes & spec_codes) / len(spec_codes)

    if total_weight == 0:
        return 0.0
    return total_score / total_weight


def score_resources(bg: dict, spec: dict) -> float:
    spec_by_name = {r["name"].lower(): r for r in spec["resources"]}
    if not spec_by_name:
        return 0.0

    total = 0.0
    seen: set[str] = set()
    for b_r in bg["resources"]:
        name = (b_r.get("name") or "").lower()
        if name not in spec_by_name or name in seen:
            continue
        seen.add(name)
        s_r = spec_by_name[name]
        spec_fields = {f["name"].lower() for f in s_r["fields"]}
        belief_fields = {f["name"].lower() for f in (b_r.get("fields") or [])
                         if isinstance(f, dict) and isinstance(f.get("name"), str)}
        if spec_fields:
            total += len(spec_fields & belief_fields) / len(spec_fields)

    return total / len(spec_by_name)


def score_state_machines(bg: dict, spec: dict) -> float:
    spec_sms = {r["name"].lower(): r.get("state_machine")
                for r in spec["resources"] if r.get("state_machine")}
    if not spec_sms:
        return 0.0

    total_weight = 0
    total_correct = 0
    for r_name, s_sm in spec_sms.items():
        s_transitions = {(t["from"], t["to"]) for t in s_sm.get("transitions", [])}
        total_weight += len(s_transitions)
        # find matching belief resource
        b_sm = None
        for b_r in bg["resources"]:
            if (b_r.get("name") or "").lower() == r_name:
                b_sm = b_r.get("state_machine")
                break
        if not isinstance(b_sm, dict):
            continue
        b_transitions = {(t.get("from"), t.get("to")) for t in (b_sm.get("transitions") or [])
                         if isinstance(t, dict) and t.get("from") and t.get("to")}
        total_correct += len(s_transitions & b_transitions)

    if total_weight == 0:
        return 0.0
    return total_correct / total_weight


def score_auth(bg: dict, spec: dict) -> float:
    b_auth = bg.get("auth", {}) or {}
    s_auth = spec.get("auth", {}) or {}

    s = 0.0
    if isinstance(b_auth.get("type"), str) and b_auth["type"].lower() == s_auth.get("type", "").lower():
        s += 0.4

    s_scopes = set(s_auth.get("scopes", []))
    b_scopes_raw = b_auth.get("scopes_observed", [])
    b_scopes = set(b_scopes_raw) if isinstance(b_scopes_raw, list) else set()
    if s_scopes:
        s += 0.6 * len(b_scopes & s_scopes) / len(s_scopes)
    return s


def false_claim_penalty(bg: dict, spec: dict) -> tuple[float, int]:
    spec_eps = _spec_endpoint_keys(spec)
    spec_resources = {r["name"].lower() for r in spec["resources"]}
    spec_scopes = set(spec.get("auth", {}).get("scopes", []))

    # Dedup belief endpoints before counting false ones
    belief_keys = {_ep_key(e) for e in bg["endpoints"]}
    belief_keys.discard(("", ""))
    false_endpoints = sum(1 for k in belief_keys if k not in spec_eps)

    belief_resources = {(r.get("name") or "").lower() for r in bg["resources"]}
    belief_resources.discard("")
    false_resources = sum(1 for n in belief_resources if n not in spec_resources)

    b_scopes_raw = (bg.get("auth", {}) or {}).get("scopes_observed", []) or []
    belief_scopes = set(b_scopes_raw) if isinstance(b_scopes_raw, list) else set()
    false_scopes = sum(1 for s in belief_scopes if s not in spec_scopes)

    raw = (PENALTY_PER_FALSE_ENDPOINT * false_endpoints
           + PENALTY_PER_FALSE_RESOURCE * false_resources
           + PENALTY_PER_FALSE_SCOPE * false_scopes)
    return min(raw, PENALTY_CAP), false_endpoints + false_resources + false_scopes


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def score(belief_graph: Any, spec: dict) -> MatcherResult:
    bg = _coerce_belief_graph(belief_graph)

    s1, n_correct = score_endpoints_discovered(bg, spec)
    s2 = score_endpoint_details(bg, spec)
    s3 = score_resources(bg, spec)
    s4 = score_state_machines(bg, spec)
    s5 = score_auth(bg, spec)
    penalty, n_false = false_claim_penalty(bg, spec)

    total = (
        WEIGHTS["endpoints_discovered"] * s1
        + WEIGHTS["endpoint_details"] * s2
        + WEIGHTS["resources"] * s3
        + WEIGHTS["state_machines"] * s4
        + WEIGHTS["auth"] * s5
        - penalty
    )
    total = max(0.0, min(1.0, total))

    return MatcherResult(
        total=total,
        endpoints_found=n_correct,
        endpoints_total=len(spec["endpoints"]),
        false_claims=n_false,
        breakdown={
            "endpoints_discovered": s1,
            "endpoint_details": s2,
            "resources": s3,
            "state_machines": s4,
            "auth": s5,
            "penalty": penalty,
        },
    )
