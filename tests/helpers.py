"""Test helpers shared between matcher tests and the verify script."""

from typing import Any


def belief_graph_from_spec(spec: dict) -> dict:
    """Convert a SPEC dict to a 'perfect' belief graph that should score ~1.0.

    Used by:
      - tests/test_matcher* (calibration: perfect input must score >= 0.9)
      - scripts/verify_phase1.py (smoke check)
      - viz code (renders the ground-truth panel from this)
    """
    bg: dict[str, Any] = {
        "auth": {
            "type": spec["auth"]["type"],
            "scopes_observed": list(spec["auth"]["scopes"]),
        },
        "resources": [],
        "endpoints": [],
    }
    for r in spec["resources"]:
        bg["resources"].append({
            "name": r["name"],
            "fields": [{"name": f["name"], "type": f["type"]} for f in r["fields"]],
            "state_machine": r.get("state_machine"),
        })
    for ep in spec["endpoints"]:
        params: list[dict[str, Any]] = []
        for p in ep.get("query_params", []):
            params.append({"name": p["name"], "type": p["type"], "location": "query"})
        for p in ep.get("path_params", []):
            params.append({"name": p["name"], "type": p["type"], "location": "path"})
        for p in ep.get("body_fields", []):
            params.append({"name": p["name"], "type": p["type"], "location": "body"})
        bg["endpoints"].append({
            "method": ep["method"],
            "path": ep["path"],
            "auth_required": ep.get("auth_required", False),
            "auth_scope": ep.get("auth_scope"),
            "params": params,
            "responses": {k: {"shape": v.get("shape", "unknown")} for k, v in ep["responses"].items()},
        })
    return bg


def build_half_belief_graph(spec: dict) -> dict:
    """Return a partial belief graph: ~half the endpoints, partial details, one resource."""
    perfect = belief_graph_from_spec(spec)
    n_eps = len(perfect["endpoints"]) // 2
    return {
        "auth": {"type": "bearer", "scopes_observed": ["users:read", "docs:read"]},
        "resources": perfect["resources"][:1],
        "endpoints": [
            # strip details: keep only method/path + auth_required for each kept ep
            {"method": e["method"], "path": e["path"], "auth_required": e["auth_required"]}
            for e in perfect["endpoints"][:n_eps]
        ],
    }
