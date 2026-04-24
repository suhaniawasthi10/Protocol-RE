"""Matcher unit tests — written BEFORE the matcher implementation.

These pin down the contract: empty -> ~0, perfect -> >=0.9, half -> ~middle,
false claims penalised but bounded. If you change the matcher, run these and
the calibration suite both.
"""

import pytest

from server.matcher import score
from server.spec import SPEC
from tests.helpers import belief_graph_from_spec, build_half_belief_graph


def test_none_belief_graph_is_safe_and_zero():
    r = score(None, SPEC)
    assert r.total == 0.0


def test_empty_dict_belief_graph_scores_near_zero():
    r = score({}, SPEC)
    assert r.total < 0.05


def test_garbage_input_does_not_crash():
    # Strings, ints, lists at the top level — matcher must coerce / ignore.
    for garbage in ["not a dict", 42, ["lol"], {"endpoints": "wat"}]:
        r = score(garbage, SPEC)
        assert r.total <= 0.1


def test_one_correct_endpoint_scores_low_positive():
    bg = {"endpoints": [{"method": "GET", "path": "/users", "auth_required": True}]}
    r = score(bg, SPEC)
    # 1/18 endpoints (=0.019) + 1/3 details on that endpoint (=0.083) ≈ 0.10
    assert 0.005 < r.total < 0.15
    assert r.endpoints_found == 1


def test_partial_belief_graph_gets_partial_credit():
    bg = {"endpoints": [
        {"method": "GET", "path": "/users"},
        {"method": "POST", "path": "/users"},
        {"method": "GET", "path": "/docs"},
    ]}
    r = score(bg, SPEC)
    # 3/18 endpoints, no other components
    assert 0.04 < r.total < 0.18
    assert r.endpoints_found == 3


def test_half_correct_belief_graph_scores_about_half():
    bg = build_half_belief_graph(SPEC)
    r = score(bg, SPEC)
    assert 0.20 < r.total < 0.55, f"half-graph scored {r.total:.3f}, expected ~0.2-0.55"


def test_perfect_belief_graph_scores_above_point_nine():
    bg = belief_graph_from_spec(SPEC)
    r = score(bg, SPEC)
    assert r.total > 0.90, f"perfect scored {r.total:.3f}"
    assert r.endpoints_found == len(SPEC["endpoints"])
    assert r.false_claims == 0


def test_false_claims_are_penalised():
    bg = belief_graph_from_spec(SPEC)
    bg["endpoints"].append({"method": "GET", "path": "/admin/fake"})
    bg["endpoints"].append({"method": "DELETE", "path": "/wat"})
    r = score(bg, SPEC)
    assert r.false_claims == 2
    perfect = score(belief_graph_from_spec(SPEC), SPEC)
    assert r.total < perfect.total


def test_path_normalization_id_placeholders():
    """{user_id} should match {id}."""
    bg = {"endpoints": [
        {"method": "GET", "path": "/users/{user_id}"},
        {"method": "PATCH", "path": "/users/{userId}"},
    ]}
    r = score(bg, SPEC)
    assert r.endpoints_found == 2  # both should normalize to /users/{id}


def test_path_normalization_trailing_slash_and_case():
    bg = {"endpoints": [
        {"method": "GET", "path": "/Users/"},
        {"method": "POST", "path": "users"},  # no leading slash
    ]}
    r = score(bg, SPEC)
    assert r.endpoints_found == 2


def test_auth_inference_scored():
    bg = {"auth": {"type": "bearer", "scopes_observed": list(SPEC["auth"]["scopes"])}}
    r = score(bg, SPEC)
    # Auth component is up to 0.10 of total; perfect auth alone should land >=0.06
    assert r.breakdown["auth"] > 0.6
    assert r.total >= 0.06


def test_resource_fields_scored():
    bg = {"resources": [
        {"name": "User", "fields": [
            {"name": "id"}, {"name": "email"}, {"name": "role"},
            {"name": "status"}, {"name": "created_at"}
        ]}
    ]}
    r = score(bg, SPEC)
    # 1 of 2 resources, all fields correct -> 0.5 of resources component (0.15)
    assert r.breakdown["resources"] >= 0.45
    assert r.breakdown["resources"] <= 0.55


def test_state_machine_transitions_scored():
    bg = {"resources": [
        {
            "name": "Document",
            "fields": [{"name": "id"}, {"name": "title"}, {"name": "state"}],
            "state_machine": {
                "states": ["draft", "published", "archived"],
                "transitions": [
                    {"from": "draft", "to": "published"},
                    {"from": "published", "to": "archived"},
                ],
            },
        }
    ]}
    r = score(bg, SPEC)
    # 2 of 4 transitions across both resources -> 0.5 of state_machines component
    assert r.breakdown["state_machines"] >= 0.45
    assert r.breakdown["state_machines"] <= 0.55


def test_total_is_clipped_to_zero_one():
    # Negative values from penalty should never escape [0, 1]
    bg = {"endpoints": [{"method": "GET", "path": f"/fake_{i}"} for i in range(50)]}
    r = score(bg, SPEC)
    assert 0.0 <= r.total <= 1.0


def test_duplicate_endpoint_claims_are_deduplicated():
    bg = {"endpoints": [
        {"method": "GET", "path": "/users"},
        {"method": "GET", "path": "/users"},
        {"method": "GET", "path": "/users/"},  # normalizes to same
    ]}
    r = score(bg, SPEC)
    assert r.endpoints_found == 1


def test_method_case_insensitive():
    bg = {"endpoints": [{"method": "get", "path": "/users"}]}
    r = score(bg, SPEC)
    assert r.endpoints_found == 1


def test_breakdown_keys_present():
    r = score(belief_graph_from_spec(SPEC), SPEC)
    expected_keys = {"endpoints_discovered", "endpoint_details", "resources",
                     "state_machines", "auth", "penalty"}
    assert set(r.breakdown.keys()) >= expected_keys


def test_determinism():
    bg = belief_graph_from_spec(SPEC)
    s1 = score(bg, SPEC).total
    s2 = score(bg, SPEC).total
    s3 = score(bg, SPEC).total
    assert s1 == s2 == s3


@pytest.mark.parametrize("dropped", [
    {"endpoints": [], "resources": [], "auth": {}},  # all empty
    {"endpoints": []},                                  # missing keys
    {"resources": [], "auth": {}},                     # missing endpoints
])
def test_missing_components_safe(dropped):
    r = score(dropped, SPEC)
    assert 0.0 <= r.total <= 1.0
