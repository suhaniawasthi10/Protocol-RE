"""Reward calibration tests.

These validate the *shape* of the reward signal (not just correctness):
- monotonicity: more correct info never lowers the score
- saturation: spam doesn't drive reward arbitrarily negative
- gradient: there's signal between 0 and 1, not a step function
"""

from server.matcher import score
from server.spec import SPEC
from tests.helpers import belief_graph_from_spec


def test_random_garbage_scores_zero_with_penalty():
    garbage = {"endpoints": [{"method": "GET", "path": f"/fake_{i}"} for i in range(20)]}
    r = score(garbage, SPEC)
    assert r.total < 0.05
    assert r.false_claims >= 15  # most should register as false


def test_all_endpoints_no_details_scores_moderate():
    bg = {"endpoints": [{"method": e["method"], "path": e["path"]} for e in SPEC["endpoints"]]}
    r = score(bg, SPEC)
    # Should get the full endpoints_discovered (0.30) but ~nothing else.
    # Endpoint_details still gives some credit because absence of fields counts as 0/0=skipped.
    assert 0.25 < r.total < 0.40


def test_monotonicity_endpoints():
    """Adding correct endpoints never decreases score."""
    bg1 = {"endpoints": [{"method": "GET", "path": "/users"}]}
    bg2 = {"endpoints": [
        {"method": "GET", "path": "/users"},
        {"method": "POST", "path": "/users"},
    ]}
    bg3 = {"endpoints": [
        {"method": "GET", "path": "/users"},
        {"method": "POST", "path": "/users"},
        {"method": "GET", "path": "/docs"},
    ]}
    s1 = score(bg1, SPEC).total
    s2 = score(bg2, SPEC).total
    s3 = score(bg3, SPEC).total
    assert s1 <= s2 <= s3


def test_monotonicity_details():
    """Adding correct details to a discovered endpoint never decreases score."""
    bare = {"endpoints": [{"method": "GET", "path": "/users"}]}
    rich = {"endpoints": [{
        "method": "GET", "path": "/users",
        "auth_required": True, "auth_scope": "users:read",
        "params": [{"name": "limit", "type": "int", "location": "query"}],
        "responses": {"200": {"shape": "list<User>"}, "401": {"shape": "error"}},
    }]}
    assert score(rich, SPEC).total >= score(bare, SPEC).total


def test_penalty_saturation():
    """False claims should not drive score below 0."""
    bg = {"endpoints": [{"method": "GET", "path": f"/fake_{i}"} for i in range(200)]}
    r = score(bg, SPEC)
    assert r.total >= 0.0


def test_perfect_graph_scores_close_to_one():
    bg = belief_graph_from_spec(SPEC)
    r = score(bg, SPEC)
    assert r.total > 0.90, f"got {r.total:.3f}"
    assert r.total <= 1.0


def test_gradient_exists_between_zero_and_one():
    """Reward should not be a step function — partial credit must show up."""
    perfect = belief_graph_from_spec(SPEC)

    quarter = {
        "endpoints": perfect["endpoints"][:5],
        "resources": [],
        "auth": {"type": "bearer", "scopes_observed": []},
    }
    half = {
        "endpoints": perfect["endpoints"][:9],
        "resources": perfect["resources"][:1],
        "auth": {"type": "bearer", "scopes_observed": ["users:read", "docs:read"]},
    }
    full = perfect

    s_q = score(quarter, SPEC).total
    s_h = score(half, SPEC).total
    s_f = score(full, SPEC).total

    # All distinct, all in (0, 1], strictly increasing
    assert 0 < s_q < s_h < s_f <= 1.0
    # Gap between successive levels >= 0.05 — there's real signal
    assert (s_h - s_q) > 0.05
    assert (s_f - s_h) > 0.05


def test_false_claim_does_not_outweigh_correct_endpoint():
    """One correct endpoint + one false claim should still be net positive."""
    bg = {"endpoints": [
        {"method": "GET", "path": "/users"},
        {"method": "GET", "path": "/totally_fake"},
    ]}
    r = score(bg, SPEC)
    assert r.total > 0
