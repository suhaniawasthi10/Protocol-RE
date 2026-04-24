#!/usr/bin/env python
"""Phase 1 verification. Run this to prove the spec/matcher/server triad is consistent.

Expected output (all green):
  ✓ Spec validates
  ✓ Server boots
  ✓ Empty belief graph    : 0.000
  ✓ Perfect belief graph  : 0.9xx
  ✓ Half belief graph     : 0.2xx - 0.5xx
  ✓ Garbage gets penalised: false_claims >= 15
  Phase 1 verified.
"""

from __future__ import annotations

import os
import sys

# Allow running as a bare script from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from server.matcher import score  # noqa: E402
from server.protocol_server import create_server  # noqa: E402
from server.spec import SPEC  # noqa: E402
from server.spec_schema import load_and_validate_spec  # noqa: E402
from tests.helpers import belief_graph_from_spec, build_half_belief_graph  # noqa: E402


def main() -> int:
    # 1. Spec validates
    load_and_validate_spec()
    print("✓ Spec validates")

    # 2. Server boots
    srv = create_server()
    c = TestClient(srv.app)
    r = c.get("/_/health")
    assert r.status_code == 200, r.text
    r = c.get("/users", headers={"Authorization": "Bearer token_full"})
    assert r.status_code == 200, r.text
    print("✓ Server boots and serves authed traffic")

    # 3. Empty belief graph near zero
    r_empty = score({}, SPEC)
    assert r_empty.total < 0.05, f"empty scored {r_empty.total}"
    print(f"✓ Empty belief graph    : {r_empty.total:.3f}")

    # 4. Perfect belief graph near one
    r_perfect = score(belief_graph_from_spec(SPEC), SPEC)
    assert r_perfect.total > 0.9, f"perfect scored {r_perfect.total}"
    print(f"✓ Perfect belief graph  : {r_perfect.total:.3f}")

    # 5. Half-ish belief graph gets partial credit
    r_half = score(build_half_belief_graph(SPEC), SPEC)
    assert 0.20 < r_half.total < 0.55, f"half scored {r_half.total}"
    print(f"✓ Half belief graph     : {r_half.total:.3f}")

    # 6. Garbage is penalised
    garbage = {"endpoints": [{"method": "GET", "path": f"/fake_{i}"} for i in range(20)]}
    r_g = score(garbage, SPEC)
    assert r_g.false_claims >= 15, f"only {r_g.false_claims} false claims detected"
    assert r_g.total < 0.05, f"garbage scored {r_g.total}"
    print(f"✓ Garbage penalised     : {r_g.false_claims} false claims, total {r_g.total:.3f}")

    # 7. Spec has exactly 18 endpoints
    assert len(SPEC["endpoints"]) == 18, f"spec has {len(SPEC['endpoints'])} endpoints, not 18"
    print(f"✓ 18 endpoints in spec")

    print("\nPhase 1 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
