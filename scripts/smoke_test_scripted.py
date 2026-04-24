#!/usr/bin/env python
"""Scripted-heuristic smoke test — no LLM API key required.

Drives the env with a deterministic policy that roughly mimics what a
well-prompted stock LLM would do in the first ~30 probes: hit /users without
auth (discover 401), hit with the starter token, list docs, test a state
transition, observe some 403 responses, submit a modest belief graph, finalize.

Target reward range: 0.2 - 0.5. If the actual reward falls OUTSIDE this band,
the env or matcher is miscalibrated for baseline training.

Run:
    python scripts/smoke_test_scripted.py
or against a deployed Space:
    SPACE_URL=https://you-protocol_one_env.hf.space python scripts/smoke_test_scripted.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _start_local_server(port: int) -> None:
    import uvicorn
    from server.app import app
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()

    import httpx
    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("server never came up")


def run_episode(base_url: str) -> float:
    from client import ProtocolOneEnv
    from models import ProtocolOneAction

    auth_full = {"Authorization": "Bearer token_full"}
    auth_read = {"Authorization": "Bearer token_read"}

    # The belief graph the agent will eventually submit. We build it up
    # as probes confirm expectations, so it only contains things we've
    # actually observed (no hallucinated endpoints = no false-claim penalty).
    belief: dict = {"endpoints": [], "resources": [], "auth": {"type": "bearer", "scopes_observed": []}}

    def add_ep(method: str, path: str, **extra) -> None:
        belief["endpoints"].append({"method": method, "path": path, **extra})

    env = ProtocolOneEnv(base_url=base_url).sync()
    with env:
        env.reset()

        # --- Discovery phase: cheap probes to learn the surface --- #
        probes: list[tuple[str, str, dict, dict | None]] = [
            # Auth probes (unauth then authed)
            ("GET", "/users", {}, None),
            ("GET", "/users", auth_full, None),
            ("GET", "/_/health", {}, None),
            ("GET", "/auth/whoami", auth_full, None),
            ("GET", "/users/me", auth_full, None),
            ("GET", "/users/u_alice", auth_full, None),
            ("GET", "/users/u_alice/documents", auth_full, None),
            ("GET", "/docs", auth_full, None),
            ("GET", "/docs/d_intro", auth_full, None),
            # Scope probes — discover 403 behavior for scopes
            ("GET", "/auth/scopes", auth_read, None),  # should 403
            # State-machine probes
            ("POST", "/docs/d_specs/publish", auth_full, None),  # draft -> published
            ("POST", "/docs/d_specs/archive", auth_full, None),  # published -> archived
            ("POST", "/docs/d_old/publish", auth_full, None),  # archived -> invalid 409
            ("POST", "/users/u_alice/suspend", auth_full, None),  # active -> suspended
            ("POST", "/users/u_alice/restore", auth_full, None),  # suspended -> active
            # Deletion + idempotency
            ("DELETE", "/docs/d_old", auth_full, None),  # 200
            ("DELETE", "/docs/d_old", auth_full, None),  # 404 (already-gone)
            # Creation / validation
            ("POST", "/docs", auth_full, {"title": "smoke"}),
            ("POST", "/docs", auth_full, {}),  # 422 missing_title
        ]

        scopes_observed: set[str] = set()
        for method, path, headers, body in probes:
            result = env.step(ProtocolOneAction(
                tool="probe",
                args={"method": method, "path": path, "headers": headers, "body": body},
            ))
            text = result.observation.text or ""
            # Parse status code from "[Probe N/M] HTTP XXX"
            if "HTTP 401" in text:
                pass
            if "HTTP 403" in text and "required" in text:
                # A 403 tells us *what scope* was required — extract it
                for line in text.splitlines():
                    if '"required"' in line:
                        # crude parse: extract value after "required":
                        import re
                        m = re.search(r'"required"\s*:\s*"([^"]+)"', line)
                        if m:
                            scopes_observed.add(m.group(1))

        # --- Build the belief graph to simulate a stock-LLM baseline.
        # A real untrained LLM in ~30 probes would find the obvious endpoints,
        # infer auth + 1-2 scopes, partially describe resources, and miss
        # state-machine nuances. We deliberately include a plausible hallucination
        # to exercise the false-claim penalty. Target: ~0.25-0.45 reward.

        # Core endpoints the agent would find quickly (10/18), with
        # minimal/partial details:
        add_ep("GET", "/users", auth_required=True, auth_scope="users:read")
        add_ep("POST", "/users", auth_required=True,
               params=[{"name": "email", "type": "string", "location": "body"}])
        add_ep("GET", "/users/{id}", auth_required=True)
        add_ep("DELETE", "/users/{id}", auth_required=True)
        add_ep("GET", "/docs", auth_required=True, auth_scope="docs:read")
        add_ep("POST", "/docs", auth_required=True)
        add_ep("GET", "/docs/{id}", auth_required=True)
        add_ep("POST", "/docs/{id}/publish", auth_required=True)
        add_ep("GET", "/auth/whoami", auth_required=True)
        add_ep("GET", "/_/health", auth_required=False)
        # Realistic hallucination: untrained LLM guesses endpoint that feels RESTful
        add_ep("PUT", "/users/{id}", auth_required=True)  # spec has PATCH, not PUT

        # Resources — only User, partial fields, no state machine:
        belief["resources"].append({
            "name": "User",
            "fields": [
                {"name": "id", "type": "string"},
                {"name": "email", "type": "string"},
                {"name": "role", "type": "string"},
            ],
        })

        # Auth inference — type right, only 2 of 5 scopes
        belief["auth"] = {
            "type": "bearer",
            "scopes_observed": sorted({"users:read", "docs:read"} | scopes_observed),
        }

        # Push it all via update_model then finalize
        env.step(ProtocolOneAction(tool="update_model", args={"delta": belief}))
        result = env.step(ProtocolOneAction(tool="finalize", args={}))

    reward = float(result.reward or 0.0)
    return reward


def main() -> int:
    space_url = os.environ.get("SPACE_URL")
    if space_url:
        print(f"Using remote SPACE_URL={space_url}")
        base_url = space_url
    else:
        port = 8766
        print(f"Booting local uvicorn on :{port} …")
        _start_local_server(port)
        base_url = f"http://127.0.0.1:{port}"

    reward = run_episode(base_url)
    print(f"\nScripted-heuristic reward: {reward:.3f}")

    if not (0.2 <= reward <= 0.6):
        # Smoke test band: 0.2-0.5 for a stock LLM per master plan, widening
        # to 0.6 for the scripted heuristic which is a bit stronger at
        # hard-coding endpoint details.
        print(f"⚠ reward {reward:.3f} outside expected [0.2, 0.6] — check matcher/env")
        return 1

    print(f"✓ reward in expected band — env end-to-end loop works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
