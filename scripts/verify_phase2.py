#!/usr/bin/env python
"""Phase 2 verification — drives the env through the real OpenEnv HTTP+WS surface.

Boots the FastAPI app in a background uvicorn, connects via the sync EnvClient
(same path training code will use), and walks through reset -> unauth probe
(expect 401) -> authed probe (expect 200) -> update_model -> finalize.

Set env var SPACE_URL to hit a remote HF Space instead of a local server.
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


def _start_local_server(port: int) -> threading.Thread:
    """Launch uvicorn in a background thread."""
    import uvicorn

    from server.app import app  # noqa: F401 — import to ensure app is built

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to come up
    import httpx
    for _ in range(50):
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5)
            if r.status_code == 200:
                return thread
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("server never came up")


def main() -> int:
    from client import ProtocolOneEnv
    from models import ProtocolOneAction

    space_url = os.environ.get("SPACE_URL")
    if space_url:
        print(f"Using remote SPACE_URL={space_url}")
        base_url = space_url
    else:
        port = 8765
        print(f"Booting local uvicorn on :{port} …")
        _start_local_server(port)
        base_url = f"http://127.0.0.1:{port}"

    env = ProtocolOneEnv(base_url=base_url).sync()
    with env:
        # 1. Reset
        result = env.reset()
        text = result.observation.text or ""
        assert "reverse-engineer" in text.lower() or "reverse engineer" in text.lower(), \
            f"unexpected reset text: {text[:200]!r}"
        assert result.observation.probes_remaining == 80
        print(f"✓ Reset — {len(text)} char initial observation, probes_remaining=80")

        # 2. Unauth probe -> expect 401
        result = env.step(ProtocolOneAction(
            tool="probe",
            args={"method": "GET", "path": "/users"},
        ))
        assert "401" in (result.observation.text or ""), (
            f"expected 401 in unauth probe, got: {result.observation.text[:200]}"
        )
        print("✓ Unauthenticated probe → 401")

        # 3. Authed probe -> expect 200
        result = env.step(ProtocolOneAction(
            tool="probe",
            args={"method": "GET", "path": "/users",
                  "headers": {"Authorization": "Bearer token_full"}},
        ))
        assert "200" in (result.observation.text or ""), (
            f"expected 200 in authed probe, got: {result.observation.text[:200]}"
        )
        print("✓ Authed probe → 200")

        # 4. update_model
        result = env.step(ProtocolOneAction(
            tool="update_model",
            args={"delta": {
                "endpoints": [
                    {"method": "GET", "path": "/users",
                     "auth_required": True, "auth_scope": "users:read"},
                ],
                "auth": {"type": "bearer", "scopes_observed": ["users:read"]},
            }},
        ))
        stats = result.observation.belief_graph_stats
        assert stats.get("endpoints") == 1, f"expected 1 endpoint, got {stats}"
        assert stats.get("auth_scopes_observed") == 1
        print(f"✓ update_model merged — stats={stats}")

        # 5. finalize -> expect reward > 0 but < 0.3 (we only submitted one endpoint)
        result = env.step(ProtocolOneAction(tool="finalize", args={}))
        assert result.done
        assert result.reward is not None
        assert 0.0 < result.reward < 0.3, (
            f"unexpected reward {result.reward} for single-endpoint finalize"
        )
        print(f"✓ Finalized — reward={result.reward:.3f} (expected in (0, 0.3))")

    print("\nPhase 2 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
