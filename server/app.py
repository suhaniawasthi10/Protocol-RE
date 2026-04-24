# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Protocol One environment.

Exposed endpoints (via openenv.core.env_server.http_server):
    POST /reset, POST /step, GET /state, GET /schema, GET /health,
    WS /ws (persistent session), GET /web (when ENABLE_WEB_INTERFACE=true).

max_concurrent_envs=64 paired with SUPPORTS_CONCURRENT_SESSIONS=True on the
Environment class is what makes GRPO rollout parallelism work — each rollout
gets its own Environment instance with isolated mock-server state.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with `uv sync` or "
        "`pip install openenv-core`"
    ) from e

try:
    from ..models import ProtocolOneAction, ProtocolOneObservation
    from .protocol_one_env_environment import ProtocolOneEnvironment
except ImportError:
    # When launched as `python -m server.app` or imported as `server.app`,
    # the parent package isn't on sys.path so the relative imports above
    # raise `attempted relative import beyond top-level package`.
    from models import ProtocolOneAction, ProtocolOneObservation  # type: ignore
    from server.protocol_one_env_environment import ProtocolOneEnvironment  # type: ignore


app = create_app(
    ProtocolOneEnvironment,
    ProtocolOneAction,
    ProtocolOneObservation,
    env_name="protocol_one_env",
    max_concurrent_envs=64,  # GRPO needs N parallel rollouts; 64 comfortably covers up to generation_batch_size=64
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
