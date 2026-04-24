# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client-side handle for the Protocol One environment.

Training code (TRL on Colab / HF compute) imports this module and instantiates
a ProtocolOneEnv pointed at the deployed HF Space URL. The client handles the
WebSocket protocol; we only need to implement action serialization and
observation parsing.
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ProtocolOneAction, ProtocolOneObservation
except ImportError:
    # Allow `import client` when protocol_one_env is on sys.path directly.
    from models import ProtocolOneAction, ProtocolOneObservation  # type: ignore


class ProtocolOneEnv(EnvClient[ProtocolOneAction, ProtocolOneObservation, State]):
    """Client for the Protocol One environment.

    Example (sync wrapper):
        >>> from protocol_one_env.client import ProtocolOneEnv
        >>> from protocol_one_env.models import ProtocolOneAction
        >>>
        >>> env = ProtocolOneEnv(base_url="http://localhost:8000").sync()
        >>> with env:
        ...     result = env.reset()
        ...     result = env.step(ProtocolOneAction(
        ...         tool="probe",
        ...         args={"method": "GET", "path": "/users",
        ...               "headers": {"Authorization": "Bearer token_full"}}
        ...     ))
    """

    def _step_payload(self, action: ProtocolOneAction) -> Dict[str, Any]:
        return {"tool": action.tool, "args": action.args}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ProtocolOneObservation]:
        obs_data = payload.get("observation", {}) or {}
        observation = ProtocolOneObservation(
            text=obs_data.get("text", ""),
            probes_used=obs_data.get("probes_used", 0),
            probes_remaining=obs_data.get("probes_remaining", 0),
            belief_graph_stats=obs_data.get("belief_graph_stats", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
