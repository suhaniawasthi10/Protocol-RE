# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Action/Observation dataclasses for the Protocol One environment.

Design note — single action with a tool discriminator:
  We use ONE pydantic action (`ProtocolOneAction`) carrying a
  `tool: Literal["probe", "update_model", "finalize"]` plus an `args` dict.
  TRL's environment_factory mode will translate LLM tool calls into this
  single shape, and the server-side Environment dispatches on `tool`.

  This is cleaner than a nested union for JSON-over-WebSocket — pydantic's
  discriminator support makes validation failures explicit instead of
  silently accepting an empty action.
"""

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ProtocolOneAction(Action):
    """Agent tool invocation.

    tool: which tool is being called
    args: tool-specific arguments. For `probe`:
            {"method": str, "path": str, "headers": dict, "body": dict | None}
          For `update_model`:
            {"delta": {"endpoints": [...], "resources": [...], "auth": {...}}}
          For `finalize`:
            {"final_belief_graph": dict | None}  (optional)
    """

    tool: Literal["probe", "update_model", "finalize"] = Field(
        ..., description="Which tool is being invoked: probe | update_model | finalize",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific arguments. See class docstring for each tool's shape.",
    )


class ProtocolOneObservation(Observation):
    """Observation returned after every env.step().

    text: primary human-readable message shown to the model (probe response,
          update confirmation, final score summary, or error).
    probes_used / probes_remaining: budget tracking — model should
          tighten its exploration when remaining is low.
    belief_graph_stats: lightweight summary so the model can see what it has
          stored without re-reading the whole belief graph.
    done: episode has ended.
    (reward is inherited from Observation; only non-None at terminal step.)
    """

    text: str = Field(default="", description="Primary text shown to the model")
    probes_used: int = Field(default=0, description="Number of probes used so far")
    probes_remaining: int = Field(default=0, description="Remaining probe budget")
    belief_graph_stats: dict[str, int] = Field(
        default_factory=dict,
        description="{endpoints, resources, auth_scopes_observed}",
    )
