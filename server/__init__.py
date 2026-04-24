# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Protocol One Env — server-side components (hidden from the agent).

Modules:
    spec: ground-truth protocol description
    spec_schema: pydantic validator for spec
    matcher: reward function (pure, deterministic)
    protocol_server: FastAPI mock exposing the 18 endpoints
    designer: between-episode mutator (scripted, disabled by default)
    protocol_one_env_environment: the OpenEnv Environment subclass
    app: the ASGI app wired up by openenv.core.http_server
"""

from .protocol_one_env_environment import ProtocolOneEnvironment

__all__ = ["ProtocolOneEnvironment"]
