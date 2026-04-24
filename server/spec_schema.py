"""Pydantic schema for SPEC validation.

Used at server startup to fail fast if the spec dict is malformed. The matcher
and protocol_server both assume the SPEC validates against this schema.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FieldDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["string", "int", "float", "bool", "timestamp", "list", "object"]
    required: bool = False
    enum: list[str] | None = None
    default: str | int | float | bool | None = None
    max: int | None = None


class ParamDef(FieldDef):
    pass


class Transition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_: str = Field(alias="from")
    to: str
    via: str


class StateMachine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial: str
    states: list[str]
    transitions: list[Transition]


class ResourceDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    fields: list[FieldDef]
    state_machine: StateMachine | None = None


class ResponseDef(BaseModel):
    model_config = ConfigDict(extra="allow")

    shape: str


class EndpointDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str
    auth_required: bool = False
    auth_scope: str | None = None
    query_params: list[ParamDef] = Field(default_factory=list)
    path_params: list[ParamDef] = Field(default_factory=list)
    body_fields: list[FieldDef] = Field(default_factory=list)
    responses: dict[str, dict[str, Any]]
    aliases: list[str] = Field(default_factory=list)


class AuthDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["bearer", "api_key", "basic", "none"]
    scopes: list[str]


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth: AuthDef
    resources: list[ResourceDef]
    endpoints: list[EndpointDef]


def load_and_validate_spec() -> Spec:
    """Validate the SPEC dict. Raises pydantic.ValidationError on malformed input."""
    from server.spec import SPEC
    return Spec.model_validate(SPEC)
