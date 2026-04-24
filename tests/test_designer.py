"""Tests for the Designer mutation module.

The designer is scripted (not RL-trained). These tests verify:
  - Default config (disabled via env vars) never mutates.
  - When enabled, each of the 5 mutation types can fire.
  - Mutations operate on a copy — the base SPEC is never modified in place.
  - Mutated specs still validate against the pydantic schema.
"""

import copy

import pytest

from server.designer import Designer
from server.spec import SPEC
from server.spec_schema import Spec


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MUTATION_PROBABILITY", raising=False)
    monkeypatch.delenv("MUTATION_START_EPISODE", raising=False)
    d = Designer(SPEC)
    assert d.mutation_probability == 0.0
    assert d.mutation_start_episode == 50
    # Run many episodes — no mutation should fire
    for _ in range(120):
        out = d.maybe_mutate()
        assert d.last_mutation_log is None
        assert out == SPEC  # identical (and a copy)


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("MUTATION_PROBABILITY", "1.0")
    monkeypatch.setenv("MUTATION_START_EPISODE", "0")
    d = Designer(SPEC)
    assert d.mutation_probability == 1.0
    assert d.mutation_start_episode == 0


def test_cooldown_respected():
    d = Designer(SPEC, mutation_start_episode=10, mutation_probability=1.0)
    for _ in range(9):
        d.maybe_mutate()
        assert d.last_mutation_log is None
    # 10th call is still under cooldown (episodes_seen == 10 < 10 is False, so it DOES fire)
    # Actually: episodes_seen increments first, so we cross the threshold at the 10th call.
    d.maybe_mutate()
    # From here, should mutate
    assert d.last_mutation_log is not None


def test_base_spec_never_modified():
    before = copy.deepcopy(SPEC)
    d = Designer(SPEC, mutation_start_episode=0, mutation_probability=1.0)
    for _ in range(30):
        d.maybe_mutate()
    assert SPEC == before, "base SPEC was mutated in place"


@pytest.mark.parametrize("mutation", Designer.MUTATION_TYPES)
def test_each_mutation_type_can_apply(mutation):
    d = Designer(SPEC, mutation_start_episode=0, mutation_probability=1.0)
    fn = getattr(d, f"_mutate_{mutation}")
    spec_copy = copy.deepcopy(SPEC)
    log = fn(spec_copy)
    assert isinstance(log, dict)
    assert "applied" in log
    assert log["applied"] is True, f"{mutation} did not apply: {log}"


def test_rename_field_changes_field_name():
    d = Designer(SPEC, mutation_start_episode=0, mutation_probability=1.0)
    spec_copy = copy.deepcopy(SPEC)
    log = d._mutate_rename_field(spec_copy)
    assert log["applied"]
    resource_name = log["resource"]
    old_name = log["old"]
    new_name = log["new"]
    resource = next(r for r in spec_copy["resources"] if r["name"] == resource_name)
    field_names = {f["name"] for f in resource["fields"]}
    assert new_name in field_names
    assert old_name not in field_names


def test_deprecate_endpoint_removes_entry():
    d = Designer(SPEC, mutation_start_episode=0, mutation_probability=1.0)
    spec_copy = copy.deepcopy(SPEC)
    log = d._mutate_deprecate_endpoint(spec_copy)
    assert log["applied"]
    removed_id = log["endpoint_id"]
    assert removed_id not in {e["id"] for e in spec_copy["endpoints"]}
    assert len(spec_copy["endpoints"]) == len(SPEC["endpoints"]) - 1


def test_tighten_auth_scope_sets_admin():
    d = Designer(SPEC, mutation_start_episode=0, mutation_probability=1.0)
    spec_copy = copy.deepcopy(SPEC)
    log = d._mutate_tighten_auth_scope(spec_copy)
    assert log["applied"]
    ep = next(e for e in spec_copy["endpoints"] if e["id"] == log["endpoint_id"])
    assert ep["auth_scope"] == "admin"


def test_shift_state_transition_adds_transition():
    d = Designer(SPEC, mutation_start_episode=0, mutation_probability=1.0)
    spec_copy = copy.deepcopy(SPEC)
    log = d._mutate_shift_state_transition(spec_copy)
    assert log["applied"]
    resource = next(r for r in spec_copy["resources"] if r["name"] == log["resource"])
    transitions = {(t["from"], t["to"]) for t in resource["state_machine"]["transitions"]}
    nt = log["new_transition"]
    assert (nt["from"], nt["to"]) in transitions


def test_mutated_spec_still_validates_against_schema():
    # Skip swap_error_code for validation check — after a swap the schema is still
    # valid (just different), but let's spot-check the others.
    for _ in range(25):
        d = Designer(SPEC, mutation_start_episode=0, mutation_probability=1.0)
        spec_copy = d.maybe_mutate()
        Spec.model_validate(spec_copy)  # raises if shape broke


def test_mutation_log_populated_when_firing():
    d = Designer(SPEC, mutation_start_episode=0, mutation_probability=1.0, seed=0)
    d.maybe_mutate()
    assert d.last_mutation_log is not None
    assert "type" in d.last_mutation_log
    assert d.last_mutation_log["type"] in Designer.MUTATION_TYPES
