"""Integration tests for the mock protocol server.

Purpose: pin the contract between spec.py and the FastAPI routes. If these
pass, every response code the matcher cares about is actually producible
from the real server.

Organization follows the spec's endpoint order.
"""

from fastapi.testclient import TestClient
import pytest

from server.protocol_server import create_server


AUTH_FULL = {"Authorization": "Bearer token_full"}
AUTH_READ = {"Authorization": "Bearer token_read"}
AUTH_WRITE = {"Authorization": "Bearer token_write"}
AUTH_ADMIN = {"Authorization": "Bearer token_admin"}
AUTH_JUNK = {"Authorization": "Bearer nonexistent"}
AUTH_MALFORMED = {"Authorization": "Basic abc"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_server().app)


# ---------------------------------------------------------------------------
# Health (no auth)
# ---------------------------------------------------------------------------

def test_health_no_auth(client):
    r = client.get("/_/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_missing_auth_returns_401(client):
    r = client.get("/users")
    assert r.status_code == 401


def test_malformed_auth_returns_401(client):
    r = client.get("/users", headers=AUTH_MALFORMED)
    assert r.status_code == 401


def test_invalid_token_returns_401(client):
    r = client.get("/users", headers=AUTH_JUNK)
    assert r.status_code == 401


def test_insufficient_scope_returns_403(client):
    # token_write does NOT have users:read
    r = client.get("/users", headers=AUTH_WRITE)
    assert r.status_code == 403
    assert r.json()["detail"]["error"] == "insufficient_scope"


def test_admin_only_endpoint_rejects_non_admin(client):
    r = client.get("/auth/scopes", headers=AUTH_READ)
    assert r.status_code == 403


def test_admin_endpoint_succeeds_with_admin_scope(client):
    r = client.get("/auth/scopes", headers=AUTH_ADMIN)
    assert r.status_code == 200
    assert "admin" in r.json()["data"]


# ---------------------------------------------------------------------------
# Users: list / get / me
# ---------------------------------------------------------------------------

def test_list_users_happy_path(client):
    r = client.get("/users", headers=AUTH_FULL)
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 3
    assert {u["email"] for u in data} == {
        "alice@example.com", "bob@example.com", "carol@example.com"
    }


def test_list_users_role_filter(client):
    r = client.get("/users?role=admin", headers=AUTH_FULL)
    assert r.status_code == 200
    assert all(u["role"] == "admin" for u in r.json()["data"])


def test_list_users_invalid_role_filter(client):
    r = client.get("/users?role=wizard", headers=AUTH_FULL)
    assert r.status_code == 422


def test_list_users_limit_too_high(client):
    r = client.get("/users?limit=999", headers=AUTH_FULL)
    assert r.status_code == 422


def test_get_user_by_id(client):
    r = client.get("/users/u_alice", headers=AUTH_FULL)
    assert r.status_code == 200
    assert r.json()["email"] == "alice@example.com"


def test_get_user_not_found(client):
    r = client.get("/users/does_not_exist", headers=AUTH_FULL)
    assert r.status_code == 404


def test_users_me_alias(client):
    r = client.get("/users/me", headers=AUTH_FULL)
    assert r.status_code == 200
    # token_full is tied to alice's id in seed
    assert r.json()["email"] == "alice@example.com"


# ---------------------------------------------------------------------------
# Users: create / update / delete
# ---------------------------------------------------------------------------

def test_create_user_happy_path(client):
    r = client.post("/users", json={"email": "new@example.com"}, headers=AUTH_FULL)
    assert r.status_code == 201
    assert r.json()["email"] == "new@example.com"
    assert r.json()["role"] == "member"  # default


def test_create_user_rejects_duplicate_email(client):
    r = client.post("/users", json={"email": "alice@example.com"}, headers=AUTH_FULL)
    assert r.status_code == 409


def test_create_user_invalid_email(client):
    r = client.post("/users", json={"email": "not-an-email"}, headers=AUTH_FULL)
    assert r.status_code == 422


def test_create_user_invalid_role(client):
    r = client.post("/users", json={"email": "x@example.com", "role": "overlord"},
                    headers=AUTH_FULL)
    assert r.status_code == 422


def test_update_user(client):
    r = client.patch("/users/u_bob", json={"role": "admin"}, headers=AUTH_FULL)
    assert r.status_code == 200
    assert r.json()["role"] == "admin"


def test_update_user_invalid_role(client):
    r = client.patch("/users/u_bob", json={"role": "wizard"}, headers=AUTH_FULL)
    assert r.status_code == 422


def test_update_user_not_found(client):
    r = client.patch("/users/missing", json={"role": "admin"}, headers=AUTH_FULL)
    assert r.status_code == 404


def test_delete_user_idempotent_gone(client):
    r1 = client.delete("/users/u_bob", headers=AUTH_FULL)
    assert r1.status_code == 200
    r2 = client.delete("/users/u_bob", headers=AUTH_FULL)
    assert r2.status_code == 410
    # GET on deleted user -> 404
    r3 = client.get("/users/u_bob", headers=AUTH_FULL)
    assert r3.status_code == 404


# ---------------------------------------------------------------------------
# Users: state machine
# ---------------------------------------------------------------------------

def test_suspend_active_user(client):
    r = client.post("/users/u_alice/suspend", headers=AUTH_FULL)
    assert r.status_code == 200
    assert r.json()["status"] == "suspended"


def test_suspend_already_suspended_user_is_409(client):
    r = client.post("/users/u_carol/suspend", headers=AUTH_FULL)
    assert r.status_code == 409


def test_restore_suspended_user(client):
    r = client.post("/users/u_carol/restore", headers=AUTH_FULL)
    assert r.status_code == 200
    assert r.json()["status"] == "active"


def test_restore_active_user_is_409(client):
    r = client.post("/users/u_alice/restore", headers=AUTH_FULL)
    assert r.status_code == 409


# ---------------------------------------------------------------------------
# Documents: list / get / create
# ---------------------------------------------------------------------------

def test_list_documents(client):
    r = client.get("/docs", headers=AUTH_FULL)
    assert r.status_code == 200
    assert len(r.json()["data"]) == 3


def test_list_documents_by_state(client):
    r = client.get("/docs?state=draft", headers=AUTH_FULL)
    assert r.status_code == 200
    assert all(d["state"] == "draft" for d in r.json()["data"])


def test_list_user_documents(client):
    r = client.get("/users/u_alice/documents", headers=AUTH_FULL)
    assert r.status_code == 200
    # alice owns two seeded docs
    owners = {d["owner_id"] for d in r.json()["data"]}
    assert owners == {"u_alice"}


def test_list_user_documents_unknown_user(client):
    r = client.get("/users/unknown/documents", headers=AUTH_FULL)
    assert r.status_code == 404


def test_create_document_happy_path(client):
    r = client.post("/docs", json={"title": "new doc"}, headers=AUTH_FULL)
    assert r.status_code == 201
    assert r.json()["state"] == "draft"
    assert r.json()["owner_id"] == "u_alice"  # token_full -> alice


def test_create_document_missing_title(client):
    r = client.post("/docs", json={"body": "no title"}, headers=AUTH_FULL)
    assert r.status_code == 422


def test_get_document(client):
    r = client.get("/docs/d_intro", headers=AUTH_FULL)
    assert r.status_code == 200


def test_get_document_not_found(client):
    r = client.get("/docs/missing", headers=AUTH_FULL)
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Documents: update / delete / state machine
# ---------------------------------------------------------------------------

def test_update_draft_document(client):
    r = client.patch("/docs/d_specs", json={"title": "updated"}, headers=AUTH_FULL)
    assert r.status_code == 200
    assert r.json()["title"] == "updated"


def test_update_published_document_is_409(client):
    r = client.patch("/docs/d_intro", json={"title": "nope"}, headers=AUTH_FULL)
    assert r.status_code == 409


def test_publish_draft_document(client):
    r = client.post("/docs/d_specs/publish", headers=AUTH_FULL)
    assert r.status_code == 200
    assert r.json()["state"] == "published"


def test_publish_non_draft_is_409(client):
    r = client.post("/docs/d_intro/publish", headers=AUTH_FULL)
    assert r.status_code == 409


def test_archive_published_document(client):
    r = client.post("/docs/d_intro/archive", headers=AUTH_FULL)
    assert r.status_code == 200
    assert r.json()["state"] == "archived"


def test_archive_draft_is_409(client):
    r = client.post("/docs/d_specs/archive", headers=AUTH_FULL)
    assert r.status_code == 409


def test_archive_already_archived_is_409(client):
    r = client.post("/docs/d_old/archive", headers=AUTH_FULL)
    assert r.status_code == 409


def test_delete_document(client):
    r = client.delete("/docs/d_specs", headers=AUTH_FULL)
    assert r.status_code == 200
    r2 = client.get("/docs/d_specs", headers=AUTH_FULL)
    assert r2.status_code == 404


# ---------------------------------------------------------------------------
# /auth/whoami and /auth/scopes
# ---------------------------------------------------------------------------

def test_whoami(client):
    r = client.get("/auth/whoami", headers=AUTH_FULL)
    assert r.status_code == 200
    assert r.json()["email"] == "alice@example.com"


def test_whoami_wrong_scope(client):
    r = client.get("/auth/whoami", headers=AUTH_WRITE)  # write has no users:read
    assert r.status_code == 403


# ---------------------------------------------------------------------------
# Sanity: every endpoint declared in SPEC has a live route
# ---------------------------------------------------------------------------

def test_every_spec_endpoint_has_a_route(client):
    from server.spec import SPEC
    # For each spec endpoint, hit it (unauthenticated) and ensure we do NOT
    # get 404. Anything but 404 (401/403/405/422/etc.) means the path exists.
    for ep in SPEC["endpoints"]:
        path = ep["path"].format(id="u_alice")
        # Substitute remaining placeholders with alice's id or a seeded doc id
        path = path.replace("{id}", "u_alice")
        r = client.request(ep["method"], path)
        assert r.status_code != 404, (
            f"spec endpoint {ep['method']} {ep['path']} has no live route; got 404"
        )
