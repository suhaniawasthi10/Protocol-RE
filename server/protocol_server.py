"""Mock HTTP API exposed to the agent through the probe() tool.

Implements the 18 endpoints described in spec.py. The agent only ever sees
this server's HTTP responses — never spec.py, never the in-memory state
dicts. The matcher reads spec.py separately to score the agent's belief
graph against ground truth.

A new MockProtocolServer is created at the start of every episode, so
state never leaks between rollouts.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from server.spec import INITIAL_TOKEN, SPEC, TOKENS  # noqa: F401  (INITIAL_TOKEN re-exported)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    # Frozen-ish timestamp for determinism within an episode. Real real time
    # would still differ between rollouts but it has no effect on reward.
    return 1700000000.0


def _seed_users(store: dict[str, dict]) -> str:
    """Seed 3 users; return the id that token_full's /users/me should resolve to."""
    seeds = [
        {"email": "alice@example.com", "role": "admin", "status": "active"},
        {"email": "bob@example.com", "role": "member", "status": "active"},
        {"email": "carol@example.com", "role": "member", "status": "suspended"},
    ]
    # Use deterministic ids so successive episodes look the same
    ids = ["u_alice", "u_bob", "u_carol"]
    me_id = ids[0]
    for uid, payload in zip(ids, seeds):
        store[uid] = {
            "id": uid,
            "email": payload["email"],
            "role": payload["role"],
            "status": payload["status"],
            "created_at": _now(),
        }
    return me_id


def _seed_documents(store: dict[str, dict]) -> None:
    seeds = [
        {"id": "d_intro", "title": "Welcome", "body": "hello world",
         "owner_id": "u_alice", "state": "published"},
        {"id": "d_specs", "title": "API specs", "body": "TBD",
         "owner_id": "u_bob", "state": "draft"},
        {"id": "d_old", "title": "Old plan", "body": "deprecated",
         "owner_id": "u_alice", "state": "archived"},
    ]
    for d in seeds:
        store[d["id"]] = d


# ---------------------------------------------------------------------------
# Server class
# ---------------------------------------------------------------------------

class MockProtocolServer:
    """One instance per episode. Owns its own FastAPI app + in-memory state."""

    def __init__(self, spec: dict | None = None):
        self.spec = spec or SPEC
        self.app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        self.users: dict[str, dict] = {}
        self.documents: dict[str, dict] = {}
        self.deleted_users: set[str] = set()
        self.deleted_documents: set[str] = set()
        # Map token -> {"scopes": set, "user_id": str}
        self.tokens: dict[str, dict] = {
            name: {"scopes": set(meta["scopes"])} for name, meta in TOKENS.items()
        }
        self._me_id = _seed_users(self.users)
        _seed_documents(self.documents)
        # Tie tokens to "me" user (alice) so /users/me / /auth/whoami resolve to a real user.
        for name in self.tokens:
            self.tokens[name]["user_id"] = self._me_id
        self._register_routes()

    # ----- Auth -----

    def _auth(self, scope_required: str | None):
        def _check(authorization: Annotated[str | None, Header()] = None) -> str:
            if not authorization or not authorization.lower().startswith("bearer "):
                raise HTTPException(401, {"error": "missing_or_invalid_token"})
            token = authorization[7:].strip()
            if token not in self.tokens:
                raise HTTPException(401, {"error": "missing_or_invalid_token"})
            if scope_required and scope_required not in self.tokens[token]["scopes"]:
                raise HTTPException(403, {"error": "insufficient_scope",
                                          "required": scope_required})
            return token
        return _check

    # ----- Routes -----

    def _register_routes(self) -> None:  # noqa: C901  (route registration is naturally long)
        app = self.app
        users = self.users
        docs = self.documents

        # 18: GET /_/health  (no auth)
        @app.get("/_/health")
        def health() -> dict:
            return {"ok": True}

        # 1: GET /users
        @app.get("/users")
        def list_users(
            limit: int = Query(20),
            cursor: str | None = Query(None),
            role: str | None = Query(None),
            _token: str = Depends(self._auth("users:read")),
        ) -> dict:
            if limit > 100 or limit < 1:
                raise HTTPException(422, {"error": "limit_too_high"})
            if role is not None and role not in ("member", "admin"):
                raise HTTPException(422, {"error": "invalid_role_filter"})
            visible = [u for uid, u in users.items() if uid not in self.deleted_users]
            if role:
                visible = [u for u in visible if u["role"] == role]
            return {"data": visible[:limit], "next_cursor": None}

        # 2: POST /users
        @app.post("/users", status_code=201)
        async def create_user(
            request: Request,
            _token: str = Depends(self._auth("users:write")),
        ) -> dict:
            body = await _safe_json(request)
            email = body.get("email")
            if not isinstance(email, str) or "@" not in email:
                raise HTTPException(422, {"error": "invalid_email"})
            role = body.get("role", "member")
            if role not in ("member", "admin"):
                raise HTTPException(422, {"error": "invalid_role"})
            if any(u["email"] == email for u in users.values()):
                raise HTTPException(409, {"error": "email_exists"})
            uid = f"u_{uuid.uuid4().hex[:8]}"
            users[uid] = {
                "id": uid,
                "email": email,
                "role": role,
                "status": "active",
                "created_at": _now(),
            }
            return users[uid]

        # 3a: GET /users/me  (alias of get_user, must be registered BEFORE /users/{id})
        @app.get("/users/me")
        def users_me(token: str = Depends(self._auth("users:read"))) -> dict:
            uid = self.tokens[token]["user_id"]
            return users[uid]

        # 3b: GET /users/{id}
        @app.get("/users/{user_id}")
        def get_user(user_id: str, _token: str = Depends(self._auth("users:read"))) -> dict:
            if user_id in self.deleted_users or user_id not in users:
                raise HTTPException(404, {"error": "not_found"})
            return users[user_id]

        # 4: PATCH /users/{id}
        @app.patch("/users/{user_id}")
        async def update_user(
            user_id: str,
            request: Request,
            _token: str = Depends(self._auth("users:write")),
        ) -> dict:
            if user_id in self.deleted_users or user_id not in users:
                raise HTTPException(404, {"error": "not_found"})
            body = await _safe_json(request)
            if "role" in body and body["role"] not in ("member", "admin"):
                raise HTTPException(422, {"error": "invalid_role"})
            for k in ("email", "role"):
                if k in body:
                    users[user_id][k] = body[k]
            return users[user_id]

        # 5: DELETE /users/{id}  (idempotent: 410 on second call)
        @app.delete("/users/{user_id}")
        def delete_user(user_id: str, _token: str = Depends(self._auth("users:write"))) -> dict:
            if user_id not in users:
                raise HTTPException(404, {"error": "not_found"})
            if user_id in self.deleted_users:
                raise HTTPException(410, {"error": "already_deleted"})
            self.deleted_users.add(user_id)
            return {"ok": True, "id": user_id}

        # 6: POST /users/{id}/suspend
        @app.post("/users/{user_id}/suspend")
        def suspend_user(user_id: str, _token: str = Depends(self._auth("users:write"))) -> dict:
            if user_id in self.deleted_users or user_id not in users:
                raise HTTPException(404, {"error": "not_found"})
            u = users[user_id]
            if u["status"] != "active":
                raise HTTPException(409, {"error": "invalid_state_transition",
                                          "from": u["status"], "to": "suspended"})
            u["status"] = "suspended"
            return u

        # 7: POST /users/{id}/restore
        @app.post("/users/{user_id}/restore")
        def restore_user(user_id: str, _token: str = Depends(self._auth("users:write"))) -> dict:
            if user_id in self.deleted_users or user_id not in users:
                raise HTTPException(404, {"error": "not_found"})
            u = users[user_id]
            if u["status"] != "suspended":
                raise HTTPException(409, {"error": "invalid_state_transition",
                                          "from": u["status"], "to": "active"})
            u["status"] = "active"
            return u

        # 8: GET /users/{id}/documents
        @app.get("/users/{user_id}/documents")
        def list_user_documents(
            user_id: str,
            state: str | None = Query(None),
            _token: str = Depends(self._auth("docs:read")),
        ) -> dict:
            if user_id in self.deleted_users or user_id not in users:
                raise HTTPException(404, {"error": "not_found"})
            owned = [d for did, d in docs.items()
                     if d["owner_id"] == user_id and did not in self.deleted_documents]
            if state:
                if state not in ("draft", "published", "archived"):
                    raise HTTPException(422, {"error": "invalid_state_filter"})
                owned = [d for d in owned if d["state"] == state]
            return {"data": owned}

        # 9: GET /docs
        @app.get("/docs")
        def list_documents(
            limit: int = Query(20),
            state: str | None = Query(None),
            _token: str = Depends(self._auth("docs:read")),
        ) -> dict:
            if limit > 100 or limit < 1:
                raise HTTPException(422, {"error": "limit_too_high"})
            visible = [d for did, d in docs.items() if did not in self.deleted_documents]
            if state:
                if state not in ("draft", "published", "archived"):
                    raise HTTPException(422, {"error": "invalid_state_filter"})
                visible = [d for d in visible if d["state"] == state]
            return {"data": visible[:limit]}

        # 10: POST /docs
        @app.post("/docs", status_code=201)
        async def create_document(
            request: Request,
            token: str = Depends(self._auth("docs:write")),
        ) -> dict:
            body = await _safe_json(request)
            title = body.get("title")
            if not isinstance(title, str) or not title.strip():
                raise HTTPException(422, {"error": "missing_title"})
            did = f"d_{uuid.uuid4().hex[:8]}"
            doc = {
                "id": did,
                "title": title,
                "body": body.get("body", ""),
                "owner_id": self.tokens[token]["user_id"],
                "state": "draft",
            }
            docs[did] = doc
            return doc

        # 11: GET /docs/{id}
        @app.get("/docs/{doc_id}")
        def get_document(doc_id: str, _token: str = Depends(self._auth("docs:read"))) -> dict:
            if doc_id in self.deleted_documents or doc_id not in docs:
                raise HTTPException(404, {"error": "not_found"})
            return docs[doc_id]

        # 12: PATCH /docs/{id}  (only allowed in draft state)
        @app.patch("/docs/{doc_id}")
        async def update_document(
            doc_id: str,
            request: Request,
            _token: str = Depends(self._auth("docs:write")),
        ) -> dict:
            if doc_id in self.deleted_documents or doc_id not in docs:
                raise HTTPException(404, {"error": "not_found"})
            d = docs[doc_id]
            if d["state"] != "draft":
                raise HTTPException(409, {"error": "not_in_draft_state",
                                          "current_state": d["state"]})
            body = await _safe_json(request)
            for k in ("title", "body"):
                if k in body and isinstance(body[k], str):
                    d[k] = body[k]
            return d

        # 13: DELETE /docs/{id}
        @app.delete("/docs/{doc_id}")
        def delete_document(doc_id: str, _token: str = Depends(self._auth("docs:write"))) -> dict:
            if doc_id in self.deleted_documents or doc_id not in docs:
                raise HTTPException(404, {"error": "not_found"})
            self.deleted_documents.add(doc_id)
            return {"ok": True, "id": doc_id}

        # 14: POST /docs/{id}/publish
        @app.post("/docs/{doc_id}/publish")
        def publish_document(doc_id: str, _token: str = Depends(self._auth("docs:write"))) -> dict:
            if doc_id in self.deleted_documents or doc_id not in docs:
                raise HTTPException(404, {"error": "not_found"})
            d = docs[doc_id]
            if d["state"] != "draft":
                raise HTTPException(409, {"error": "invalid_state_transition",
                                          "from": d["state"], "to": "published"})
            d["state"] = "published"
            return d

        # 15: POST /docs/{id}/archive
        @app.post("/docs/{doc_id}/archive")
        def archive_document(doc_id: str, _token: str = Depends(self._auth("docs:write"))) -> dict:
            if doc_id in self.deleted_documents or doc_id not in docs:
                raise HTTPException(404, {"error": "not_found"})
            d = docs[doc_id]
            if d["state"] != "published":
                raise HTTPException(409, {"error": "invalid_state_transition",
                                          "from": d["state"], "to": "archived"})
            d["state"] = "archived"
            return d

        # 16: GET /auth/whoami
        @app.get("/auth/whoami")
        def whoami(token: str = Depends(self._auth("users:read"))) -> dict:
            uid = self.tokens[token]["user_id"]
            return users[uid]

        # 17: GET /auth/scopes  (requires admin)
        @app.get("/auth/scopes")
        def list_scopes(_token: str = Depends(self._auth("admin"))) -> dict:
            return {"data": sorted(self.spec["auth"]["scopes"])}


async def _safe_json(request: Request) -> dict[str, Any]:
    """Return parsed JSON body or {} if missing/invalid."""
    try:
        body = await request.json()
        if isinstance(body, dict):
            return body
        return {}
    except Exception:
        return {}


def create_server(spec: dict | None = None) -> MockProtocolServer:
    return MockProtocolServer(spec)
