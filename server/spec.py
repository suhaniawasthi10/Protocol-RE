"""Ground-truth protocol specification.

This module is the canonical description of what the mock protocol server does.
It is consumed by:
  - protocol_server.py (to drive the FastAPI route handlers)
  - matcher.py (to score the agent's belief graph)
  - designer.py (to know what to mutate)

The agent NEVER sees this spec. The whole task of Protocol One is for the
agent to reconstruct an approximation of this dict by probing the server.

Invariants:
  - Exactly 18 endpoints
  - Two resources (User, Document) with state machines
  - Bearer auth with five named scopes
  - Stable, alphabetically-sortable structure (matcher relies on determinism,
    not on dict ordering, but it is friendly to humans reading rollouts).
"""

SPEC: dict = {
    "auth": {
        "type": "bearer",
        "scopes": ["users:read", "users:write", "docs:read", "docs:write", "admin"],
    },
    "resources": [
        {
            "name": "User",
            "fields": [
                {"name": "id", "type": "string", "required": True},
                {"name": "email", "type": "string", "required": True},
                {"name": "role", "type": "string", "required": True, "enum": ["member", "admin"]},
                {"name": "status", "type": "string", "required": True, "enum": ["active", "suspended"]},
                {"name": "created_at", "type": "timestamp", "required": True},
            ],
            "state_machine": {
                "initial": "active",
                "states": ["active", "suspended"],
                "transitions": [
                    {"from": "active", "to": "suspended", "via": "POST /users/{id}/suspend"},
                    {"from": "suspended", "to": "active", "via": "POST /users/{id}/restore"},
                ],
            },
        },
        {
            "name": "Document",
            "fields": [
                {"name": "id", "type": "string", "required": True},
                {"name": "title", "type": "string", "required": True},
                {"name": "body", "type": "string", "required": False},
                {"name": "owner_id", "type": "string", "required": True},
                {"name": "state", "type": "string", "required": True,
                 "enum": ["draft", "published", "archived"]},
            ],
            "state_machine": {
                "initial": "draft",
                "states": ["draft", "published", "archived"],
                "transitions": [
                    {"from": "draft", "to": "published", "via": "POST /docs/{id}/publish"},
                    {"from": "published", "to": "archived", "via": "POST /docs/{id}/archive"},
                    # archived -> anything is forbidden; agent must discover this
                ],
            },
        },
    ],
    "endpoints": [
        # 1
        {
            "id": "list_users",
            "method": "GET",
            "path": "/users",
            "auth_required": True,
            "auth_scope": "users:read",
            "query_params": [
                {"name": "limit", "type": "int", "required": False, "default": 20, "max": 100},
                {"name": "cursor", "type": "string", "required": False},
                {"name": "role", "type": "string", "required": False, "enum": ["member", "admin"]},
            ],
            "responses": {
                "200": {"shape": "list<User>", "has_pagination": True},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "422": {"shape": "error", "when": "invalid_role_filter_or_limit_too_high"},
            },
            "aliases": [],
        },
        # 2
        {
            "id": "create_user",
            "method": "POST",
            "path": "/users",
            "auth_required": True,
            "auth_scope": "users:write",
            "body_fields": [
                {"name": "email", "type": "string", "required": True},
                {"name": "role", "type": "string", "required": False, "default": "member",
                 "enum": ["member", "admin"]},
            ],
            "responses": {
                "201": {"shape": "User"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "409": {"shape": "error", "when": "email_exists"},
                "422": {"shape": "error", "when": "invalid_email_or_role"},
            },
            "aliases": [],
        },
        # 3
        {
            "id": "get_user",
            "method": "GET",
            "path": "/users/{id}",
            "auth_required": True,
            "auth_scope": "users:read",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "responses": {
                "200": {"shape": "User"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
            },
            "aliases": ["/users/me"],  # /users/me resolves to current user (token-scoped)
        },
        # 4
        {
            "id": "update_user",
            "method": "PATCH",
            "path": "/users/{id}",
            "auth_required": True,
            "auth_scope": "users:write",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "body_fields": [
                {"name": "email", "type": "string", "required": False},
                {"name": "role", "type": "string", "required": False, "enum": ["member", "admin"]},
            ],
            "responses": {
                "200": {"shape": "User"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
                "422": {"shape": "error", "when": "invalid_role"},
            },
            "aliases": [],
        },
        # 5
        {
            "id": "delete_user",
            "method": "DELETE",
            "path": "/users/{id}",
            "auth_required": True,
            "auth_scope": "users:write",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "responses": {
                "200": {"shape": "ack"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
                "410": {"shape": "error", "when": "already_deleted"},  # idempotent: 2nd call
            },
            "aliases": [],
        },
        # 6
        {
            "id": "suspend_user",
            "method": "POST",
            "path": "/users/{id}/suspend",
            "auth_required": True,
            "auth_scope": "users:write",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "responses": {
                "200": {"shape": "User"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
                "409": {"shape": "error", "when": "invalid_state_transition"},
            },
            "aliases": [],
        },
        # 7
        {
            "id": "restore_user",
            "method": "POST",
            "path": "/users/{id}/restore",
            "auth_required": True,
            "auth_scope": "users:write",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "responses": {
                "200": {"shape": "User"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
                "409": {"shape": "error", "when": "invalid_state_transition"},
            },
            "aliases": [],
        },
        # 8
        {
            "id": "list_user_documents",
            "method": "GET",
            "path": "/users/{id}/documents",
            "auth_required": True,
            "auth_scope": "docs:read",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "query_params": [
                {"name": "state", "type": "string", "required": False,
                 "enum": ["draft", "published", "archived"]},
            ],
            "responses": {
                "200": {"shape": "list<Document>"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
            },
            "aliases": [],
        },
        # 9
        {
            "id": "list_documents",
            "method": "GET",
            "path": "/docs",
            "auth_required": True,
            "auth_scope": "docs:read",
            "query_params": [
                {"name": "limit", "type": "int", "required": False, "default": 20, "max": 100},
                {"name": "state", "type": "string", "required": False,
                 "enum": ["draft", "published", "archived"]},
            ],
            "responses": {
                "200": {"shape": "list<Document>"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
            },
            "aliases": [],
        },
        # 10
        {
            "id": "create_document",
            "method": "POST",
            "path": "/docs",
            "auth_required": True,
            "auth_scope": "docs:write",
            "body_fields": [
                {"name": "title", "type": "string", "required": True},
                {"name": "body", "type": "string", "required": False},
            ],
            "responses": {
                "201": {"shape": "Document"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "422": {"shape": "error", "when": "missing_title"},
            },
            "aliases": [],
        },
        # 11
        {
            "id": "get_document",
            "method": "GET",
            "path": "/docs/{id}",
            "auth_required": True,
            "auth_scope": "docs:read",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "responses": {
                "200": {"shape": "Document"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
            },
            "aliases": [],
        },
        # 12
        {
            "id": "update_document",
            "method": "PATCH",
            "path": "/docs/{id}",
            "auth_required": True,
            "auth_scope": "docs:write",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "body_fields": [
                {"name": "title", "type": "string", "required": False},
                {"name": "body", "type": "string", "required": False},
            ],
            "responses": {
                "200": {"shape": "Document"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
                "409": {"shape": "error", "when": "not_in_draft_state"},
            },
            "aliases": [],
        },
        # 13
        {
            "id": "delete_document",
            "method": "DELETE",
            "path": "/docs/{id}",
            "auth_required": True,
            "auth_scope": "docs:write",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "responses": {
                "200": {"shape": "ack"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
            },
            "aliases": [],
        },
        # 14
        {
            "id": "publish_document",
            "method": "POST",
            "path": "/docs/{id}/publish",
            "auth_required": True,
            "auth_scope": "docs:write",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "responses": {
                "200": {"shape": "Document"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
                "409": {"shape": "error", "when": "invalid_state_transition"},
            },
            "aliases": [],
        },
        # 15
        {
            "id": "archive_document",
            "method": "POST",
            "path": "/docs/{id}/archive",
            "auth_required": True,
            "auth_scope": "docs:write",
            "path_params": [{"name": "id", "type": "string", "required": True}],
            "responses": {
                "200": {"shape": "Document"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
                "404": {"shape": "error", "when": "not_found"},
                "409": {"shape": "error", "when": "invalid_state_transition"},
            },
            "aliases": [],
        },
        # 16
        {
            "id": "whoami",
            "method": "GET",
            "path": "/auth/whoami",
            "auth_required": True,
            "auth_scope": "users:read",
            "responses": {
                "200": {"shape": "User"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
            },
            "aliases": [],
        },
        # 17
        {
            "id": "list_scopes",
            "method": "GET",
            "path": "/auth/scopes",
            "auth_required": True,
            "auth_scope": "admin",
            "responses": {
                "200": {"shape": "list<string>"},
                "401": {"shape": "error", "when": "missing_or_invalid_token"},
                "403": {"shape": "error", "when": "insufficient_scope"},
            },
            "aliases": [],
        },
        # 18
        {
            "id": "health",
            "method": "GET",
            "path": "/_/health",
            "auth_required": False,
            "auth_scope": None,
            "responses": {
                "200": {"shape": "ack"},
            },
            "aliases": [],
        },
    ],
}


# Tokens that exist on the mock server. Agent gets `INITIAL_TOKEN` in the
# system prompt; other tokens (with different scopes) must be discovered.
TOKENS: dict = {
    "token_full": {"scopes": ["users:read", "users:write", "docs:read", "docs:write", "admin"]},
    "token_read": {"scopes": ["users:read", "docs:read"]},
    "token_write": {"scopes": ["users:write", "docs:write"]},
    "token_admin": {"scopes": ["admin"]},
}

INITIAL_TOKEN = "token_full"
