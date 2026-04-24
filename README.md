---
title: Protocol One Env Environment Server
emoji: 📡
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - llm
---

# Protocol One

An OpenEnv-compatible RL environment where an LLM agent learns to **reverse-engineer an undocumented HTTP API by probing it**.

A hidden ground-truth protocol spec lives inside the environment. The agent sends HTTP requests through a `probe` tool, observes responses, and emits a structured belief model of the API (endpoints, parameters, state transitions, error semantics, auth scopes). Reward is computed by comparing the agent's belief graph against the hidden spec. A scripted designer agent mutates the protocol between episodes, forcing the discoverer to keep its world model fresh — that's the self-improvement angle.

Built for the Meta OpenEnv Hackathon finale (Apr 25-26, Bangalore). This repo is the pre-hackathon environment delivery; training (GRPO on Qwen 2.5-1.5B via Unsloth) happens on-site.

## What's in this Space

A single FastAPI app that exposes:
- An OpenEnv `Environment` class (`ProtocolOneEnvironment`) with `reset()`/`step()`.
- Three tools for the agent: `probe(method, path, headers, body)`, `update_model(delta)`, `finalize()`.
- A hidden mock API with 18 endpoints, bearer-token auth over 5 scopes, two resources (`User`, `Document`) with state machines, soft-delete idempotency, and scoped aliases (`/users/me`).
- A deterministic 6-component reward matcher that returns a score in [0, 1] plus per-component breakdown.
- A scripted mutation designer (5 mutation types), disabled by default via `MUTATION_PROBABILITY=0.0`.

## Run locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m server.app --port 8000
```

The env is reachable at `http://localhost:8000`. OpenEnv HTTP endpoints: `/reset`, `/step`, `/state`, `/schema`, `/health`, `/ws` (WebSocket).

## Verify Phase 1 (spec + matcher + mock server)

```bash
python scripts/verify_phase1.py
```

Expected output:

```
✓ Spec validates
✓ Server boots and serves authed traffic
✓ Empty belief graph    : 0.000
✓ Perfect belief graph  : 1.000
✓ Half belief graph     : 0.474
✓ Garbage penalised     : 20 false claims, total 0.000
✓ 18 endpoints in spec
Phase 1 verified.
```

## Verify Phase 2 (OpenEnv wrapper end-to-end)

```bash
python scripts/verify_phase2.py
```

Drives the env through `ProtocolOneEnv` (WebSocket client): unauth probe → 401, authed probe → 200, `update_model` merges, `finalize` computes reward.

## Smoke test (no API key)

```bash
python scripts/smoke_test_scripted.py
```

A deterministic heuristic agent exercises the env end-to-end. Target reward: 0.2 - 0.5 (baseline band). Currently **0.459**.

## Smoke test with a real LLM

```bash
ANTHROPIC_API_KEY=sk-ant-... python scripts/smoke_test_with_llm.py
# or:
OPENAI_API_KEY=sk-...       python scripts/smoke_test_with_llm.py
```

## Full test suite

```bash
pytest tests/ -q
```

89 tests covering the matcher (17), matcher calibration (8), the mock server (45), and the designer (15).

## Deploy to Hugging Face Spaces

1. Authenticate: `huggingface-cli login` (paste a write-scoped HF token).
2. Push:
   ```bash
   openenv push
   ```
   or, if you already have the repo created:
   ```bash
   openenv push --repo-id <your-username>/protocol_one_env
   ```
3. After the Docker build finishes (2-5 min), the Space URL is `https://<your-username>-protocol-one-env.hf.space`.
4. Smoke test against the Space:
   ```bash
   SPACE_URL=https://<your-username>-protocol-one-env.hf.space \
       python scripts/smoke_test_scripted.py
   ```

## Hackathon-day toggles

All mutation behavior is env-var controlled so you never edit code at the venue:

| Variable | Default | Hackathon value | Effect |
|---|---|---|---|
| `MUTATION_PROBABILITY` | `0.0` | `0.2` then `0.4` | Per-episode chance of applying one mutation |
| `MUTATION_START_EPISODE` | `50` | `0` on resume | Cooldown — first N episodes always run base spec |

Set on the HF Space via the Space settings UI (Variables & Secrets → Environment Variables).

## Architecture

```
Colab / HF training node
    │  WebSocket (OpenEnv)
    ▼
HF Space (Docker)
    └── FastAPI app (openenv-core)
          ├── ProtocolOneEnvironment  ← one instance per WebSocket session
          │     ├── MockProtocolServer  (18 endpoints, in-process TestClient)
          │     ├── Designer (mutates SPEC between episodes; off by default)
          │     └── Matcher (6-component scoring, pure & deterministic)
          └── SPEC  ← hidden from the agent
```

## Reward

Six components weighted to total in [0, 1]:

| Component | Weight | Meaning |
|---|---|---|
| `endpoints_discovered` | 0.35 | % of 18 endpoints agent correctly named (method + normalized path) |
| `endpoint_details` | 0.25 | For discovered endpoints: correct auth, params, response codes |
| `resources` | 0.15 | % of resource fields correctly listed |
| `state_machines` | 0.10 | % of state transitions correctly identified |
| `auth` | 0.15 | Auth type + scopes observed |
| `false_claim_penalty` | −0.15 cap | 0.01 per hallucinated endpoint / resource / scope |

Normalization: paths lower-cased, trailing slash stripped, `{user_id}` / `{userId}` collapsed to `{id}`. Deterministic. Calibrated via monotonicity, saturation, and gradient unit tests.

## Anti-reward-hacking

Per the hackathon help-guide recommendations:
- Six independent reward components with a false-claim penalty — no single signal can be gamed.
- Reward is **only** emitted at terminal step (`finalize()` or probe-budget exhaustion). Spamming `update_model` mid-episode does nothing.
- Matcher is pure & deterministic (no time, no shared globals).
- Mock server state is recreated fresh every `reset()`, so rollout N can't leak into rollout N+1.

## Directory layout

```
protocol_one_env/
├── __init__.py
├── client.py                # EnvClient subclass (what trainers import)
├── models.py                # ProtocolOneAction (discriminator) + Observation
├── openenv.yaml
├── pyproject.toml
├── README.md
├── server/
│   ├── app.py                    # FastAPI create_app, max_concurrent_envs=64
│   ├── designer.py               # 5 scripted mutation types (off by default)
│   ├── matcher.py                # 6-component reward
│   ├── protocol_one_env_environment.py  # OpenEnv Environment subclass
│   ├── protocol_server.py        # FastAPI mock, 18 endpoints
│   ├── spec.py                   # Hidden ground-truth protocol
│   ├── spec_schema.py            # Pydantic validator
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   ├── verify_phase1.py          # Spec + matcher + server consistency
│   ├── verify_phase2.py          # End-to-end env via WebSocket
│   ├── smoke_test_scripted.py    # No-API-key baseline (~0.46)
│   └── smoke_test_with_llm.py    # Claude / OpenAI driver
└── tests/
    ├── test_matcher.py                   # 17 tests
    ├── test_matcher_reward_calibration.py  # 8 tests
    ├── test_protocol_server.py           # 45 tests
    └── test_designer.py                  # 15 tests
```

## Status

- [x] Phase 0 — scaffold via `openenv init`, venv, deps
- [x] Phase 1 — spec (18 endpoints), matcher (6 components, 29 tests pass), mock server (45 tests pass)
- [x] Phase 2 — OpenEnv wrapper, reward wiring, concurrent sessions, smoke test reward ~0.46
- [x] Phase 4 designer code — 5 mutation types, 15 tests, disabled by default
- [ ] Phase 3 — GRPO training (hackathon day 1)
- [ ] Phase 5 — viz, video, pitch (hackathon day 2)
