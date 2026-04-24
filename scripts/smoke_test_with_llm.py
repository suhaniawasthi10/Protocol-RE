#!/usr/bin/env python
"""LLM-driven smoke test (Claude or OpenAI). Requires an API key.

Use ANTHROPIC_API_KEY or OPENAI_API_KEY. If you have neither, run
scripts/smoke_test_scripted.py instead — it reaches the same verification
without an API key.

Target: a stock LLM with no training should land in 0.2 - 0.5 on this env.
  - Below 0.2 → matcher broken, tool calling broken, or prompt too weak.
  - Above 0.6 → matcher too forgiving, retune before training.

Run:
    ANTHROPIC_API_KEY=sk-ant-... python scripts/smoke_test_with_llm.py
Remote:
    ANTHROPIC_API_KEY=... SPACE_URL=https://.../protocol_one_env.hf.space \\
        python scripts/smoke_test_with_llm.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


MAX_LLM_TURNS = 30
MODEL_DEFAULT_ANTHROPIC = "claude-sonnet-4-6"
MODEL_DEFAULT_OPENAI = "gpt-4o-mini"


SYSTEM_PROMPT = """You are reverse-engineering an undocumented HTTP API to \
build a complete belief graph of it.

On each turn, respond with EXACTLY one JSON object on its own line (no prose, \
no markdown fence) describing one tool call:
  {"tool": "probe", "args": {"method": "GET", "path": "/users", "headers": {}, "body": null}}
  {"tool": "update_model", "args": {"delta": {"endpoints": [...], "resources": [...], "auth": {...}}}}
  {"tool": "finalize", "args": {}}

Probe efficiently — 80 probes max. When you have enough information, call finalize.
"""


def _start_local_server(port: int) -> None:
    import uvicorn
    from server.app import app
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    import httpx
    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.5).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("server never came up")


def _llm_call_anthropic(messages: list[dict], model: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    # Anthropic API expects "user" and "assistant" roles; system is separate
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    msgs = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=msgs,  # type: ignore
    )
    return resp.content[0].text  # type: ignore


def _llm_call_openai(messages: list[dict], model: str) -> str:
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=1024)
    return resp.choices[0].message.content or ""


def _parse_tool_call(text: str) -> dict | None:
    """Pull out the first JSON object in `text` that has a `tool` key."""
    text = text.strip()
    # Strip markdown fences if present
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[len("json"):].strip()
            try:
                obj = json.loads(p)
                if isinstance(obj, dict) and "tool" in obj:
                    return obj
            except Exception:
                continue
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except Exception:
        pass
    # Last resort: scan for {...} blocks
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict) and "tool" in obj:
                        return obj
                except Exception:
                    pass
    return None


def run_episode(base_url: str, provider: str) -> float:
    from client import ProtocolOneEnv
    from models import ProtocolOneAction

    env = ProtocolOneEnv(base_url=base_url).sync()
    reward = 0.0

    with env:
        result = env.reset()
        initial_text = result.observation.text or ""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_text},
        ]

        for turn in range(MAX_LLM_TURNS):
            if provider == "anthropic":
                raw = _llm_call_anthropic(messages, MODEL_DEFAULT_ANTHROPIC)
            else:
                raw = _llm_call_openai(messages, MODEL_DEFAULT_OPENAI)
            messages.append({"role": "assistant", "content": raw})

            call = _parse_tool_call(raw)
            if call is None:
                messages.append({"role": "user", "content":
                                 "Could not parse your JSON tool call. "
                                 "Respond with a single JSON object with 'tool' and 'args' keys."})
                continue

            try:
                result = env.step(ProtocolOneAction(
                    tool=call["tool"], args=call.get("args", {})
                ))
            except Exception as e:
                messages.append({"role": "user", "content": f"Env error: {e}. Try again."})
                continue

            text = (result.observation.text or "")[:1500]
            print(f"[t={turn:2d}] tool={call['tool']!r}  "
                  f"probes_used={result.observation.probes_used}  reward={result.reward}")
            messages.append({"role": "user", "content": text})

            if result.done:
                reward = float(result.reward or 0.0)
                break

    return reward


def main() -> int:
    # Pick provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        provider = "anthropic"
    elif os.environ.get("OPENAI_API_KEY"):
        provider = "openai"
    else:
        print("Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY set.")
        print("To skip needing an API key, run: python scripts/smoke_test_scripted.py")
        return 2

    space_url = os.environ.get("SPACE_URL")
    if space_url:
        print(f"Using remote SPACE_URL={space_url}")
        base_url = space_url
    else:
        port = 8767
        print(f"Booting local uvicorn on :{port} …")
        _start_local_server(port)
        base_url = f"http://127.0.0.1:{port}"

    reward = run_episode(base_url, provider)
    print(f"\nLLM ({provider}) reward: {reward:.3f}")
    if 0.1 <= reward <= 0.7:
        print("✓ reward in plausible band for a stock LLM baseline")
        return 0
    print(f"⚠ reward {reward:.3f} outside expected [0.1, 0.7]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
