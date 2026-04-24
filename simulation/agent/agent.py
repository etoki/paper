"""Anthropic Messages API wrapper for the Big Five -> hensachi agent.

Design
------
- Model: claude-opus-4-7 (strongest available Claude, 1M context).
- Extended thinking enabled with a generous budget to let the three-stage
  reasoning (backstory -> learning behavior -> examination outcome) happen
  inside the hidden thinking block.
- Output is constrained to a single call of the submit_prediction tool,
  guaranteeing a structured JSON payload conforming to the schema declared
  in prompts.PREDICTION_TOOL.
- Prompt caching (5-min TTL) is applied to the static system prompt so that
  repeated calls across 103 participants x N samples only pay full price
  on the first call in a 5-minute window.
- Temperature is fixed at 1.0 because extended thinking requires it.
  Per-participant variability across the N samples is driven by
  sampling stochasticity, not by temperature knob.

Environment
-----------
ANTHROPIC_API_KEY must be set (via shell export or .env file; .env is
loaded automatically at import time). The Anthropic() client reads the key
from the environment.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Mapping

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from anthropic import Anthropic, APIError

from load_env import load_env
from prompts import PREDICTION_TOOL, SYSTEM_PROMPT, build_user_prompt

load_env()


MODEL = "claude-opus-4-7"
MAX_TOKENS = 8000
# Opus 4.7 uses adaptive thinking + output_config.effort rather than an
# explicit budget_tokens. "high" effort allocates more reasoning depth.
THINKING_EFFORT = "high"
# Thinking requires temperature = 1.0.
TEMPERATURE = 1.0


@dataclass
class AgentOutput:
    parsed: dict[str, Any]
    thinking: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    stop_reason: str


def _extract_tool_call(msg: Any) -> tuple[dict[str, Any], str]:
    """Pull the submit_prediction tool input and the thinking trace from a message."""
    tool_input: dict[str, Any] | None = None
    thinking_parts: list[str] = []
    for block in msg.content:
        btype = getattr(block, "type", None)
        if btype == "tool_use" and getattr(block, "name", "") == "submit_prediction":
            tool_input = block.input
        elif btype == "thinking":
            thinking_parts.append(getattr(block, "thinking", "") or "")
    if tool_input is None:
        raise ValueError(
            "Model did not call the submit_prediction tool. "
            f"content types: {[getattr(b, 'type', None) for b in msg.content]}"
        )
    return tool_input, "\n".join(thinking_parts)


def run_one(
    client: Anthropic,
    big_five: Mapping[str, float],
    *,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    thinking_effort: str = THINKING_EFFORT,
    max_retries: int = 3,
) -> AgentOutput:
    """Run a single Big Five -> hensachi prediction via tool use + extended thinking."""
    user_prompt = build_user_prompt(big_five)
    system_block = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    last_err = None
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
                thinking={"type": "adaptive"},
                output_config={"effort": thinking_effort},
                system=system_block,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[PREDICTION_TOOL],
                # Opus 4.7 API constraint: extended thinking is incompatible
                # with tool_choice that forces a specific tool. We use "auto"
                # and rely on the system-prompt directive ("You must call the
                # submit_prediction tool exactly once") plus the retry loop
                # to guarantee a tool call.
                tool_choice={"type": "auto"},
            )
            parsed, thinking = _extract_tool_call(msg)
            usage = msg.usage
            return AgentOutput(
                parsed=parsed,
                thinking=thinking,
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                stop_reason=getattr(msg, "stop_reason", "") or "",
            )
        except (APIError, ValueError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"run_one failed after {max_retries} retries: {last_err}")


def main_smoke_test() -> None:
    """Single-call smoke test to verify API connectivity, thinking, and tool call."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Set it before running.")

    client = Anthropic()
    big_five = {
        "Openness": 3.42,
        "Conscientiousness": 3.92,
        "Extraversion": 4.25,
        "Agreeableness": 4.25,
        "Neuroticism": 3.42,
    }
    print(f"[smoke test] Big Five = {big_five}")
    out = run_one(client, big_five)
    print(
        f"[smoke test] stop_reason={out.stop_reason} "
        f"input={out.input_tokens} output={out.output_tokens} "
        f"cache_read={out.cache_read_tokens} cache_write={out.cache_write_tokens}"
    )
    print(f"[smoke test] thinking (first 400 chars):\n{out.thinking[:400]}...")
    print()
    print("[smoke test] tool-call arguments:")
    print(json.dumps(out.parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main_smoke_test()
