"""Minimal .env loader (no external dependency).

Reads simulation/agent/.env and injects KEY=VALUE pairs into os.environ.
Pre-existing non-empty environment variables take precedence (so CI/CD or
shell exports are respected), but empty-string placeholders (common on
Windows when a user runs `setx KEY` with no value) are treated as unset
and overwritten from the .env file. Lines starting with '#' and blank
lines are skipped. Quoted values (single or double) are unquoted.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(env_path: str | Path | None = None) -> dict[str, str]:
    """Load variables from a .env file. Returns the dict of loaded keys."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
    env_path = Path(env_path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key, "").strip():
            os.environ[key] = value
            loaded[key] = value
    return loaded


if __name__ == "__main__":
    loaded = load_env()
    for k in loaded:
        print(f"loaded: {k}")
