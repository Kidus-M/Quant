"""Configuration loading.

Every tunable number in this project lives in YAML. If you find yourself typing a
constant into a module, it belongs here instead -- a backtester with hidden
constants cannot be audited.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(KeyError):
    """A required configuration key is missing or malformed."""


@dataclass(frozen=True)
class Config:
    """Read-only view over the parsed YAML with dotted lookup.

    ``cfg.get("costs.slippage_usd_per_oz_per_side")`` raises rather than silently
    returning a default, unless a default is passed explicitly. Silent defaults are
    how a cost model quietly becomes optimistic.
    """

    data: dict[str, Any]
    source_path: Path | None = None

    _MISSING = object()

    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is Config._MISSING:
                    raise ConfigError(
                        f"missing config key {dotted!r}"
                        + (f" in {self.source_path}" if self.source_path else "")
                    )
                return default
            node = node[part]
        return node

    def section(self, name: str) -> "Config":
        node = self.get(name)
        if not isinstance(node, dict):
            raise ConfigError(f"config key {name!r} is not a section")
        return Config(node, self.source_path)

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with dotted-key overrides applied. Used by walk-forward."""
        merged = _deep_copy(self.data)
        for dotted, value in overrides.items():
            node = merged
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(merged, self.source_path)

    def as_dict(self) -> dict[str, Any]:
        return _deep_copy(self.data)


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open("r", encoding="utf-8") as fh:
        parsed = yaml.safe_load(fh) or {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    return Config(parsed, path)


def load_configs(*paths: str | Path) -> Config:
    """Load several YAML files and deep-merge them left to right.

    Used to layer ``config/alerts.yaml`` on top of ``config/backtest.yaml`` so the
    phase 2 runner reads one merged view without the backtest config growing a
    section it never uses.
    """
    if not paths:
        raise ConfigError("load_configs needs at least one path")
    merged: dict[str, Any] = {}
    last: Path | None = None
    for path in paths:
        loaded = load_config(path)
        merged = _deep_merge(merged, loaded.data)
        last = loaded.source_path
    return Config(merged, last)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = _deep_copy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = _deep_copy(value)
    return out


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Minimal .env reader. Values are placed in os.environ if not already set.

    Credentials are never committed; this only reads a gitignored file.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    found: dict[str, str] = {}
    if not path.exists():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if not value:
            continue
        found[key] = value
        os.environ.setdefault(key, value)
    return found


def resolve_path(p: str | Path) -> Path:
    """Resolve a config-supplied path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
