"""YAML config loading + schema validation.

CLAUDE.md §5 / agent.md §5: "配置驱动，禁止关键超参散落在代码中". Every entry
point loads a YAML file through :func:`load_config` and validates it with a
small declarative spec, so a typo in a config key fails loudly *before* any
GPU time is spent instead of silently falling back to a default.

We intentionally avoid Hydra/pydantic here: Phase 0 must run on a CPU-only box
with the legacy environment, so the validator is ~100 lines of stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Type

import yaml


class ConfigError(ValueError):
    """Raised for any missing/extra/ill-typed config field."""


@dataclass(frozen=True)
class FieldSpec:
    """Declarative spec for one config field.

    ``types`` is a tuple of accepted python types. ``choices`` restricts values.
    ``bounds`` is an inclusive ``(low, high)`` range for numeric fields.
    ``required`` fields must be present; otherwise ``default`` is injected.
    """

    types: Tuple[Type[Any], ...]
    required: bool = True
    default: Any = None
    choices: Sequence[Any] | None = None
    bounds: Tuple[float | None, float | None] | None = None
    doc: str = ""


Schema = Mapping[str, "FieldSpec | Mapping[str, Any]"]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        raise ConfigError(f"config file is empty: {p}")
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}: {p}")
    return data


def validate(config: Mapping[str, Any], schema: Schema, path: str = "") -> Dict[str, Any]:
    """Validate ``config`` against ``schema`` and return a normalised copy.

    Unknown keys are an error (protects against silently ignored overrides).
    """
    if not isinstance(config, Mapping):
        raise ConfigError(f"{path or '<root>'} must be a mapping, got {type(config).__name__}")

    unknown = set(config) - set(schema)
    if unknown:
        raise ConfigError(f"unknown config keys at {path or '<root>'}: {sorted(unknown)}")

    out: Dict[str, Any] = {}
    for key, spec in schema.items():
        full = f"{path}.{key}" if path else key
        if isinstance(spec, Mapping):  # nested section
            section = config.get(key)
            if section is None:
                raise ConfigError(f"missing required config section: {full}")
            out[key] = validate(section, spec, full)
            continue

        if key not in config:
            if spec.required:
                raise ConfigError(f"missing required config key: {full} ({spec.doc})")
            out[key] = spec.default
            continue

        value = config[key]
        # An explicit YAML null on an optional field means "use the default".
        # This lets a config document a knob (e.g. `max_samples: null`) instead of
        # omitting it, which is clearer for reviewers.
        if value is None and not spec.required:
            out[key] = spec.default
            continue
        if value is None:
            raise ConfigError(f"{full} is required but was set to null")
        # bool is a subclass of int; reject the accidental "true" for an int field
        if bool in spec.types and isinstance(value, bool):
            pass
        elif isinstance(value, bool) and bool not in spec.types:
            raise ConfigError(f"{full} must be one of {[t.__name__ for t in spec.types]}, got bool")
        if not isinstance(value, spec.types):
            # allow int where float is expected
            if float in spec.types and isinstance(value, int):
                value = float(value)
            else:
                raise ConfigError(
                    f"{full} must be one of {[t.__name__ for t in spec.types]}, "
                    f"got {type(value).__name__}"
                )
        if spec.choices is not None and value not in spec.choices:
            raise ConfigError(f"{full} must be one of {list(spec.choices)}, got {value!r}")
        if spec.bounds is not None:
            low, high = spec.bounds
            if low is not None and value < low:
                raise ConfigError(f"{full} must be >= {low}, got {value}")
            if high is not None and value > high:
                raise ConfigError(f"{full} must be <= {high}, got {value}")
        out[key] = value
    return out


def load_config(path: str | Path, schema: Schema | None = None) -> Dict[str, Any]:
    raw = load_yaml(path)
    if schema is None:
        return raw
    return validate(raw, schema)
