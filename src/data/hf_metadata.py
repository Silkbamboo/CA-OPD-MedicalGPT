"""Validate Hugging Face builder config and split names before downloading data.

The source adapters are unit-tested with local rows, but that cannot catch an
invented remote builder name (the previous config used C-Eval ``name=all``, which
does not exist). This gate asks only for repository metadata and therefore fails
fast before the large data build or a paid GPU session.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.data.sources import MEDICAL_CEVAL_SUBJECTS, SourceSpec
from src.data.build_splits import DATA_SCHEMA
from src.data.schema import SchemaError
from src.utils.config import load_config
from src.utils.io import write_json

ConfigNamesFn = Callable[[str, str, bool], Sequence[str]]
SplitNamesFn = Callable[[str, str, str, bool], Sequence[str]]


def hf_source_specs(config_path: str | Path) -> List[SourceSpec]:
    """Load and flatten only the HF source specs from a data config."""
    config = load_config(config_path, DATA_SCHEMA)
    specs: List[SourceSpec] = []
    for entries in config["sources"].values():
        for raw in entries:
            spec = SourceSpec.from_mapping(raw)
            if spec.kind == "hf":
                specs.append(spec)
    return specs


def verify_hf_metadata(
    config_path: str | Path,
    config_names_fn: Optional[ConfigNamesFn] = None,
    split_names_fn: Optional[SplitNamesFn] = None,
) -> Dict[str, Any]:
    """Verify every declared HF config and split; download no dataset rows.

    Function arguments are injectable so CI can prove all failure modes without
    network access. With no overrides, Hugging Face ``datasets`` metadata APIs
    are used on the target/data-preparation box.
    """
    if config_names_fn is None or split_names_fn is None:
        try:
            from datasets import get_dataset_config_names, get_dataset_split_names
        except ImportError as exc:  # pragma: no cover - target environment path
            raise RuntimeError("HF metadata verification requires the 'datasets' package") from exc
        def default_config_names(path: str, revision: str, trust: bool) -> Sequence[str]:
            kwargs: Dict[str, Any] = {"revision": revision}
            if trust:
                kwargs["trust_remote_code"] = True
            return get_dataset_config_names(path, **kwargs)

        def default_split_names(path: str, name: str, revision: str, trust: bool) -> Sequence[str]:
            kwargs: Dict[str, Any] = {"revision": revision}
            if trust:
                kwargs["trust_remote_code"] = True
            return get_dataset_split_names(path, name, **kwargs)

        config_names_fn = config_names_fn or default_config_names
        split_names_fn = split_names_fn or default_split_names

    specs = hf_source_specs(config_path)
    available_cache: Dict[tuple[str, str, bool], set[str]] = {}
    split_cache: Dict[tuple[str, str, str, bool], set[str]] = {}
    checks: List[Dict[str, Any]] = []

    for spec in specs:
        path = str(spec.hf_path)
        revision = str(spec.hf_revision)
        repo_key = (path, revision, spec.hf_trust_remote_code)
        if repo_key not in available_cache:
            try:
                available_cache[repo_key] = set(
                    config_names_fn(path, revision, spec.hf_trust_remote_code)
                )
            except Exception as exc:  # noqa: BLE001 - include remote error context
                raise RuntimeError(
                    f"cannot query configs for {path}@{revision}: {type(exc).__name__}: {exc}"
                ) from exc

        requested = set(spec.hf_config_names)
        missing = sorted(requested - available_cache[repo_key])
        if missing:
            raise SchemaError(
                f"source {spec.name}: {path}@{revision} is missing declared config(s) {missing}; "
                f"available={sorted(available_cache[repo_key])}"
            )
        if spec.converter == "ceval":
            leaked = sorted(requested & MEDICAL_CEVAL_SUBJECTS)
            if leaked:
                raise SchemaError(f"source {spec.name}: medical C-Eval configs in general pool: {leaked}")

        for config_name in spec.hf_config_names:
            key = (path, config_name, revision, spec.hf_trust_remote_code)
            if key not in split_cache:
                try:
                    split_cache[key] = set(
                        split_names_fn(path, config_name, revision, spec.hf_trust_remote_code)
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"cannot query splits for {path}/{config_name}@{revision}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            if spec.hf_split not in split_cache[key]:
                raise SchemaError(
                    f"source {spec.name}: split {spec.hf_split!r} does not exist for "
                    f"{path}/{config_name}; available={sorted(split_cache[key])}"
                )

        checks.append(
            {
                "source": spec.name,
                "hf_path": path,
                "revision": revision,
                "trust_remote_code": spec.hf_trust_remote_code,
                "configs": list(spec.hf_config_names),
                "config_count": len(spec.hf_config_names),
                "split": spec.hf_split,
                "status": "PASS",
            }
        )

    return {
        "config_path": str(config_path),
        "status": "PASS",
        "repositories": [
            {"hf_path": path, "revision": revision, "trust_remote_code": trust}
            for path, revision, trust in sorted(available_cache)
        ],
        "source_checks": checks,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI/network
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/base.yaml")
    parser.add_argument("--output", help="optional JSON evidence path")
    args = parser.parse_args(argv)
    try:
        report = verify_hf_metadata(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
