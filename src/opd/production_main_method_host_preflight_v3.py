"""Hardware/environment preflight for the immutable P6 IDT/CA packages."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

from src.opd.production_main_method_package_v3 import verify_method_package_v3
from src.opd.production_main_method_v3 import P6FormalMethodError


def _gpu_rows() -> list[tuple[str, int, int]]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    rows: list[tuple[str, int, int]] = []
    for line in process.stdout.splitlines():
        name, total, used = (part.strip() for part in line.split(","))
        rows.append((name, int(total), int(used)))
    return rows


def host_preflight_v3(package: Path) -> dict[str, Any]:
    if os.environ.get("CA_OPD_P6_METHOD_CONFIRM") not in {"IDT", "CA-OPD"}:
        raise P6FormalMethodError("P6 formal method host authorization is absent")
    package_gate = verify_method_package_v3(package, require_clean_git=True)
    rows = _gpu_rows()
    if not (
        len(rows) == 2
        and all("RTX 3090" in name and total >= 24576 and used <= 64 for name, total, used in rows)
    ):
        raise P6FormalMethodError("P6 formal method requires two idle RTX 3090 GPUs")
    import torch

    if not (
        torch.cuda.is_available()
        and torch.cuda.device_count() == 2
        and all(torch.cuda.is_bf16_supported(index) for index in range(2))
    ):
        raise P6FormalMethodError("P6 formal method CUDA/BF16 host differs")
    config = json.loads((Path(package) / "formal_method_config.json").read_text(encoding="utf-8"))
    model = Path(str(config["model"]["base_path"]))
    required = (
        model / "config.json",
        model / "tokenizer.json",
        model / "model.safetensors.index.json",
    )
    if not model.is_dir() or not all(path.is_file() for path in required):
        raise P6FormalMethodError("P6 formal method Base artifact is incomplete")
    shm_free = shutil.disk_usage("/dev/shm").free
    persist_free = shutil.disk_usage(Path(str(config["run"]["output_dir"])).parent).free
    if shm_free < 8 * 1024**3:
        raise P6FormalMethodError("P6 formal method /dev/shm free space is below 8 GiB")
    price_value = os.environ.get("CA_OPD_LIVE_PRICE_CNY_PER_HOUR")
    live_price = None
    if price_value not in {None, "", "null"}:
        live_price = float(price_value)
        if live_price <= 0:
            raise P6FormalMethodError("P6 live price, when available, must be positive")
    return {
        "schema_version": 3,
        "artifact_kind": "p6_formal_method_host_preflight_v3",
        "passed": True,
        "method_id": package_gate["method_id"],
        "package_content_sha256": package_gate["package_content_sha256"],
        "gpu_rows": [
            {"name": name, "memory_total_mib": total, "memory_used_mib": used}
            for name, total, used in rows
        ],
        "torch": torch.__version__,
        "transformers": metadata.version("transformers"),
        "peft": metadata.version("peft"),
        "vllm": metadata.version("vllm"),
        "persistent_free_bytes": persist_free,
        "shm_free_bytes": shm_free,
        "live_price_cny_per_hour": live_price,
        "platform_actual_cost_cny": None,
        "global_runtime_hard_limit": None,
        "final_access_count": 0,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(host_preflight_v3(args.package), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["host_preflight_v3"]
