# Reproducibility guide

## Reproducibility levels

This public release supports three distinct levels. Keeping them separate avoids claiming that
restricted datasets and multi-hour GPU runs are bundled with the repository.

1. **Public CPU verification**: algorithmic invariants, synthetic data adapters, leakage gates,
   paired statistics and aggregate-result arithmetic.
2. **Protocol reconstruction**: acquire the pinned upstream data/model assets, rebuild manifests
   and render machine-local configurations.
3. **Full GPU rerun**: execute SFT and three-policy OPD on two 24GB GPUs. This requires external
   weights, datasets and many hours of compute.

## 1. Public CPU verification

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
bash scripts/run_public_checks.sh
```

The suite uses synthetic fixtures only. To verify just the published numerical claims:

```bash
python scripts/verify_public_results.py
```

The release check was executed in the pinned project environment with `313 passed, 1 skipped`.
The skipped test requires a local Qwen3 tokenizer, which is not redistributed.

## 2. Full GPU environment

The recorded stack is pinned in [`env/requirements-opd.txt`](../env/requirements-opd.txt) and
[`env/requirements-opd.lock`](../env/requirements-opd.lock): PyTorch 2.8.0, Transformers
4.56.2, PEFT 0.17.1, TRL 0.23.0, veRL 0.8.0, vLLM 0.11.0 and Ray 2.48.0. The final training
loop uses Transformers/PEFT; veRL/vLLM packages remain pinned because their adapters and
diagnostic paths are tested.

The historical environment also used a CUDA/PyTorch-specific flash-attention wheel. That wheel
is an external asset and is not distributed here.

## 3. Data reconstruction

The adapters and pipeline are under [`src/data`](../src/data). A synthetic smoke build is:

```bash
python -m src.data.build_splits \
  --config configs/data/fixture_cpu.yaml \
  --output-dir outputs/data/fixture-smoke
```

A CPU-only toy OPD loop validates rollout, scoring, routing, update and artifact plumbing:

```bash
python -m src.opd.loop_cli \
  --config configs/opd/dev_cpu.yaml \
  --output-dir outputs/opd-cpu-demo
```

Its controller trajectory is synthetic and must not be reported as model accuracy.

Formal reconstruction must use the upstream revisions and role rules in
[`DATA_PROTOCOL.md`](DATA_PROTOCOL.md). Do not commit the resulting raw or processed files.
MedQA licensing remains unresolved, so obtain it independently and review its terms.

## 4. Recorded GPU protocols

Sanitized protocol snapshots are in [`configs/public`](../configs/public). Paths were replaced
with repository-local `artifacts/...` locations; model weights, LoRA checkpoints, processed
records and label files are deliberately absent. The snapshots preserve the measured
hyperparameters and immutable identities but are not a promise of a one-command rerun without
reconstructing those assets.

Key recorded settings:

| Stage | Main settings |
|---|---|
| SFT-v3 | 2-process DDP, BF16, LoRA r16/alpha32, 600 steps, effective batch 16, max length 2048 |
| B2/IDT/CA | BF16, LoRA r16/alpha32, prompt microbatch 1, accumulation 4, response length 1024 |
| Controller | Transformers direct logits, FP32 log-softmax, deterministic decoding, label isolation |
| P10 | B0 then B2 predictions, combined hash freeze, one label join, 10,000 paired bootstraps |

The source includes the exact state machines and historical package validators. Some production
entry points deliberately require clean Git identities and SHA-bound parent artifacts; a new
rerun must build a new package rather than impersonate the historical run.

## 5. Expected external layout

The public code uses repository-relative `artifacts/` defaults after sanitization. A full rerun
should provide equivalent locations (or adapt the recorded configuration before package freeze):

```text
artifacts/
├── models/Qwen3-4B/
├── data/
│   ├── formal_v2/
│   └── sft_v3/
└── outputs/
    ├── qwen3-4b-medical-sft-v3-.../
    └── qwen3-4b-b2-.../
```

Never reuse the historical result hashes after changing data, model, code or configuration.
Generate a new run identity and report the new evidence separately.

## 6. Evaluation boundary

The repository publishes aggregate Controller and confirmation results. It does not publish
per-question predictions or labels. The final capability was never opened. Reproduction should
preserve prediction-first execution and must not use confirmation/final data for training,
routing, early stopping or hyperparameter selection.
