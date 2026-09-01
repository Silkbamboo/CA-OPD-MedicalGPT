# Environment policy

CA-OPD keeps the historical Qwen2.5 work and the Qwen3 OPD main line in separate environments. Never upgrade the legacy environment in place.

## Files and truth level

| File | Meaning |
|---|---|
| `requirements-legacy.txt` | Observed direct-package snapshot of the retained legacy environment; provenance, not a complete transitive lock. |
| `requirements-opd.txt` | Exact direct pins selected from the documented veRL/vLLM compatibility envelope. |
| `requirements-opd.lock` | A resolved lock installed in the one persistent Qwen3-4B environment; it is CPU/package evidence, not a GPU-kernel result. |

The distinction is intentional: package resolution and `pip check` now pass in the persistent environment, while GPU runtime verification remains false until the 2x RTX 3090 host preflight and 20-step calibration run. The lock does not claim that CUDA kernels, vLLM engine startup, Ray placement or veRL training have passed.

## Candidate matrix

- Python 3.12
- NVIDIA driver compatible with CUDA 12.8; CUDA runtime 12.8; cuDNN >= 9.10
- PyTorch 2.8.0
- Transformers 4.56.2 (Qwen3 requires at least 4.51; vLLM 0.11 requires at least 4.55.2)
- TRL 0.23.0 + PEFT 0.17.1
- veRL 0.8.0 + vLLM 0.11.0 + Ray 2.48.0
- Hugging Face Datasets 3.6.0 (intentional: pinned BigBio `med_qa` still uses a reviewed dataset script; Datasets 4.x removes that loading path)
- flash-attn 2.8.3, installed after PyTorch
- NumPy 1.26.4 (veRL 0.8.0 requires NumPy < 2.0; the earlier 2.2.6 candidate was unsatisfiable)
- Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`

Model and tokenizer must use the same declared revision; local snapshot paths do not replace this provenance field.

Published metadata was checked on 2026-07-30. In particular, veRL 0.6.0 was **rejected** because its `vllm` extra caps vLLM at 0.9.1; veRL 0.8.0 accepts vLLM 0.8.5 through 0.12.0. This avoids a superficially pinned but internally contradictory environment.

## Rebuild the single persistent environment

Use the checked script and one persistent root. It verifies the official prebuilt flash-attn wheel SHA and never compiles CUDA code:

```bash
export CA_OPD_PERSIST_ROOT=/persistent/ca-opd
bash scripts/prepare_qwen3_4b_env.sh
```

Do not run these commands in the current legacy environment. The final direct-pin record includes `flash-attn`,
but its build is deliberately split out so `--no-build-isolation` can use the installed CUDA/PyTorch stack.

## Validation and lock promotion

On the target box, before SFT or paid OPD:

```bash
python scripts/target_env_smoke.py \
  --model Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --require-gpu --require-two-gpus
python scripts/preflight.py \
  --run-config configs/runs/b2_medical_opd_qwen3_1_7b.yaml \
  --strict-env --with-tests
```

The smoke gate must verify package pins, CUDA/BF16, Qwen3 config and tokenizer, TRL `SFTConfig`, PEFT `all-linear`, vLLM LoRA APIs, veRL OPD imports and the real-tokenizer assistant-only mask. It may download the 1.7B tokenizer/config but must not start a paid training run.

After the GPU smoke gate and 20-step run pass, record a new GPU verification report. Do not silently rewrite this lock merely because a run starts:

```bash
python -m pip check
sha256sum env/requirements-opd.lock
```

Commit that lock together with the image digest, smoke report and 20-step run metadata. Any later dependency change requires a new lock and an ADR note because it can affect kernel numerics, tokenizer behavior, throughput and comparability.
