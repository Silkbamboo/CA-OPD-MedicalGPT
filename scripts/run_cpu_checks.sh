#!/usr/bin/env bash
# Run every CPU check for Phase 0/P1: unit tests, the OPD dry-run, the data build
# and the paid-run preflight gate.
#
# Memory note (measured on this container, 2026-07-29):
#   /sys/fs/cgroup/memory.max = 2 GiB, shared with the editor/agent runtime
#   `import torch` alone            -> 374 MiB RSS
#   `src.opd.loop_cli` 8-step run   -> 654 MiB peak RSS
#   ambient usage while the agent is active leaves ~420-470 MiB
# So a torch-importing group needs the box to be reasonably idle. Each group runs
# in its own interpreter, waits for ~750 MiB of headroom (NEED_MB), and retries
# once if it is OOM-killed; a persistent kill is reported as `OOM`, not `FAIL`, so
# an environment limit is never mistaken for a broken test. If groups report OOM,
# run the script again with the editor/agent idle, or run a single group directly:
#
#   python -m pytest tests/test_opd_loop.py -q -k resume
#
# On a machine without this ceiling, plain `pytest -q` covers everything in one go.
#
# Usage: bash scripts/run_cpu_checks.sh [--quick]     # --quick skips the slow gates
#        NEED_MB=500 bash scripts/run_cpu_checks.sh   # lower the headroom bar

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# cap glibc malloc arenas: cuts per-process virtual/anon overhead under the 2 GiB cgroup
export MALLOC_ARENA_MAX=2
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

FAILED=0
declare -a RESULTS=()

# Available memory inside the cgroup, in MiB. The editor/agent runtime shares this
# 2 GiB budget, so a group can be OOM-killed (exit 137) purely because of ambient
# pressure - the same command passes standalone. We therefore wait for headroom
# before each group and retry once.
available_mb() {
  local cur max
  cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
  max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)
  if [[ "$max" == "max" || "$max" -eq 0 ]]; then echo 100000; return; fi
  echo $(( (max - cur) / 1048576 ))
}

# Measured peaks: the torch-importing groups need ~650 MiB (opd cpu dry-run
# measured at 654 MiB RSS), so wait for ~750 MiB before starting one.
NEED_MB=${NEED_MB:-400}   # reachable on this box; groups that need more retry once and report OOM

wait_for_headroom() {
  local need=${1:-$NEED_MB} waited=0
  while [[ $(available_mb) -lt $need && $waited -lt 150 ]]; do
    sync
    sleep 5
    waited=$((waited + 5))
  done
}

run_group() {
  local label="$1"; shift
  local out ec last attempt=1
  while : ; do
    wait_for_headroom
    out=$("$@" 2>&1)
    ec=$?
    [[ $ec -ne 137 || $attempt -ge 2 ]] && break
    attempt=$((attempt + 1))
    sleep 8
  done
  last=$(printf '%s\n' "$out" | tail -1)
  if [[ $ec -eq 0 ]]; then
    RESULTS+=("PASS  ${label}: ${last}")
  elif [[ $ec -eq 137 ]]; then
    FAILED=1
    RESULTS+=("OOM   ${label}: killed by the container memory limit (avail $(available_mb) MiB) - rerun this group standalone")
  else
    FAILED=1
    RESULTS+=("FAIL  ${label} (exit ${ec}): ${last}")
    printf '%s\n' "$out" | tail -40
  fi
}

echo "=== CA-OPD CPU checks ==============================================="
echo "memory.current=$(( $(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0) / 1048576 )) MB" \
     "memory.max=$(( $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0) / 1048576 )) MB"
echo

run_group "opd math"        python -m pytest tests/test_opd_math.py -q
run_group "router"          python -m pytest tests/test_router.py -q
run_group "data splits"     python -m pytest tests/test_data_splits.py -q
run_group "chat template"   python -m pytest tests/test_chat_template.py -q
run_group "eval"            python -m pytest tests/test_eval.py -q
run_group "sft (dry-run path)"   python -m pytest tests/test_sft.py -q
run_group "run plan + veRL cfg"  python -m pytest tests/test_run_plan_and_verl_config.py -q
# loop tests are the memory-heavy ones: three fresh processes
run_group "loop: rollout/teacher" python -m pytest tests/test_opd_loop.py -q -k "rollout or teacher or synthetic"
run_group "loop: artifacts"       python -m pytest tests/test_opd_loop.py -q -k "dry_run or metrics_file or kl_safety or router_windows"
run_group "loop: resume/routers"  python -m pytest tests/test_opd_loop.py -q -k "resume or single_teacher or fixed_ratio or early_stop or config"

if [[ $QUICK -eq 0 ]]; then
  run_group "data build (fixtures)" python -m src.data.build_splits \
      --config configs/data/fixture_cpu.yaml --output-dir outputs/data/fixture-check
  run_group "opd cpu dry-run" python -m src.opd.loop_cli \
      --config configs/opd/dev_cpu.yaml --output-dir outputs/opd-cpu-dryrun/latest
  run_group "preflight: B2" python scripts/preflight.py \
      --run-config configs/runs/b2_medical_opd_qwen3_1_7b.yaml --emit-plan outputs/plans
  run_group "preflight: O1" python scripts/preflight.py \
      --run-config configs/runs/o1_ca_opd_qwen3_1_7b.yaml --emit-plan outputs/plans
fi

echo
echo "=== summary =========================================================="
for line in "${RESULTS[@]}"; do echo "$line"; done
echo
if [[ $FAILED -eq 0 ]]; then
  echo "ALL CPU CHECKS PASSED"
else
  echo "SOME CPU CHECKS FAILED"
fi
exit $FAILED
