#!/usr/bin/env bash
# Run every CPU check for Phase 0: unit tests, the OPD dry-run and the data build.
#
# Why chunked: this dev container has a 2 GiB cgroup memory ceiling
# (/sys/fs/cgroup/memory.max) that is shared with the editor/agent runtime, which
# already occupies ~1.4-1.8 GiB. A single pytest process that imports torch and
# instantiates a model per integration test gets OOM-killed (exit 137) near the
# end of the run. Each group below therefore runs in a fresh interpreter, so peak
# RSS stays under the remaining headroom. On a normal box `pytest -q` works in
# one go; nothing about the tests themselves depends on this split.
#
# Usage: bash scripts/run_cpu_checks.sh [--quick]

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

FAILED=0
declare -a RESULTS=()

run_group() {
  local label="$1"; shift
  local out
  out=$("$@" 2>&1)
  local ec=$?
  local last
  last=$(printf '%s\n' "$out" | tail -1)
  if [[ $ec -eq 0 ]]; then
    RESULTS+=("PASS  ${label}: ${last}")
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
# loop tests are the memory-heavy ones: three fresh processes
run_group "loop: rollout/teacher" python -m pytest tests/test_opd_loop.py -q -k "rollout or teacher or synthetic"
run_group "loop: artifacts"       python -m pytest tests/test_opd_loop.py -q -k "dry_run or metrics_file or kl_safety or router_windows"
run_group "loop: resume/routers"  python -m pytest tests/test_opd_loop.py -q -k "resume or single_teacher or fixed_ratio or early_stop or config"

if [[ $QUICK -eq 0 ]]; then
  run_group "data build (fixtures)" python -m src.data.build_splits \
      --config configs/data/fixture_cpu.yaml --output-dir outputs/data/fixture-check
  run_group "opd cpu dry-run" python -m src.opd.loop_cli \
      --config configs/opd/dev_cpu.yaml --output-dir outputs/opd-cpu-dryrun/latest
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
