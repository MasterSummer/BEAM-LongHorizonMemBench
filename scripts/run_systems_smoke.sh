#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/systems_common.sh"
source "${SCRIPT_DIR}/lib/systems_services.sh"
REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
RUN_NAME="${LHMSB_SMOKE_RUN_NAME:-systems-smoke}"
DATASET="${LHMSB_SYSTEM_DATASET:-}"
CONFIG="${LHMSB_SYSTEM_CONFIG:-${REPO_ROOT}/configs/experiments/systems_controlled_gpt_only_aaai.yaml}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_systems_smoke.sh [options]

Run one full-horizon episode from the frozen native multisystem dataset.

Options:
  --data-root PATH  persistent root (default: /data/lhmsb)
  --env-file PATH   operator-owned settings file (default: .env)
  --dataset PATH    frozen schema-v2 dataset
  --config PATH     schema-v2 experiment config
  --run-name NAME   run name (default: systems-smoke)
  --dry-run         print commands without network, GPU, secrets, or writes
  -h, --help        show this help
EOF
}

while (($#)); do
  case "$1" in
    --data-root) systems_require_value "$1" "${2:-}" || exit 2; DATA_ROOT="$2"; shift 2 ;;
    --env-file) systems_require_value "$1" "${2:-}" || exit 2; ENV_FILE="$2"; shift 2 ;;
    --dataset) systems_require_value "$1" "${2:-}" || exit 2; DATASET="$2"; shift 2 ;;
    --config) systems_require_value "$1" "${2:-}" || exit 2; CONFIG="$2"; shift 2 ;;
    --run-name) systems_require_value "$1" "${2:-}" || exit 2; RUN_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) systems_unknown_argument "$1" || exit $? ;;
  esac
done

DATASET="${DATASET:-${DATA_ROOT}/datasets/software_v10}"

RUN_DIR="${DATA_ROOT}/runs/systems/${RUN_NAME}"
if [[ "${DRY_RUN}" == "1" ]]; then
  systems_print_command "${SCRIPT_DIR}/verify_system_runtime.sh" --dry-run --data-root "${DATA_ROOT}"
  systems_print_command systems_start_all_services "${DATA_ROOT}"
  systems_print_command "${DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification plan-systems \
    --dataset "${DATASET}" --config "${CONFIG}" --out "${RUN_DIR}" --episode-limit 1 \
    --analysis-phase diagnostic
  for pair in "core 0" "mem0 1" "amem 2" "memos 3"; do
    read -r environment task_index <<<"${pair}"
    systems_print_command "${DATA_ROOT}/venvs/${environment}/bin/python" -m lhmsb.qualification \
      prepare-task --run-dir "${RUN_DIR}" --task-index "${task_index}"
  done
  systems_print_command "${DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
    finalize-evaluation-plan --run-dir "${RUN_DIR}"
  systems_print_command "${DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
    run-evaluation-matrix --run-dir "${RUN_DIR}" --keep-going
  systems_print_command "${DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
    aggregate-systems --run-dir "${RUN_DIR}" --out "${RUN_DIR}/report"
  systems_print_command "${DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
    validate-systems --report "${RUN_DIR}/report" --json "${RUN_DIR}/validation.json"
  exit 0
fi

systems_load_env "${ENV_FILE}"
systems_configure_gpus
systems_require_live_secrets
"${SCRIPT_DIR}/verify_system_runtime.sh" --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}"
systems_acquire_run_lock "${DATA_ROOT}" "${RUN_NAME}"
export LHMSB_DATA_ROOT="${DATA_ROOT}" LHMSB_REPO_ROOT="${REPO_ROOT}"
export LHMSB_SERVICE_INSTANCE="${RUN_NAME}-${SLURM_JOB_ID:-$$}"
systems_start_all_services "${DATA_ROOT}"
cleanup() {
  local status="$?"
  trap - EXIT INT TERM
  systems_stop_all_services "${DATA_ROOT}" || true
  exit "${status}"
}
trap cleanup EXIT INT TERM
systems_write_runtime_env "${DATA_ROOT}"

systems_run_cli "${DATA_ROOT}" core plan-systems --dataset "${DATASET}" \
  --config "${CONFIG}" --out "${RUN_DIR}" --episode-limit 1 \
  --analysis-phase diagnostic
for pair in "core 0" "mem0 1" "amem 2" "memos 3"; do
  read -r environment task_index <<<"${pair}"
  systems_run_cli "${DATA_ROOT}" "${environment}" prepare-task \
    --run-dir "${RUN_DIR}" --task-index "${task_index}"
done
systems_run_cli "${DATA_ROOT}" core finalize-evaluation-plan --run-dir "${RUN_DIR}"
systems_run_cli "${DATA_ROOT}" core run-evaluation-matrix --run-dir "${RUN_DIR}" --keep-going
systems_run_cli "${DATA_ROOT}" core aggregate-systems --run-dir "${RUN_DIR}" --out "${RUN_DIR}/report"
systems_run_cli "${DATA_ROOT}" core validate-systems --report "${RUN_DIR}/report" \
  --json "${RUN_DIR}/validation.json"
