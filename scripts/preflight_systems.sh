#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/systems_common.sh
source "${SCRIPT_DIR}/lib/systems_common.sh"
# shellcheck source=scripts/lib/systems_services.sh
source "${SCRIPT_DIR}/lib/systems_services.sh"

REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
DATASET="${LHMSB_SYSTEM_DATASET:-${DATA_ROOT}/datasets/software_v3}"
CONFIG="${REPO_ROOT}/configs/experiments/systems_controlled_gpt_only.yaml"
DRY_RUN=0
ALLOW_DIRTY=0

usage() {
  cat <<'EOF'
Usage: scripts/preflight_systems.sh [options]

Run repository, native runtime, model, and service gates.

Options:
  --data-root PATH  persistent root (default: /data/lhmsb)
  --env-file PATH   operator-owned settings file (default: .env)
  --dataset PATH    frozen schema-v2 dataset
  --config PATH     schema-v2 experiment config
  --allow-dirty     allow a non-formal dirty checkout
  --dry-run         print commands without network, GPU, secrets, or writes
  -h, --help        show this help
EOF
}

while (($#)); do
  case "$1" in
    --data-root) systems_require_value "$1" "${2:-}" || exit 2; DATA_ROOT="$2"; DATASET="${DATA_ROOT}/datasets/software_v3"; shift 2 ;;
    --env-file) systems_require_value "$1" "${2:-}" || exit 2; ENV_FILE="$2"; shift 2 ;;
    --dataset) systems_require_value "$1" "${2:-}" || exit 2; DATASET="$2"; shift 2 ;;
    --config) systems_require_value "$1" "${2:-}" || exit 2; CONFIG="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) systems_unknown_argument "$1" || exit $? ;;
  esac
done

if [[ "${DRY_RUN}" == "1" ]]; then
  systems_print_command "${SCRIPT_DIR}/verify_system_runtime.sh" --dry-run --data-root "${DATA_ROOT}"
  systems_print_command systems_configure_gpus
  systems_print_command systems_start_all_services "${DATA_ROOT}"
  systems_print_command "${DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
    preflight-systems --repository-only --dataset "${DATASET}" --config "${CONFIG}" \
    --data-root "${DATA_ROOT}"
  exit 0
fi

systems_load_env "${ENV_FILE}"
systems_configure_gpus
systems_require_live_secrets
"${SCRIPT_DIR}/verify_system_runtime.sh" --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}"
systems_write_host_manifest "${DATA_ROOT}"
export LHMSB_DATA_ROOT="${DATA_ROOT}" LHMSB_REPO_ROOT="${REPO_ROOT}"
export LHMSB_SERVICE_INSTANCE="preflight-${SLURM_JOB_ID:-$$}"
systems_start_all_services "${DATA_ROOT}"
cleanup() {
  local status="$?"
  trap - EXIT INT TERM
  systems_stop_all_services "${DATA_ROOT}" || true
  exit "${status}"
}
trap cleanup EXIT INT TERM
systems_write_runtime_env "${DATA_ROOT}"

COMMAND=(
  preflight-systems
  --repository-only
  --dataset "${DATASET}"
  --config "${CONFIG}"
  --data-root "${DATA_ROOT}"
  --json "${DATA_ROOT}/runs/preflight-systems/latest.json"
)
if [[ "${ALLOW_DIRTY}" == "1" ]]; then
  COMMAND+=(--allow-dirty)
fi
systems_run_cli "${DATA_ROOT}" core "${COMMAND[@]}"
