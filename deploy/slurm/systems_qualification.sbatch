#!/usr/bin/env bash
#SBATCH --job-name=lhmsb-systems
# Request two generic NVIDIA GPUs.  Set LHMSB_REQUIRE_A100=1 only on a legacy
# partition whose benchmark contract explicitly requires A100 hardware.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/data/lhmsb/logs/slurm-%x-%j.out
#SBATCH --error=/data/lhmsb/logs/slurm-%x-%j.err
set -euo pipefail

REPO_ROOT="${LHMSB_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/scripts/lib/systems_common.sh" ]]; then
  printf 'submit from the repository root or export LHMSB_REPO_ROOT\n' >&2
  exit 2
fi
SCRIPT_DIR="${REPO_ROOT}/scripts"
source "${SCRIPT_DIR}/lib/systems_common.sh"
source "${SCRIPT_DIR}/lib/systems_services.sh"
REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
MODE="${LHMSB_SLURM_MODE:-qualification}"
RUN_NAME="${LHMSB_RUN_NAME:-systems-${SLURM_JOB_ID:-manual}}"
DRY_RUN="${LHMSB_SLURM_DRY_RUN:-0}"

cleanup() {
  local status="$?"
  trap - EXIT INT TERM USR1
  if [[ "${DRY_RUN}" != "1" ]]; then
    systems_stop_all_services "${DATA_ROOT}" || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM USR1

if [[ "${DRY_RUN}" == "1" ]]; then
  systems_print_command systems_select_devices
  systems_print_command systems_acquire_run_lock "${DATA_ROOT}" "${RUN_NAME}"
  systems_print_command systems_start_all_services "${DATA_ROOT}"
  systems_print_command "${SCRIPT_DIR}/preflight_systems.sh" --dry-run \
    --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}"
  if [[ "${MODE}" == "smoke" ]]; then
    systems_print_command "${SCRIPT_DIR}/run_systems_smoke.sh" --dry-run \
      --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}" --run-name "${RUN_NAME}"
  else
    systems_print_command "${SCRIPT_DIR}/run_systems_qualification.sh" --dry-run \
      --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}" --run-name "${RUN_NAME}" \
      --keep-going
  fi
  exit 0
fi

systems_load_env "${ENV_FILE}"
systems_configure_gpus
systems_acquire_run_lock "${DATA_ROOT}" "${RUN_NAME}"
systems_write_host_manifest "${DATA_ROOT}"
export LHMSB_DATA_ROOT="${DATA_ROOT}" LHMSB_REPO_ROOT="${REPO_ROOT}"
export LHMSB_RUN_NAME="${RUN_NAME}"
export LHMSB_SERVICE_INSTANCE="${RUN_NAME}-${SLURM_JOB_ID:-manual}"
"${SCRIPT_DIR}/preflight_systems.sh" --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}"
if [[ "${MODE}" == "smoke" ]]; then
  "${SCRIPT_DIR}/run_systems_smoke.sh" --data-root "${DATA_ROOT}" \
    --env-file "${ENV_FILE}" --run-name "${RUN_NAME}"
else
  "${SCRIPT_DIR}/run_systems_qualification.sh" --data-root "${DATA_ROOT}" \
    --env-file "${ENV_FILE}" --run-name "${RUN_NAME}" --keep-going
fi
