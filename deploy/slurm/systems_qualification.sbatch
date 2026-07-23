#!/usr/bin/env bash
#SBATCH --job-name=lhmsb-systems
# Request two generic NVIDIA GPUs.  Set LHMSB_REQUIRE_A100=1 only on a legacy
# partition whose benchmark contract explicitly requires A100 hardware.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@120
# Keep default scheduler logs outside the checkout so formal runs remain clean.
# Production submissions may override these paths with ``sbatch --output/--error``.
#SBATCH --output=/tmp/lhmsb-%x-%j.out
#SBATCH --error=/tmp/lhmsb-%x-%j.err
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
EPISODE_LIMIT="${LHMSB_EPISODE_LIMIT:-}"
ANALYSIS_PHASE="${LHMSB_ANALYSIS_PHASE:-development}"
case "${ANALYSIS_PHASE}" in
  development|diagnostic|calibration|confirmatory) ;;
  *)
    printf 'unknown LHMSB_ANALYSIS_PHASE: %s\n' "${ANALYSIS_PHASE}" >&2
    exit 2
    ;;
esac
RUN_ARGS=(--analysis-phase "${ANALYSIS_PHASE}")
if [[ -n "${EPISODE_LIMIT}" ]]; then
  [[ "${EPISODE_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'LHMSB_EPISODE_LIMIT must be a positive integer\n' >&2
    exit 2
  }
  RUN_ARGS+=(--episode-limit "${EPISODE_LIMIT}")
fi

case "${MODE}" in
  smoke|prepare|qualification) ;;
  *)
    printf 'unknown LHMSB_SLURM_MODE: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac

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
  systems_print_command systems_start_all_services "${DATA_ROOT}"
  systems_print_command "${SCRIPT_DIR}/preflight_systems.sh" --dry-run \
    --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}"
  if [[ "${MODE}" == "smoke" ]]; then
    systems_print_command "${SCRIPT_DIR}/run_systems_smoke.sh" --dry-run \
      --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}" --run-name "${RUN_NAME}"
  elif [[ "${MODE}" == "prepare" ]]; then
    systems_print_command "${SCRIPT_DIR}/run_systems_qualification.sh" --dry-run \
      --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}" --run-name "${RUN_NAME}" \
      --prepare-only "${RUN_ARGS[@]}"
  else
    systems_print_command "${SCRIPT_DIR}/run_systems_qualification.sh" --dry-run \
      --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}" --run-name "${RUN_NAME}" \
      --keep-going "${RUN_ARGS[@]}"
  fi
  exit 0
fi

systems_load_env "${ENV_FILE}"
systems_configure_gpus
systems_write_host_manifest "${DATA_ROOT}"
export LHMSB_DATA_ROOT="${DATA_ROOT}" LHMSB_REPO_ROOT="${REPO_ROOT}"
export LHMSB_RUN_NAME="${RUN_NAME}"
export LHMSB_SERVICE_INSTANCE="${RUN_NAME}-${SLURM_JOB_ID:-manual}"
"${SCRIPT_DIR}/preflight_systems.sh" --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}"
if [[ "${MODE}" == "smoke" ]]; then
  "${SCRIPT_DIR}/run_systems_smoke.sh" --data-root "${DATA_ROOT}" \
    --env-file "${ENV_FILE}" --run-name "${RUN_NAME}"
elif [[ "${MODE}" == "prepare" ]]; then
  "${SCRIPT_DIR}/run_systems_qualification.sh" --data-root "${DATA_ROOT}" \
    --env-file "${ENV_FILE}" --run-name "${RUN_NAME}" --prepare-only \
    "${RUN_ARGS[@]}"
else
  "${SCRIPT_DIR}/run_systems_qualification.sh" --data-root "${DATA_ROOT}" \
    --env-file "${ENV_FILE}" --run-name "${RUN_NAME}" --keep-going \
    "${RUN_ARGS[@]}"
fi
