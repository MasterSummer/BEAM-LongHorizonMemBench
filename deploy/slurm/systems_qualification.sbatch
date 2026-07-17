#!/usr/bin/env bash
#SBATCH --job-name=lhmsb-systems
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/data/lhmsb/logs/slurm-%x-%j.out
#SBATCH --error=/data/lhmsb/logs/slurm-%x-%j.err
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts"
# shellcheck source=scripts/lib/systems_common.sh
source "${SCRIPT_DIR}/lib/systems_common.sh"

REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
MODE="${LHMSB_SLURM_MODE:-qualification}"
RUN_NAME="${LHMSB_RUN_NAME:-systems-${SLURM_JOB_ID:-manual}}"
export LHMSB_COMPOSE_PROJECT="${LHMSB_COMPOSE_PROJECT:-lhmsb-${SLURM_JOB_ID:-manual}}"
export LHMSB_SYSTEM_NAMESPACE="${LHMSB_SYSTEM_NAMESPACE:-${LHMSB_COMPOSE_PROJECT}}"

cleanup() {
  local status="$?"
  trap - EXIT INT TERM USR1
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT}" \
      --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
      down --remove-orphans
  else
    systems_compose "${REPO_ROOT}" "${ENV_FILE}" down --remove-orphans || true
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM USR1

if [[ "${MODE}" == "smoke" ]]; then
  RUN_SCRIPT="${SCRIPT_DIR}/run_systems_smoke.sh"
else
  RUN_SCRIPT="${SCRIPT_DIR}/run_systems_qualification.sh"
fi

if [[ "${LHMSB_SLURM_DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN=1
  systems_print_command systems_configure_gpus
  systems_print_command systems_acquire_slurm_lock "${DATA_ROOT}"
  systems_print_command "${SCRIPT_DIR}/preflight_systems.sh" --dry-run \
    --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}"
  systems_print_command "${RUN_SCRIPT}" --dry-run \
    --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}" --run-name "${RUN_NAME}"
  exit 0
fi

command -v nvidia-smi >/dev/null
command -v docker >/dev/null
systems_configure_gpus
systems_acquire_slurm_lock "${DATA_ROOT}"
systems_write_host_manifest "${DATA_ROOT}"
systems_restore_archived_images "${DATA_ROOT}"
systems_verify_runtime_images "${DATA_ROOT}"
export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_REPO_ROOT="${REPO_ROOT}"
export LHMSB_LIVE_PREFLIGHT=1
"${SCRIPT_DIR}/preflight_systems.sh" \
  --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}"
"${RUN_SCRIPT}" \
  --data-root "${DATA_ROOT}" --env-file "${ENV_FILE}" --run-name "${RUN_NAME}" \
  --keep-going

