#!/usr/bin/env bash
#SBATCH --job-name=lhmsb-report
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail

REPO_ROOT="${LHMSB_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/scripts/lib/systems_common.sh" ]]; then
  printf 'submit from the repository root or export LHMSB_REPO_ROOT\n' >&2
  exit 2
fi
source "${REPO_ROOT}/scripts/lib/systems_common.sh"
REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
RUN_NAME="${LHMSB_RUN_NAME:-}"
DRY_RUN="${LHMSB_SLURM_DRY_RUN:-0}"
if [[ -z "${RUN_NAME}" ]]; then
  printf 'export LHMSB_RUN_NAME before submitting the report job\n' >&2
  exit 2
fi
RUN_DIR="${DATA_ROOT}/runs/systems/${RUN_NAME}"

if [[ "${DRY_RUN}" == "1" ]]; then
  systems_print_command "${DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
    aggregate-systems --run-dir "${RUN_DIR}" --out "${RUN_DIR}/report"
  systems_print_command "${DATA_ROOT}/venvs/core/bin/python" -m lhmsb.qualification \
    validate-systems --report "${RUN_DIR}/report" --json "${RUN_DIR}/validation.json"
  exit 0
fi

[[ -d "${RUN_DIR}" ]] || {
  printf 'run directory does not exist: %s\n' "${RUN_DIR}" >&2
  exit 2
}
export LHMSB_DATA_ROOT="${DATA_ROOT}" LHMSB_REPO_ROOT="${REPO_ROOT}"
systems_run_cli "${DATA_ROOT}" core aggregate-systems --run-dir "${RUN_DIR}" \
  --out "${RUN_DIR}/report"
systems_run_cli "${DATA_ROOT}" core validate-systems --report "${RUN_DIR}/report" \
  --json "${RUN_DIR}/validation.json"
