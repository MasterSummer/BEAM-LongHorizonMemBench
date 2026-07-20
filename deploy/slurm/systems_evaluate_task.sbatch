#!/usr/bin/env bash
# One read-only GPT policy task for the prepared native prefix matrix.
# Submit after run_systems_qualification.sh --prepare-only:
#   sbatch --array=0-349%16 --export=ALL,LHMSB_RUN_NAME=<name> \
#     deploy/slurm/systems_evaluate_task.sbatch
# The evaluator consumes frozen prefixes and does not start memory services.
#SBATCH --job-name=lhmsb-gpt-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
set -euo pipefail

REPO_ROOT="${LHMSB_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/scripts/lib/systems_common.sh" ]]; then
  printf 'submit from the repository root or export LHMSB_REPO_ROOT\n' >&2
  exit 2
fi
SCRIPT_DIR="${REPO_ROOT}/scripts"
source "${SCRIPT_DIR}/lib/systems_common.sh"
REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
RUN_NAME="${LHMSB_RUN_NAME:-systems-${SLURM_ARRAY_JOB_ID:-manual}}"
TASK_INDEX="${LHMSB_TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"

[[ -n "${TASK_INDEX}" ]] || {
  printf 'LHMSB_TASK_INDEX or SLURM_ARRAY_TASK_ID is required\n' >&2
  exit 2
}
systems_load_env "${ENV_FILE}"
systems_require_live_secrets
export LHMSB_DATA_ROOT="${DATA_ROOT}" LHMSB_REPO_ROOT="${REPO_ROOT}"
RUN_DIR="${DATA_ROOT}/runs/systems/${RUN_NAME}"
systems_run_cli "${DATA_ROOT}" core evaluate-task \
  --run-dir "${RUN_DIR}" --task-index "${TASK_INDEX}"
