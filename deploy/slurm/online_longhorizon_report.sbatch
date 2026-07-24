#!/usr/bin/env bash
# Submit with an afterok dependency on the online array:
#   sbatch --dependency=afterok:<array_job_id> \
#     --export=ALL,LHMSB_ONLINE_RUN_NAME=<name> \
#     deploy/slurm/online_longhorizon_report.sbatch
#SBATCH --job-name=memhorizon-report
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/memhorizon-%x-%j.out
#SBATCH --error=/tmp/memhorizon-%x-%j.err
set -euo pipefail

REPO_ROOT="${LHMSB_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
if [[ ! -x "${REPO_ROOT}/scripts/run_online_longhorizon.sh" ]]; then
  printf 'submit from a prepared MemHorizon repository checkout\n' >&2
  exit 2
fi
exec "${REPO_ROOT}/scripts/run_online_longhorizon.sh" aggregate
