#!/usr/bin/env bash
# Submit after the matrix plan has been frozen:
#   sbatch --array=0-19%1 --export=ALL,LHMSB_ONLINE_RUN_NAME=<name> \
#     deploy/slurm/online_longhorizon_array.sbatch
# `%1` keeps service state isolated. One 24-GiB 4090 hosts both small TEI
# models; two-GPU allocations still retain split placement.
#SBATCH --job-name=memhorizon-online
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=/tmp/memhorizon-%x-%A_%a.out
#SBATCH --error=/tmp/memhorizon-%x-%A_%a.err
set -euo pipefail

REPO_ROOT="${LHMSB_REPO_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
if [[ ! -x "${REPO_ROOT}/scripts/run_online_longhorizon.sh" ]]; then
  printf 'submit from a prepared MemHorizon repository checkout\n' >&2
  exit 2
fi
exec "${REPO_ROOT}/scripts/run_online_longhorizon.sh" task \
  --task-index "${SLURM_ARRAY_TASK_ID}"
