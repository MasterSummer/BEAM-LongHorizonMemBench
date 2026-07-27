#!/usr/bin/env bash
# Submit after the matrix plan has been frozen:
#   sbatch --array=0-19%1 --export=ALL,LHMSB_ONLINE_RUN_NAME=<name> \
#     deploy/slurm/online_longhorizon_array.sbatch
# `%1` matches a host dedicated to two 4090s; increase only when independent
# two-GPU allocations are genuinely available.
#SBATCH --job-name=memhorizon-online
#SBATCH --gres=gpu:2
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
# Slurm may expose physical GPU IDs in SLURM_JOB_GPUS while the device cgroup
# remaps the allocation to logical CUDA indices 0,1.  Prefer the cgroup-safe
# CUDA_VISIBLE_DEVICES mapping inside the task.
exec env -u SLURM_JOB_GPUS "${REPO_ROOT}/scripts/run_online_longhorizon.sh" task \
  --task-index "${SLURM_ARRAY_TASK_ID}"
