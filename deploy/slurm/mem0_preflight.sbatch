#!/usr/bin/env bash
#SBATCH --job-name=lhmsb-mem0-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-mem0-preflight-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/mem0_common.sh
source "${REPO_ROOT}/scripts/lib/mem0_common.sh"
COMPOSE_FILE="${REPO_ROOT}/deploy/compose.mem0.yaml"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
CONFIG_PATH="/app/configs/experiments/mem0_controlled_zen.yaml"
DATASET_PATH="/data/lhmsb/datasets/software_mem0_v2"
REPORT_PATH="/data/lhmsb/runs/preflight/latest.json"
ARCHIVE_PATH="${DATA_ROOT}/runs/preflight/preflight-${SLURM_JOB_ID:-manual}.json"
JOB_KEY="${SLURM_JOB_ID:-manual-$$}"
COMPOSE_PROJECT_NAME="lhmsb-mem0-${JOB_KEY}"

cleanup() {
  docker compose \
    --project-name "${COMPOSE_PROJECT_NAME}" \
    --env-file "${ENV_FILE}" \
    -f "${COMPOSE_FILE}" \
    down --remove-orphans || true
}
trap cleanup EXIT

# Override at submission time when the cluster uses a different GRES label:
# sbatch --gres="${LHMSB_SLURM_GRES:-gpu:a100:2}" deploy/slurm/mem0_preflight.sbatch
export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_LIVE_PREFLIGHT=1
export COMPOSE_PROJECT_NAME
export LHMSB_QDRANT_NAMESPACE="slurm-${JOB_KEY}"

mkdir -p "${DATA_ROOT}/runs/preflight"
mem0_acquire_slurm_lock "${DATA_ROOT}"
mem0_configure_slurm_gpus
mem0_restore_archived_images "${DATA_ROOT}"
mem0_write_host_manifest "${DATA_ROOT}"

srun --ntasks=1 docker compose \
  --project-name "${COMPOSE_PROJECT_NAME}" \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  up --detach --wait qdrant embedding reranker

srun --ntasks=1 docker compose \
  --project-name "${COMPOSE_PROJECT_NAME}" \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  run --rm worker \
  preflight --dataset "${DATASET_PATH}" \
  --config "${CONFIG_PATH}" \
  --data-root /data/lhmsb \
  --json "${REPORT_PATH}"

cp "${DATA_ROOT}/runs/preflight/latest.json" "${ARCHIVE_PATH}"
