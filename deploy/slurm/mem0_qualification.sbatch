#!/usr/bin/env bash
#SBATCH --job-name=lhmsb-mem0-qualification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --output=slurm-mem0-qualification-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/mem0_common.sh
source "${REPO_ROOT}/scripts/lib/mem0_common.sh"
COMPOSE_FILE="${REPO_ROOT}/deploy/compose.mem0.yaml"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
CONFIG_PATH="/app/configs/experiments/mem0_controlled_zen.yaml"
DATASET_PATH="/data/lhmsb/datasets/software_mem0_v2"
RUN_DIR="/data/lhmsb/runs/mem0/${LHMSB_RUN_NAME:-qualification}"
PREFLIGHT_REPORT="/data/lhmsb/runs/preflight/latest.json"
PREFLIGHT_ARCHIVE="${DATA_ROOT}/runs/preflight/preflight-${SLURM_JOB_ID:-manual}.json"
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

# The #SBATCH line is the portable default. More GPUs may be requested with:
# sbatch --gres="${LHMSB_SLURM_GRES:-gpu:a100:2}" deploy/slurm/mem0_qualification.sbatch
export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_LIVE_PREFLIGHT=1
export LHMSB_LIVE_QUALIFICATION=1
export COMPOSE_PROJECT_NAME
export LHMSB_QDRANT_NAMESPACE="slurm-${JOB_KEY}"

mkdir -p "${DATA_ROOT}/runs/mem0" "${DATA_ROOT}/runs/preflight"
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
  --json "${PREFLIGHT_REPORT}"

cp "${DATA_ROOT}/runs/preflight/latest.json" "${PREFLIGHT_ARCHIVE}"

srun --ntasks=1 docker compose \
  --project-name "${COMPOSE_PROJECT_NAME}" \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  run --rm worker \
  plan --dataset "${DATASET_PATH}" \
  --config "${CONFIG_PATH}" \
  --out "${RUN_DIR}"

srun --ntasks=1 docker compose \
  --project-name "${COMPOSE_PROJECT_NAME}" \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  run --rm worker \
  run-matrix --run-dir "${RUN_DIR}" \
  --keep-going \
  --json "${RUN_DIR}/matrix-status.json"

srun --ntasks=1 docker compose \
  --project-name "${COMPOSE_PROJECT_NAME}" \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  run --rm worker \
  validate --report "${RUN_DIR}/report" \
  --json "${RUN_DIR}/validation.json"
