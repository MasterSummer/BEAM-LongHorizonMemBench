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
COMPOSE_FILE="${REPO_ROOT}/deploy/compose.mem0.yaml"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
CONFIG_PATH="/app/configs/experiments/mem0_qualification.yaml"
DATASET_PATH="/data/lhmsb/datasets/software_mem0_v2"
RUN_DIR="/data/lhmsb/runs/mem0/${LHMSB_RUN_NAME:-qualification}"

# The #SBATCH line is the portable default. More GPUs may be requested with:
# sbatch --gres="${LHMSB_SLURM_GRES:-gpu:a100:2}" deploy/slurm/mem0_qualification.sbatch
export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_LIVE_QUALIFICATION=1

mkdir -p "${DATA_ROOT}/runs/mem0"

srun --ntasks=1 docker compose \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  up --detach --wait qdrant embedding reranker

srun --ntasks=1 docker compose \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  run --rm worker \
  plan --dataset "${DATASET_PATH}" \
  --config "${CONFIG_PATH}" \
  --out "${RUN_DIR}"

srun --ntasks=1 docker compose \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  run --rm worker \
  run-matrix --run-dir "${RUN_DIR}" \
  --json "${RUN_DIR}/matrix-status.json"
