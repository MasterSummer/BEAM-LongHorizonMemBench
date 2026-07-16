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
COMPOSE_FILE="${REPO_ROOT}/deploy/compose.mem0.yaml"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
CONFIG_PATH="/app/configs/experiments/mem0_qualification.yaml"
DATASET_PATH="/data/lhmsb/datasets/software_mem0_v2"
REPORT_PATH="/data/lhmsb/runs/preflight/preflight-${SLURM_JOB_ID:-manual}.json"

# Override at submission time when the cluster uses a different GRES label:
# sbatch --gres="${LHMSB_SLURM_GRES:-gpu:a100:2}" deploy/slurm/mem0_preflight.sbatch
export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_LIVE_PREFLIGHT=1

mkdir -p "${DATA_ROOT}/runs/preflight"

srun --ntasks=1 docker compose \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  up --detach --wait qdrant embedding reranker

srun --ntasks=1 docker compose \
  --env-file "${ENV_FILE}" \
  -f "${COMPOSE_FILE}" \
  run --rm worker \
  preflight --dataset "${DATASET_PATH}" \
  --config "${CONFIG_PATH}" \
  --data-root /data/lhmsb \
  --json "${REPORT_PATH}"
