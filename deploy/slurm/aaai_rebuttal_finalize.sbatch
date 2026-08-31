#!/usr/bin/env bash
#SBATCH --job-name=memhorizon-rebuttal-finalize
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/memhorizon-%x-%j.out
#SBATCH --error=/tmp/memhorizon-%x-%j.err
set -euo pipefail

CONTROL_REPO="${LHMSB_CONTROL_REPO:?LHMSB_CONTROL_REPO is required}"
POLICY_REPO="${LHMSB_POLICY_REPO:?LHMSB_POLICY_REPO is required}"
exec "${CONTROL_REPO}/scripts/run_aaai_rebuttal_experiment.sh" finalize \
  --policy-repo "${POLICY_REPO}" \
  --source-plan "${LHMSB_REBUTTAL_SOURCE_PLAN:?LHMSB_REBUTTAL_SOURCE_PLAN is required}" \
  --data-root "${LHMSB_DATA_ROOT:?LHMSB_DATA_ROOT is required}" \
  --run-name "${LHMSB_ONLINE_RUN_NAME:?LHMSB_ONLINE_RUN_NAME is required}" \
  --dataset "${LHMSB_SYSTEM_DATASET:?LHMSB_SYSTEM_DATASET is required}" \
  --env-file "${LHMSB_ENV_FILE:?LHMSB_ENV_FILE is required}"
