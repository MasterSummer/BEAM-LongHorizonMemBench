#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/systems_common.sh
source "${SCRIPT_DIR}/lib/systems_common.sh"

REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
RUN_NAME="${LHMSB_RUN_NAME:-systems-qualification}"
DATASET="${LHMSB_SYSTEM_DATASET:-${DATA_ROOT}/datasets/software_v2}"
CONFIG="${REPO_ROOT}/configs/experiments/systems_controlled_zen.yaml"
DRY_RUN=0
FORCE=0
KEEP_GOING=0

usage() {
  cat <<'EOF'
Usage: scripts/run_systems_qualification.sh [options]

Run the frozen 16-session multisystem qualification matrix.

Options:
  --data-root PATH  persistent root (default: /data/lhmsb)
  --env-file PATH   Compose env file (default: .env)
  --dataset PATH    frozen 16-session schema-v2 dataset
  --config PATH     schema-v2 experiment config
  --run-name NAME   run name (default: systems-qualification)
  --force           replace a conflicting run identity
  --keep-going      continue independent tasks after a failed cell
  --dry-run         print commands without Docker, GPU, or secrets
  -h, --help        show this help
EOF
}

while (($#)); do
  case "$1" in
    --data-root)
      systems_require_value "$1" "${2:-}" || exit 2
      DATA_ROOT="$2"
      DATASET="${DATA_ROOT}/datasets/software_v2"
      shift 2
      ;;
    --env-file)
      systems_require_value "$1" "${2:-}" || exit 2
      ENV_FILE="$2"
      shift 2
      ;;
    --dataset)
      systems_require_value "$1" "${2:-}" || exit 2
      DATASET="$2"
      shift 2
      ;;
    --config)
      systems_require_value "$1" "${2:-}" || exit 2
      CONFIG="$2"
      shift 2
      ;;
    --run-name)
      systems_require_value "$1" "${2:-}" || exit 2
      RUN_NAME="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      systems_unknown_argument "$1" || exit $?
      ;;
  esac
done

RUN_DIR="/data/lhmsb/runs/systems/${RUN_NAME}"
if [[ "${DRY_RUN}" == "1" ]]; then
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    up --detach --wait qdrant neo4j embedding reranker
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    run --rm core-worker plan-systems --dataset /data/lhmsb/datasets/software_v2 \
    --config /app/configs/experiments/systems_controlled_zen.yaml --out "${RUN_DIR}"
  for pair in "core-worker 0" "mem0-worker 1" "amem-worker 2" "memos-worker 3"; do
    read -r service task_index <<<"${pair}"
    systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
      --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
      run --rm "${service}" prepare-task --run-dir "${RUN_DIR}" --task-index "${task_index}"
  done
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    run --rm core-worker finalize-evaluation-plan --run-dir "${RUN_DIR}"
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    run --rm core-worker run-evaluation-matrix --run-dir "${RUN_DIR}" --keep-going
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    run --rm core-worker aggregate-systems --run-dir "${RUN_DIR}" --out "${RUN_DIR}/report"
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    run --rm core-worker validate-systems --report "${RUN_DIR}/report" \
    --json "${RUN_DIR}/validation.json"
  exit 0
fi

command -v docker >/dev/null
if [[ ! -f "${ENV_FILE}" ]]; then
  printf 'missing env file: %s\n' "${ENV_FILE}" >&2
  exit 1
fi
chmod 600 "${ENV_FILE}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
systems_configure_gpus
systems_require_live_secrets
systems_restore_archived_images "${DATA_ROOT}"
systems_verify_runtime_images "${DATA_ROOT}"
systems_acquire_slurm_lock "${DATA_ROOT}"
export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_REPO_ROOT="${REPO_ROOT}"
export LHMSB_LIVE_QUALIFICATION=1
systems_compose "${REPO_ROOT}" "${ENV_FILE}" up --detach --wait \
  qdrant neo4j embedding reranker

PLAN_COMMAND=(
  run --rm core-worker plan-systems
  --dataset /data/lhmsb/datasets/software_v2
  --config /app/configs/experiments/systems_controlled_zen.yaml
  --out "${RUN_DIR}"
)
if [[ "${FORCE}" == "1" ]]; then
  PLAN_COMMAND+=(--force)
fi
systems_compose "${REPO_ROOT}" "${ENV_FILE}" "${PLAN_COMMAND[@]}"

for pair in "core-worker 0" "mem0-worker 1" "amem-worker 2" "memos-worker 3"; do
  read -r service task_index <<<"${pair}"
  systems_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm "${service}" prepare-task \
    --run-dir "${RUN_DIR}" --task-index "${task_index}"
done
systems_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm core-worker \
  finalize-evaluation-plan --run-dir "${RUN_DIR}"

MATRIX_COMMAND=(run --rm core-worker run-evaluation-matrix --run-dir "${RUN_DIR}")
if [[ "${KEEP_GOING}" == "1" ]]; then
  MATRIX_COMMAND+=(--keep-going)
fi
systems_compose "${REPO_ROOT}" "${ENV_FILE}" "${MATRIX_COMMAND[@]}"
systems_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm core-worker \
  aggregate-systems --run-dir "${RUN_DIR}" --out "${RUN_DIR}/report"
systems_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm core-worker \
  validate-systems --report "${RUN_DIR}/report" \
  --json "${RUN_DIR}/validation.json"
