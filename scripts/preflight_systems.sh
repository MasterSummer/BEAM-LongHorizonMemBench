#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/systems_common.sh
source "${SCRIPT_DIR}/lib/systems_common.sh"

REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
DRY_RUN=0
ALLOW_DIRTY=0
DATASET="${LHMSB_SYSTEM_DATASET:-${DATA_ROOT}/datasets/software_v2}"
CONFIG="${REPO_ROOT}/configs/experiments/systems_controlled_zen.yaml"

usage() {
  cat <<'EOF'
Usage: scripts/preflight_systems.sh [options]

Run repository and live service gates for Flat, Mem0, A-MEM, and MemOS-Tree.

Options:
  --data-root PATH  persistent root (default: /data/lhmsb)
  --env-file PATH   Compose env file (default: .env)
  --dataset PATH    frozen schema-v2 dataset
  --config PATH     schema-v2 experiment config
  --allow-dirty     allow a non-formal dirty checkout
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
    --allow-dirty)
      ALLOW_DIRTY=1
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

if [[ "${DRY_RUN}" == "1" ]]; then
  systems_print_command docker load --input "${DATA_ROOT}/images/qdrant.tar"
  systems_print_command docker load --input "${DATA_ROOT}/images/neo4j.tar"
  systems_print_command docker load --input "${DATA_ROOT}/images/tei.tar"
  systems_print_command docker load --input "${DATA_ROOT}/images/core-worker.tar"
  systems_print_command docker load --input "${DATA_ROOT}/images/mem0-worker.tar"
  systems_print_command docker load --input "${DATA_ROOT}/images/amem-worker.tar"
  systems_print_command docker load --input "${DATA_ROOT}/images/memos-worker.tar"
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    up --detach --wait qdrant neo4j embedding reranker
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    run --rm core-worker preflight-systems --repository-only \
    --dataset /data/lhmsb/datasets/software_v2 \
    --config /app/configs/experiments/systems_controlled_zen.yaml \
    --data-root /data/lhmsb --json /data/lhmsb/runs/preflight-systems/latest.json
  exit 0
fi

command -v docker >/dev/null
command -v python3 >/dev/null
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
systems_write_host_manifest "${DATA_ROOT}"
export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_REPO_ROOT="${REPO_ROOT}"
export LHMSB_LIVE_PREFLIGHT=1
systems_compose "${REPO_ROOT}" "${ENV_FILE}" up --detach --wait \
  qdrant neo4j embedding reranker
COMMAND=(
  run --rm core-worker preflight-systems
  --dataset /data/lhmsb/datasets/software_v2
  --config /app/configs/experiments/systems_controlled_zen.yaml
  --data-root /data/lhmsb
  --json /data/lhmsb/runs/preflight-systems/latest.json
)
if [[ "${ALLOW_DIRTY}" == "1" ]]; then
  COMMAND+=(--allow-dirty)
fi
systems_compose "${REPO_ROOT}" "${ENV_FILE}" "${COMMAND[@]}"

