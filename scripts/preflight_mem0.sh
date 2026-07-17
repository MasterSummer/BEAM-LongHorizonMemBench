#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/mem0_common.sh
source "${SCRIPT_DIR}/lib/mem0_common.sh"

REPO_ROOT="$(mem0_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
DRY_RUN=0
ALLOW_DIRTY=0

usage() {
  cat <<'EOF'
Usage: scripts/preflight_mem0.sh [options]

Start local services and run every Mem0 qualification preflight gate.

Options:
  --data-root PATH  Persistent benchmark root (default: /data/lhmsb)
  --env-file PATH   Compose environment file (default: .env)
  --allow-dirty     Permit a dirty source snapshot for non-formal smoke work
  --dry-run         Print Compose commands without executing them
  -h, --help        Show this help
EOF
}

while (($#)); do
  case "$1" in
    --data-root)
      mem0_require_value "$1" "${2:-}" || exit 2
      DATA_ROOT="$2"
      shift 2
      ;;
    --env-file)
      mem0_require_value "$1" "${2:-}" || exit 2
      ENV_FILE="$2"
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
      mem0_unknown_argument "$1" || exit $?
      ;;
  esac
done

export LHMSB_DATA_ROOT="${DATA_ROOT}"
export LHMSB_LIVE_PREFLIGHT=1

if [[ "${DRY_RUN}" == "1" ]]; then
  mem0_print_command mem0_write_host_manifest "${DATA_ROOT}"
else
  mem0_write_host_manifest "${DATA_ROOT}"
fi

mem0_compose "${REPO_ROOT}" "${ENV_FILE}" up --detach --wait \
  qdrant embedding reranker

COMMAND=(
  run --rm worker
  preflight --dataset /data/lhmsb/datasets/software_mem0_v2
  --config /app/configs/experiments/mem0_controlled_zen.yaml
  --data-root /data/lhmsb
  --json /data/lhmsb/runs/preflight/latest.json
)
if [[ "${ALLOW_DIRTY}" == "1" ]]; then
  COMMAND+=(--allow-dirty)
fi
mem0_compose "${REPO_ROOT}" "${ENV_FILE}" "${COMMAND[@]}"
