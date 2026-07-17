#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/systems_common.sh
source "${SCRIPT_DIR}/lib/systems_common.sh"

REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/verify_system_images.sh [options]

Restore every pinned OCI archive and verify worker entrypoints/build manifests
before a provider request is allowed.

Options:
  --data-root PATH  persistent root (default: /data/lhmsb)
  --env-file PATH   Compose env file (default: .env)
  --dry-run         print checks without Docker, GPU, or secrets
  -h, --help        show this help
EOF
}

while (($#)); do
  case "$1" in
    --data-root)
      systems_require_value "$1" "${2:-}" || exit 2
      DATA_ROOT="$2"
      shift 2
      ;;
    --env-file)
      systems_require_value "$1" "${2:-}" || exit 2
      ENV_FILE="$2"
      shift 2
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
  systems_print_command systems_restore_archived_images "${DATA_ROOT}"
  systems_print_command systems_verify_runtime_images "${DATA_ROOT}"
  systems_print_command docker run --rm "${LHMSB_CORE_WORKER_IMAGE_DIGEST:-sha256:core}" --help
  systems_print_command docker run --rm "${LHMSB_MEM0_WORKER_IMAGE_DIGEST:-sha256:mem0}" --help
  systems_print_command docker run --rm "${LHMSB_AMEM_WORKER_IMAGE_DIGEST:-sha256:amem}" --help
  systems_print_command docker run --rm "${LHMSB_MEMOS_WORKER_IMAGE_DIGEST:-sha256:memos}" --help
  systems_print_command docker compose --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${ENV_FILE}" -f "${REPO_ROOT}/deploy/compose.systems.yaml" \
    run --rm core-worker verify-system-images \
    --data-root /data/lhmsb --json /data/lhmsb/manifests/image-verification.json
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
systems_restore_archived_images "${DATA_ROOT}"
systems_verify_runtime_images "${DATA_ROOT}"
for image in core-worker mem0-worker amem-worker memos-worker; do
  docker run --rm "lhmsb/${image}:qualification" --help >/dev/null
done
systems_compose "${REPO_ROOT}" "${ENV_FILE}" run --rm core-worker \
  verify-system-images --data-root /data/lhmsb \
  --json /data/lhmsb/manifests/image-verification.json

