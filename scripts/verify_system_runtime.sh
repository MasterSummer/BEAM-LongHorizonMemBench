#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/systems_common.sh"
REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${DATA_ROOT}/env/operator.env}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/verify_system_runtime.sh [options]

Verify Python environments, native executables, models, and generated locks.

Options:
  --data-root PATH  persistent root (default: /data/lhmsb)
  --env-file PATH   operator-owned settings file
  --dry-run         print checks without network, GPU, secrets, or writes
  -h, --help        show this help
EOF
}

while (($#)); do
  case "$1" in
    --data-root) systems_require_value "$1" "${2:-}" || exit 2; DATA_ROOT="$2"; shift 2 ;;
    --env-file) systems_require_value "$1" "${2:-}" || exit 2; ENV_FILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) systems_unknown_argument "$1" || exit $? ;;
  esac
done

if [[ "${DRY_RUN}" == "1" ]]; then
  for environment in core mem0 amem memos; do
    systems_print_command test -x "${DATA_ROOT}/venvs/${environment}/bin/python"
    systems_print_command "${DATA_ROOT}/venvs/${environment}/bin/python" -m lhmsb.qualification --help
  done
  systems_print_command test -x "${DATA_ROOT}/manifests/native-runtime.json"
  systems_print_command test -d "${DATA_ROOT}/models/bge-m3"
  systems_print_command test -d "${DATA_ROOT}/models/bge-reranker-v2-m3"
  systems_print_command "${DATA_ROOT}/bin/qdrant" --version
  systems_print_command "${DATA_ROOT}/bin/text-embeddings-router" --help
  exit 0
fi

systems_load_env "${ENV_FILE}"
systems_assert_lock_contract "${REPO_ROOT}"
for environment in core mem0 amem memos; do
  systems_assert_generated_lock "${DATA_ROOT}" "${environment}"
  python="$(systems_venv_python "${DATA_ROOT}" "${environment}")"
  [[ -x "${python}" ]] || { printf 'missing Python environment: %s\n' "${python}" >&2; exit 1; }
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python}" -c 'import sys; assert sys.version_info[:2] == (3, 11); import lhmsb'
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python}" -m lhmsb.qualification --help >/dev/null
done

for required in "${LHMSB_QDRANT_BIN}" "${LHMSB_NEO4J_HOME}/bin/neo4j" \
  "${LHMSB_NEO4J_HOME}/bin/cypher-shell" "${LHMSB_JAVA_HOME}/bin/java" \
  "${LHMSB_TEI_BIN}" "${LHMSB_EMBEDDING_MODEL_DIR}" "${LHMSB_RERANKER_MODEL_DIR}"; do
  [[ -e "${required}" ]] || { printf 'missing runtime artifact: %s\n' "${required}" >&2; exit 1; }
done

[[ -s "${DATA_ROOT}/manifests/native-runtime.json" ]] || {
  printf 'missing native runtime manifest\n' >&2
  exit 1
}
[[ -s "${DATA_ROOT}/manifests/model-bundle.json" ]] || {
  printf 'missing model bundle manifest\n' >&2
  exit 1
}
qdrant_server_version="$("${LHMSB_QDRANT_BIN}" --version | awk '{print $2}')"
qdrant_locked_version="$(
  awk '
    /^qdrant:/ { in_qdrant = 1; next }
    in_qdrant && /^[^[:space:]]/ { exit }
    in_qdrant && $1 == "version:" {
      gsub(/"/, "", $2)
      print $2
      exit
    }
  ' "${REPO_ROOT}/deploy/native-runtime.lock.yaml"
)"
if [[ -z "${qdrant_locked_version}" ]]; then
  printf 'Qdrant version is missing from deploy/native-runtime.lock.yaml\n' >&2
  exit 1
fi
if [[ "${qdrant_server_version}" != "${qdrant_locked_version}" ]]; then
  printf 'Qdrant server does not match native runtime lock: server=%s lock=%s\n' \
    "${qdrant_server_version}" "${qdrant_locked_version}" >&2
  exit 1
fi
for environment in core mem0; do
  qdrant_client_version="$(
    "$(systems_venv_python "${DATA_ROOT}" "${environment}")" -c \
      'import importlib.metadata; print(importlib.metadata.version("qdrant-client"))'
  )"
  python3 - "${qdrant_server_version}" "${qdrant_client_version}" <<'PY'
import re
import sys

def major_minor(value: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise SystemExit(f"invalid Qdrant version: {value!r}")
    return int(match.group(1)), int(match.group(2))

server = major_minor(sys.argv[1])
client = major_minor(sys.argv[2])
if server[0] != client[0] or abs(server[1] - client[1]) > 1:
    raise SystemExit(
        "Qdrant client/server versions exceed the supported compatibility window: "
        f"server={sys.argv[1]}, client={sys.argv[2]}"
    )
PY
done
JAVA_HOME="${LHMSB_JAVA_HOME}" "${LHMSB_NEO4J_HOME}/bin/neo4j" version >/dev/null
"${LHMSB_JAVA_HOME}/bin/java" -version >/dev/null 2>&1
"${LHMSB_TEI_BIN}" --help >/dev/null
printf 'native runtime verification passed: %s\n' "${DATA_ROOT}/manifests/native-runtime.json"
