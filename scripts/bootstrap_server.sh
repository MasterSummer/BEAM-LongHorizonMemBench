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

LEGACY_RELEASE="software-vertical-v0.1.0"
LEGACY_ARCHIVE="software_v1-6b4edbf.tar.gz"
LEGACY_SHA256="c1b35c1a554c2ad8d1e1f895a563a6bc5a67979b54b8857ce287468c2efe8130"
MEM0_RELEASE="software-vertical-mem0-v0.2.0"
MEM0_ARCHIVE="software_mem0_v2.tar.gz"
MEM0_RELEASE_SHA256="4a455e1a16cc66fa7c218ba48543174426ec710989a301de3fa61f694c170380"
MEM0_WHEEL_SHA256="6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2"
EMBEDDING_REPOSITORY="BAAI/bge-m3"
EMBEDDING_REVISION="5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_REPOSITORY="BAAI/bge-reranker-v2-m3"
RERANKER_REVISION="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_server.sh [options]

Prepare an online A100 host for the frozen Mem0 qualification.

Options:
  --data-root PATH  Persistent benchmark root (default: /data/lhmsb)
  --env-file PATH   Compose environment file (default: .env)
  --allow-dirty     Permit a non-reproducible image build from local changes
  --dry-run         Print every external action without writing or downloading
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
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

if [[ "${DRY_RUN}" == "1" ]]; then
  mem0_print_command mkdir -p \
    "${DATA_ROOT}/datasets" \
    "${DATA_ROOT}/models" \
    "${DATA_ROOT}/qdrant" \
    "${DATA_ROOT}/history" \
    "${DATA_ROOT}/hf-cache" \
    "${DATA_ROOT}/wheelhouse" \
    "${DATA_ROOT}/images" \
    "${DATA_ROOT}/manifests" \
    "${DATA_ROOT}/runs" \
    "${DATA_ROOT}/logs" \
    "${DATA_ROOT}/bundles"
  mem0_print_command verify-release "${LEGACY_RELEASE}" "${LEGACY_SHA256}"
  mem0_print_command verify-release "${MEM0_RELEASE}" "${MEM0_RELEASE_SHA256}"
  mem0_print_command build-wheelhouse "manifests/wheels.json" "${MEM0_WHEEL_SHA256}"
  mem0_print_command download-model "${EMBEDDING_REPOSITORY}" "${EMBEDDING_REVISION}"
  mem0_print_command download-model "${RERANKER_REPOSITORY}" "${RERANKER_REVISION}"
  mem0_print_command hash-models "manifests/models.json"
  mem0_print_command pull-build-save-images "manifests/images.json"
  PREFLIGHT_COMMAND=(
    uv run python -m lhmsb.qualification preflight --repository-only
    --dataset "${DATA_ROOT}/datasets/software_mem0_v2"
    --config "${REPO_ROOT}/configs/experiments/mem0_controlled_zen.yaml"
    --data-root "${DATA_ROOT}"
  )
  if [[ "${ALLOW_DIRTY}" == "1" ]]; then
    PREFLIGHT_COMMAND+=(--allow-dirty)
  fi
  mem0_print_command "${PREFLIGHT_COMMAND[@]}"
  exit 0
fi

command -v docker >/dev/null
command -v git >/dev/null
command -v python3 >/dev/null
command -v uv >/dev/null

SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_REF="$(git -C "${REPO_ROOT}" symbolic-ref --short -q HEAD || printf 'detached')"
SOURCE_DIRTY=false
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
  SOURCE_DIRTY=true
  if [[ "${ALLOW_DIRTY}" != "1" ]]; then
    printf '%s\n' \
      'repository is dirty; commit changes or pass --allow-dirty for a non-formal build' >&2
    exit 1
  fi
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${REPO_ROOT}/.env.example" "${ENV_FILE}"
fi
chmod 600 "${ENV_FILE}"

set -a
# This file is operator-owned and must contain shell-compatible KEY=VALUE rows.
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
if [[ "${HOST_UID}" == "0" ]]; then
  printf '%s\n' \
    'run bootstrap_server.sh as the non-root user that will own /data/lhmsb' >&2
  exit 1
fi
LHMSB_WORKER_UID="${HOST_UID}"
LHMSB_WORKER_GID="${HOST_GID}"

if ! mkdir -p "${DATA_ROOT}"; then
  printf 'cannot create data root %s; pre-create it as the current non-root user\n' \
    "${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -w "${DATA_ROOT}" ]]; then
  printf 'data root is not writable: %s; pre-create it as the current non-root user\n' \
    "${DATA_ROOT}" >&2
  exit 1
fi

PYTHON_BASE_IMAGE="${PYTHON_BASE_IMAGE:-python:3.11-slim}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant}"
QDRANT_IMAGE_TAG="${QDRANT_IMAGE_TAG:-v1.15.4}"
TEI_IMAGE="${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference}"
TEI_IMAGE_TAG="${TEI_IMAGE_TAG:-1.8.0}"
LHMSB_WORKER_IMAGE="${LHMSB_WORKER_IMAGE:-lhmsb-mem0-worker}"

mkdir -p \
  "${DATA_ROOT}/datasets" \
  "${DATA_ROOT}/models" \
  "${DATA_ROOT}/qdrant" \
  "${DATA_ROOT}/history" \
  "${DATA_ROOT}/hf-cache" \
  "${DATA_ROOT}/wheelhouse" \
  "${DATA_ROOT}/images" \
  "${DATA_ROOT}/manifests" \
  "${DATA_ROOT}/runs" \
  "${DATA_ROOT}/logs" \
  "${DATA_ROOT}/bundles" \
  "${REPO_ROOT}/docker/wheelhouse"

verify_and_extract_release() {
  local release="$1"
  local archive="$2"
  local expected="$3"
  local source_path="${REPO_ROOT}/datasets/releases/${release}/${archive}"
  local actual
  actual="$(python3 - "${source_path}" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)"
  if [[ "${actual}" != "${expected}" ]]; then
    printf 'release hash mismatch for %s: %s != %s\n' \
      "${release}" "${actual}" "${expected}" >&2
    exit 1
  fi
  tar -xzf "${source_path}" -C "${DATA_ROOT}/datasets"
}

verify_and_extract_release \
  "${LEGACY_RELEASE}" "${LEGACY_ARCHIVE}" "${LEGACY_SHA256}"
verify_and_extract_release \
  "${MEM0_RELEASE}" "${MEM0_ARCHIVE}" "${MEM0_RELEASE_SHA256}"

HF_HOME="${DATA_ROOT}/hf-cache" uvx --from "huggingface_hub==0.34.4" hf download \
  "${EMBEDDING_REPOSITORY}" \
  --revision "${EMBEDDING_REVISION}" \
  --local-dir "${DATA_ROOT}/models/bge-m3"
HF_HOME="${DATA_ROOT}/hf-cache" uvx --from "huggingface_hub==0.34.4" hf download \
  "${RERANKER_REPOSITORY}" \
  --revision "${RERANKER_REVISION}" \
  --local-dir "${DATA_ROOT}/models/bge-reranker-v2-m3"

python3 - \
  "${DATA_ROOT}" \
  "${DATA_ROOT}/manifests/models.json" \
  "${EMBEDDING_REVISION}" \
  "${RERANKER_REVISION}" <<'PY'
import hashlib
import json
import pathlib
import sys

data_root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
model_roots = (
    data_root / "models" / "bge-m3",
    data_root / "models" / "bge-reranker-v2-m3",
)
files = {}
for root in model_roots:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(data_root).as_posix()
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "revisions": {
        "BAAI/bge-m3": sys.argv[3],
        "BAAI/bge-reranker-v2-m3": sys.argv[4],
    },
    "files": files,
}
output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

resolve_repo_digest() {
  local reference="$1"
  local resolved
  docker pull "${reference}" >/dev/null
  resolved="$(docker image inspect \
    --format '{{index .RepoDigests 0}}' "${reference}")"
  if [[ "${resolved}" != *@sha256:* ]]; then
    printf 'could not resolve immutable digest for %s\n' "${reference}" >&2
    exit 1
  fi
  printf '%s\n' "${resolved##*@}"
}

PYTHON_BASE_DIGEST="$(resolve_repo_digest "${PYTHON_BASE_IMAGE}")"
QDRANT_REFERENCE="${QDRANT_IMAGE}:${QDRANT_IMAGE_TAG}"
QDRANT_IMAGE_DIGEST="$(resolve_repo_digest "${QDRANT_REFERENCE}")"
TEI_REFERENCE="${TEI_IMAGE}:${TEI_IMAGE_TAG}"
TEI_IMAGE_DIGEST="$(resolve_repo_digest "${TEI_REFERENCE}")"

REQUIREMENTS_PATH="${DATA_ROOT}/manifests/qualification-requirements.txt"
uv export --frozen --extra qualification --no-dev --no-emit-project \
  --format requirements-txt --output-file "${REQUIREMENTS_PATH}"
find "${DATA_ROOT}/wheelhouse" -type f -name '*.whl' -delete
docker run --rm \
  --user "${HOST_UID}:${HOST_GID}" \
  --env HOME=/tmp \
  --volume "${REQUIREMENTS_PATH}:/tmp/qualification-requirements.txt:ro" \
  --volume "${DATA_ROOT}/wheelhouse:/wheelhouse" \
  "${PYTHON_BASE_IMAGE}@${PYTHON_BASE_DIGEST}" \
  python -m pip download \
    --disable-pip-version-check \
    --only-binary=:all: \
    --dest /wheelhouse \
    --requirement /tmp/qualification-requirements.txt
uv build --wheel --out-dir "${DATA_ROOT}/wheelhouse"

MEM0_WHEEL="${DATA_ROOT}/wheelhouse/mem0ai-2.0.12-py3-none-any.whl"
if [[ ! -f "${MEM0_WHEEL}" ]]; then
  printf 'missing exact Mem0 wheel: %s\n' "${MEM0_WHEEL}" >&2
  exit 1
fi
MEM0_WHEEL_ACTUAL="$(python3 - "${MEM0_WHEEL}" <<'PY'
import hashlib
import pathlib
import sys

print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
if [[ "${MEM0_WHEEL_ACTUAL}" != "${MEM0_WHEEL_SHA256}" ]]; then
  printf 'Mem0 wheel hash mismatch: %s != %s\n' \
    "${MEM0_WHEEL_ACTUAL}" "${MEM0_WHEEL_SHA256}" >&2
  exit 1
fi

find "${REPO_ROOT}/docker/wheelhouse" \
  -mindepth 1 -maxdepth 1 -type f ! -name .gitkeep -delete
cp "${DATA_ROOT}"/wheelhouse/*.whl "${REPO_ROOT}/docker/wheelhouse/"

python3 - "${DATA_ROOT}/wheelhouse" "${DATA_ROOT}/manifests/wheels.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
files = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.glob("*.whl"))
}
output.write_text(
    json.dumps({"schema_version": 1, "files": files}, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

docker build \
  --build-arg "PYTHON_BASE_IMAGE=${PYTHON_BASE_IMAGE}" \
  --build-arg "PYTHON_BASE_DIGEST=${PYTHON_BASE_DIGEST}" \
  --build-arg "SOURCE_COMMIT=${SOURCE_COMMIT}" \
  --build-arg "SOURCE_REF=${SOURCE_REF}" \
  --build-arg "SOURCE_DIRTY=${SOURCE_DIRTY}" \
  --tag "${LHMSB_WORKER_IMAGE}:qualification" \
  --file "${REPO_ROOT}/docker/mem0-worker.Dockerfile" \
  "${REPO_ROOT}"
LHMSB_WORKER_IMAGE_DIGEST="$(docker image inspect \
  --format '{{.Id}}' "${LHMSB_WORKER_IMAGE}:qualification")"

docker save --output "${DATA_ROOT}/images/python-base.tar" \
  "${PYTHON_BASE_IMAGE}@${PYTHON_BASE_DIGEST}"
docker save --output "${DATA_ROOT}/images/qdrant.tar" \
  "${QDRANT_IMAGE}@${QDRANT_IMAGE_DIGEST}"
docker save --output "${DATA_ROOT}/images/tei.tar" \
  "${TEI_IMAGE}@${TEI_IMAGE_DIGEST}"
docker save --output "${DATA_ROOT}/images/worker.tar" \
  "${LHMSB_WORKER_IMAGE}:qualification"

python3 - \
  "${DATA_ROOT}/manifests/images.json" \
  "${PYTHON_BASE_DIGEST}" \
  "${QDRANT_IMAGE_DIGEST}" \
  "${TEI_IMAGE_DIGEST}" \
  "${LHMSB_WORKER_IMAGE_DIGEST}" <<'PY'
import json
import pathlib
import sys

payload = {
    "python_base": sys.argv[2],
    "qdrant": sys.argv[3],
    "tei": sys.argv[4],
    "worker": sys.argv[5],
}
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

python3 - \
  "${ENV_FILE}" \
  "${PYTHON_BASE_DIGEST}" \
  "${QDRANT_IMAGE_TAG}" \
  "${QDRANT_IMAGE_DIGEST}" \
  "${TEI_IMAGE_TAG}" \
  "${TEI_IMAGE_DIGEST}" \
  "${LHMSB_WORKER_IMAGE_DIGEST}" \
  "${LHMSB_WORKER_UID}" \
  "${LHMSB_WORKER_GID}" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
updates = {
    "PYTHON_BASE_DIGEST": sys.argv[2],
    "QDRANT_IMAGE_TAG": sys.argv[3],
    "QDRANT_IMAGE_DIGEST": sys.argv[4],
    "TEI_IMAGE_TAG": sys.argv[5],
    "TEI_IMAGE_DIGEST": sys.argv[6],
    "LHMSB_WORKER_IMAGE_DIGEST": sys.argv[7],
    "LHMSB_WORKER_UID": sys.argv[8],
    "LHMSB_WORKER_GID": sys.argv[9],
}
lines = path.read_text(encoding="utf-8").splitlines()
seen = set()
rewritten = []
for line in lines:
    key = line.split("=", 1)[0] if "=" in line and not line.startswith("#") else None
    if key in updates:
        rewritten.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        rewritten.append(line)
for key, value in updates.items():
    if key not in seen:
        rewritten.append(f"{key}={value}")
path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
PY

PREFLIGHT_COMMAND=(
  uv run python -m lhmsb.qualification preflight --repository-only
  --dataset "${DATA_ROOT}/datasets/software_mem0_v2"
  --config "${REPO_ROOT}/configs/experiments/mem0_controlled_zen.yaml"
  --data-root "${DATA_ROOT}"
)
if [[ "${ALLOW_DIRTY}" == "1" ]]; then
  PREFLIGHT_COMMAND+=(--allow-dirty)
fi
"${PREFLIGHT_COMMAND[@]}"

printf 'Mem0 server bootstrap complete: %s\n' "${DATA_ROOT}"
printf 'Manifests: manifests/models.json manifests/wheels.json manifests/images.json\n'
