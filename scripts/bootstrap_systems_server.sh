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

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_systems_server.sh [options]

Prepare the reproducible four-worker multisystem A100 environment.

Options:
  --data-root PATH  persistent root (default: /data/lhmsb)
  --env-file PATH   operator-owned KEY=VALUE file (default: .env)
  --allow-dirty     allow a non-formal dirty checkout
  --dry-run         print actions without Docker, network, GPU, or writes
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
  systems_print_command mkdir -p \
    "${DATA_ROOT}/datasets" "${DATA_ROOT}/models" "${DATA_ROOT}/qdrant" \
    "${DATA_ROOT}/neo4j" "${DATA_ROOT}/wheelhouse" "${DATA_ROOT}/images" \
    "${DATA_ROOT}/manifests" "${DATA_ROOT}/runs" "${DATA_ROOT}/logs" \
    "${DATA_ROOT}/locks" "${DATA_ROOT}/bundles"
  systems_print_command git clone --no-checkout https://github.com/agiresearch/A-mem \
    "${DATA_ROOT}/sources/amem"
  systems_print_command git -C "${DATA_ROOT}/sources/amem" checkout \
    ceffb860f0712bbae97b184d440df62bc910ca8d
  systems_print_command git clone --no-checkout https://github.com/MemTensor/MemOS \
    "${DATA_ROOT}/sources/memos"
  systems_print_command git -C "${DATA_ROOT}/sources/memos" checkout \
    583b07b998afc4debb6c5078439b0b3896f5b097
  systems_print_command uv pip compile --generate-hashes --python-version 3.11 \
    "${REPO_ROOT}/pyproject.toml" --extra qualification \
    --output-file "${DATA_ROOT}/wheelhouse/core-requirements.txt"
  systems_print_command uv pip download --require-hashes \
    --dest "${DATA_ROOT}/wheelhouse" -r "${DATA_ROOT}/wheelhouse/core-requirements.txt"
  systems_print_command docker build --pull=false --file docker/core-worker.Dockerfile \
    --tag lhmsb/core-worker:qualification .
  systems_print_command docker build --pull=false --file docker/amem-worker.Dockerfile \
    --tag lhmsb/amem-worker:qualification .
  systems_print_command docker build --pull=false --file docker/memos-worker.Dockerfile \
    --tag lhmsb/memos-worker:qualification .
  systems_print_command docker save --output "${DATA_ROOT}/images/core-worker.tar" \
    lhmsb/core-worker:qualification
  systems_print_command docker save --output "${DATA_ROOT}/images/amem-worker.tar" \
    lhmsb/amem-worker:qualification
  systems_print_command docker save --output "${DATA_ROOT}/images/memos-worker.tar" \
    lhmsb/memos-worker:qualification
  systems_print_command docker pull "${QDRANT_IMAGE:-qdrant/qdrant}:${QDRANT_IMAGE_TAG:-v1.15.4}"
  systems_print_command docker pull "${NEO4J_IMAGE:-neo4j}:${NEO4J_IMAGE_TAG:-5.26.3-community}"
  systems_print_command docker pull "${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference}:${TEI_IMAGE_TAG:-1.8.0}"
  systems_print_command docker save --output "${DATA_ROOT}/images/qdrant.tar" \
    "${QDRANT_IMAGE:-qdrant/qdrant}:${QDRANT_IMAGE_TAG:-v1.15.4}"
  systems_print_command docker save --output "${DATA_ROOT}/images/neo4j.tar" \
    "${NEO4J_IMAGE:-neo4j}:${NEO4J_IMAGE_TAG:-5.26.3-community}"
  systems_print_command docker save --output "${DATA_ROOT}/images/tei.tar" \
    "${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference}:${TEI_IMAGE_TAG:-1.8.0}"
  systems_print_command uv run python -m lhmsb.qualification preflight-systems \
    --repository-only --dataset "${DATA_ROOT}/datasets/software_v2" \
    --config "${REPO_ROOT}/configs/experiments/systems_controlled_zen.yaml" \
    --data-root "${DATA_ROOT}"
  exit 0
fi

command -v docker >/dev/null
command -v git >/dev/null
command -v python3 >/dev/null
command -v uv >/dev/null

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${REPO_ROOT}/.env.example" "${ENV_FILE}"
fi
chmod 600 "${ENV_FILE}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_REF="$(git -C "${REPO_ROOT}" symbolic-ref --short -q HEAD || printf 'detached')"
SOURCE_DIRTY=false
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
  SOURCE_DIRTY=true
  if [[ "${ALLOW_DIRTY}" != "1" ]]; then
    printf 'repository is dirty; commit changes or pass --allow-dirty\n' >&2
    exit 1
  fi
fi

if [[ "$(id -u)" == "0" ]]; then
  printf 'run bootstrap as the non-root owner of the persistent data root\n' >&2
  exit 1
fi

systems_prepare_dirs "${DATA_ROOT}"
mkdir -p "${DATA_ROOT}/sources/amem" "${DATA_ROOT}/sources/memos"
export LHMSB_WORKER_UID="${LHMSB_WORKER_UID:-$(id -u)}"
export LHMSB_WORKER_GID="${LHMSB_WORKER_GID:-$(id -g)}"
export LHMSB_REPO_ROOT="${REPO_ROOT}"

python3 - "${REPO_ROOT}/configs/systems-v2.lock.yaml" <<'PY'
import pathlib
import sys

import yaml

data = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
systems = data.get("systems", {})
expected = {
    "amem": "ceffb860f0712bbae97b184d440df62bc910ca8d",
    "memos": "583b07b998afc4debb6c5078439b0b3896f5b097",
}
for name, commit in expected.items():
    if systems.get(name, {}).get("source_commit") != commit:
        raise SystemExit(f"unexpected {name} source pin")
PY

clone_at_pin() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --no-checkout "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --tags --force origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
  [[ "$(git -C "${destination}" rev-parse HEAD)" == "${commit}" ]]
}

clone_at_pin https://github.com/agiresearch/A-mem \
  "${DATA_ROOT}/sources/amem" ceffb860f0712bbae97b184d440df62bc910ca8d
clone_at_pin https://github.com/MemTensor/MemOS \
  "${DATA_ROOT}/sources/memos" 583b07b998afc4debb6c5078439b0b3896f5b097

# Generate hash-locked transitive requirements and wheelhouses before any
# image build. A stale checked-in placeholder manifest is never accepted as a
# live image input.
uv pip compile --generate-hashes --python-version 3.11 \
  "${REPO_ROOT}/pyproject.toml" --extra qualification \
  --output-file "${DATA_ROOT}/wheelhouse/core-requirements.txt"
uv pip download --require-hashes --dest "${DATA_ROOT}/wheelhouse" \
  -r "${DATA_ROOT}/wheelhouse/core-requirements.txt"
uv pip compile --generate-hashes --python-version 3.11 \
  "${DATA_ROOT}/sources/amem" --output-file "${REPO_ROOT}/docker/locks/amem-requirements.txt"
uv pip download --require-hashes --dest "${DATA_ROOT}/wheelhouse" \
  -r "${REPO_ROOT}/docker/locks/amem-requirements.txt"
uv pip compile --generate-hashes --python-version 3.11 \
  "${DATA_ROOT}/sources/memos" --output-file "${REPO_ROOT}/docker/locks/memos-requirements.txt"
uv pip download --require-hashes --dest "${DATA_ROOT}/wheelhouse" \
  -r "${REPO_ROOT}/docker/locks/memos-requirements.txt"

export PYTHON_BASE_IMAGE="${PYTHON_BASE_IMAGE:-python:3.11-slim}"
export PYTHON_BASE_DIGEST="${PYTHON_BASE_DIGEST:?bootstrap must resolve PYTHON_BASE_DIGEST}"
export LHMSB_COMPOSE_PROJECT="${LHMSB_COMPOSE_PROJECT:-lhmsb-systems-bootstrap}"
for image in core-worker amem-worker memos-worker; do
  docker build --pull=false \
    --build-arg "PYTHON_BASE_IMAGE=${PYTHON_BASE_IMAGE}" \
    --build-arg "PYTHON_BASE_DIGEST=${PYTHON_BASE_DIGEST}" \
    --build-arg "SOURCE_COMMIT=${SOURCE_COMMIT}" \
    --build-arg "SOURCE_REF=${SOURCE_REF}" \
    --build-arg "SOURCE_DIRTY=${SOURCE_DIRTY}" \
    --file "${REPO_ROOT}/docker/${image}.Dockerfile" \
    --tag "lhmsb/${image}:qualification" "${REPO_ROOT}"
  docker save --output "${DATA_ROOT}/images/${image}.tar" "lhmsb/${image}:qualification"
done

git -C "${REPO_ROOT}" rev-parse HEAD >"${DATA_ROOT}/manifests/benchmark.commit"

resolve_runtime_image() {
  local reference="$1"
  local alias="$2"
  local archive="$3"
  local digest
  docker pull "${reference}" >/dev/null
  digest="$(docker image inspect --format '{{index .RepoDigests 0}}' "${reference}")"
  if [[ "${digest}" != *@sha256:* ]]; then
    printf 'could not resolve immutable digest for %s\n' "${reference}" >&2
    exit 1
  fi
  docker tag "${reference}" "${alias}"
  docker save --output "${DATA_ROOT}/images/${archive}" "${alias}"
  docker image inspect --format '{{.Id}}' "${alias}"
}

QDRANT_REFERENCE="${QDRANT_IMAGE:-qdrant/qdrant}:${QDRANT_IMAGE_TAG:-v1.15.4}"
NEO4J_REFERENCE="${NEO4J_IMAGE:-neo4j}:${NEO4J_IMAGE_TAG:-5.26.3-community}"
TEI_REFERENCE="${TEI_IMAGE:-ghcr.io/huggingface/text-embeddings-inference}:${TEI_IMAGE_TAG:-1.8.0}"
QDRANT_ID="$(resolve_runtime_image "${QDRANT_REFERENCE}" lhmsb/qdrant:qualification qdrant.tar)"
NEO4J_ID="$(resolve_runtime_image "${NEO4J_REFERENCE}" lhmsb/neo4j:qualification neo4j.tar)"
TEI_ID="$(resolve_runtime_image "${TEI_REFERENCE}" lhmsb/tei:qualification tei.tar)"
CORE_ID="$(docker image inspect --format '{{.Id}}' lhmsb/core-worker:qualification)"
MEM0_ID="$(docker image inspect --format '{{.Id}}' lhmsb/mem0-worker:qualification)"
AMEM_ID="$(docker image inspect --format '{{.Id}}' lhmsb/amem-worker:qualification)"
MEMOS_ID="$(docker image inspect --format '{{.Id}}' lhmsb/memos-worker:qualification)"
QDRANT_ID="${QDRANT_ID}" NEO4J_ID="${NEO4J_ID}" TEI_ID="${TEI_ID}" \
CORE_ID="${CORE_ID}" MEM0_ID="${MEM0_ID}" AMEM_ID="${AMEM_ID}" MEMOS_ID="${MEMOS_ID}" \
  python3 - "${DATA_ROOT}/manifests/images.json" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "qdrant_runtime": os.environ["QDRANT_ID"],
    "neo4j_runtime": os.environ["NEO4J_ID"],
    "tei_runtime": os.environ["TEI_ID"],
    "core_worker": os.environ["CORE_ID"],
    "mem0_worker": os.environ["MEM0_ID"],
    "amem_worker": os.environ["AMEM_ID"],
    "memos_worker": os.environ["MEMOS_ID"],
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
SOURCE_COMMIT="${SOURCE_COMMIT}" SOURCE_DIRTY="${SOURCE_DIRTY}" python3 - "${DATA_ROOT}/manifests/build.json" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_dirty": os.environ["SOURCE_DIRTY"].lower() == "true",
    "systems": {
        "amem": "ceffb860f0712bbae97b184d440df62bc910ca8d",
        "memos": "583b07b998afc4debb6c5078439b0b3896f5b097",
    },
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

uv run python -m lhmsb.qualification preflight-systems \
  --repository-only \
  --dataset "${DATA_ROOT}/datasets/software_v2" \
  --config "${REPO_ROOT}/configs/experiments/systems_controlled_zen.yaml" \
  --data-root "${DATA_ROOT}" \
  --json "${DATA_ROOT}/runs/preflight-systems/repository.json"
