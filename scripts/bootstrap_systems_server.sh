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

Prepare the native Python environments, pinned sources, models, and service paths.

Options:
  --data-root PATH  persistent root (default: /data/lhmsb)
  --env-file PATH   operator-owned KEY=VALUE file (default: .env)
  --allow-dirty     allow a non-formal dirty checkout
  --dry-run         print actions without network, GPU, or writes
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
    "${DATA_ROOT}/neo4j" "${DATA_ROOT}/wheelhouse" \
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
  for environment in core mem0 amem memos; do
    systems_print_command python3 -m venv "${DATA_ROOT}/venvs/${environment}"
    systems_print_command uv pip compile --generate-hashes --python-version 3.11 \
      --output-file "${DATA_ROOT}/locks/${environment}-requirements.txt" \
      "${REPO_ROOT}/pyproject.toml"
    systems_print_command uv pip sync --python "${DATA_ROOT}/venvs/${environment}/bin/python" \
      --require-hashes "${DATA_ROOT}/locks/${environment}-requirements.txt"
  done
  systems_print_command verify_system_runtime.sh --data-root "${DATA_ROOT}"
  systems_print_command uv run python -m lhmsb.qualification preflight-systems \
    --repository-only --dataset "${DATA_ROOT}/datasets/software_v4" \
    --config "${REPO_ROOT}/configs/experiments/systems_controlled_gpt_only_aaai.yaml" \
    --data-root "${DATA_ROOT}"
  exit 0
fi

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
[[ -f "${REPO_ROOT}/deploy/native-runtime.lock.yaml" ]] || {
  printf 'missing native-runtime.lock.yaml\n' >&2
  exit 1
}
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

# Generate hash-locked transitive requirements and wheelhouses for each native
# environment. A stale checked-in contract is never accepted as a live lock.
uv pip compile --generate-hashes --python-version 3.11 \
  "${REPO_ROOT}/pyproject.toml" --extra qualification \
  --output-file "${DATA_ROOT}/locks/core-requirements.txt"
uv pip download --require-hashes --dest "${DATA_ROOT}/wheelhouse/core" \
  -r "${DATA_ROOT}/locks/core-requirements.txt"
uv pip compile --generate-hashes --python-version 3.11 \
  "${REPO_ROOT}/pyproject.toml" --output-file "${DATA_ROOT}/locks/mem0-requirements.txt"
uv pip download --require-hashes --dest "${DATA_ROOT}/wheelhouse/mem0" \
  -r "${DATA_ROOT}/locks/mem0-requirements.txt"
uv pip compile --generate-hashes --python-version 3.11 \
  "${DATA_ROOT}/sources/amem" --output-file "${DATA_ROOT}/locks/amem-requirements.txt"
uv pip download --require-hashes --dest "${DATA_ROOT}/wheelhouse/amem" \
  -r "${DATA_ROOT}/locks/amem-requirements.txt"
uv pip compile --generate-hashes --python-version 3.11 \
  "${DATA_ROOT}/sources/memos" --output-file "${DATA_ROOT}/locks/memos-requirements.txt"
uv pip download --require-hashes --dest "${DATA_ROOT}/wheelhouse/memos" \
  -r "${DATA_ROOT}/locks/memos-requirements.txt"

for environment in core mem0 amem memos; do
  python="${DATA_ROOT}/venvs/${environment}/bin/python"
  [[ -x "${python}" ]] || python3 -m venv "${DATA_ROOT}/venvs/${environment}"
  uv pip sync --python "${python}" --require-hashes \
    "${DATA_ROOT}/locks/${environment}-requirements.txt"
  uv pip install --python "${python}" --no-deps --editable "${REPO_ROOT}"
done

for required in LHMSB_QDRANT_BIN LHMSB_NEO4J_HOME LHMSB_JAVA_HOME LHMSB_TEI_BIN \
  LHMSB_EMBEDDING_MODEL_DIR LHMSB_RERANKER_MODEL_DIR; do
  [[ -n "${!required:-}" ]] || {
    printf '%s must be configured for native execution\n' "${required}" >&2
    exit 1
  }
done
EMBEDDING_MODEL_DIR="${LHMSB_EMBEDDING_MODEL_DIR}" \
RERANKER_MODEL_DIR="${LHMSB_RERANKER_MODEL_DIR}" \
python3 - "${DATA_ROOT}/manifests/model-bundle.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def snapshot(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise SystemExit(f"model directory is not a directory: {root}")
    files: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {"root": str(root), "files": files}


payload = {
    "schema_version": 1,
    "models": {
        "embedding": snapshot(Path(os.environ["EMBEDDING_MODEL_DIR"])),
        "reranker": snapshot(Path(os.environ["RERANKER_MODEL_DIR"])),
    },
}
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
DATA_ROOT="${DATA_ROOT}" QDRANT_BIN="${LHMSB_QDRANT_BIN}" \
NEO4J_HOME="${LHMSB_NEO4J_HOME}" JAVA_HOME="${LHMSB_JAVA_HOME}" \
TEI_BIN="${LHMSB_TEI_BIN}" python3 - "${DATA_ROOT}/manifests/native-runtime.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

paths = {
    "qdrant": Path(os.environ["QDRANT_BIN"]),
    "neo4j": Path(os.environ["NEO4J_HOME"]) / "bin/neo4j",
    "java": Path(os.environ["JAVA_HOME"]) / "bin/java",
    "text-embeddings-router": Path(os.environ["TEI_BIN"]),
}
payload = {"schema_version": 1, "executables": {name: digest(path) for name, path in paths.items()}}
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
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

if [[ -f "${DATA_ROOT}/datasets/software_v4/MANIFEST.json" ]]; then
  uv run python -m lhmsb.qualification preflight-systems \
    --repository-only \
    --dataset "${DATA_ROOT}/datasets/software_v4" \
    --config "${REPO_ROOT}/configs/experiments/systems_controlled_gpt_only_aaai.yaml" \
    --data-root "${DATA_ROOT}" \
    --json "${DATA_ROOT}/runs/preflight-systems/repository.json"
else
  printf '\nBootstrap complete. The GPT-only v0.4 dataset is not present yet.\n'
  printf 'Generate and freeze it before running preflight or qualification:\n'
  printf '  SEEDS=$(seq 0 49)\n'
  printf '  python -m lhmsb.datasets generate-mem0-stateful --seeds ${SEEDS} --n-episodes 1 --n-sessions 16 --out %q/datasets/software_v4.stage\n' "${DATA_ROOT}"
  printf '  python -m lhmsb.datasets freeze-mem0-stateful --src %q/datasets/software_v4.stage --out %q/datasets/software_v4\n' "${DATA_ROOT}" "${DATA_ROOT}"
fi
