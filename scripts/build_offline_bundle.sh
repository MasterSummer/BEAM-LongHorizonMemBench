#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/mem0_common.sh
source "${SCRIPT_DIR}/lib/mem0_common.sh"

REPO_ROOT="$(mem0_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
OUT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/build_offline_bundle.sh --out PATH [options]

Build a credential-free transfer archive containing repository.tar.gz,
wheelhouse, images, models, software-vertical-v0.1.0,
software-vertical-mem0-v0.2.0, BUNDLE_MANIFEST.json, and a .sha256 sidecar.

Options:
  --data-root PATH  Prepared benchmark root (default: /data/lhmsb)
  --out PATH        Destination .tar.gz archive
  --dry-run         Print the bundle plan without reading or writing assets
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
    --out)
      mem0_require_value "$1" "${2:-}" || exit 2
      OUT="$2"
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
      mem0_unknown_argument "$1" || exit $?
      ;;
  esac
done

if [[ -z "${OUT}" ]]; then
  printf '%s\n' '--out is required' >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  mem0_print_command stage "repository.tar.gz"
  mem0_print_command stage "${DATA_ROOT}/wheelhouse"
  mem0_print_command stage "${DATA_ROOT}/images"
  mem0_print_command stage "${DATA_ROOT}/models"
  mem0_print_command stage "software-vertical-v0.1.0"
  mem0_print_command stage "software-vertical-mem0-v0.2.0"
  mem0_print_command write "BUNDLE_MANIFEST.json"
  mem0_print_command archive "${OUT}"
  mem0_print_command checksum "${OUT}.sha256"
  exit 0
fi

for required in \
  "${DATA_ROOT}/wheelhouse" \
  "${DATA_ROOT}/images" \
  "${DATA_ROOT}/models" \
  "${DATA_ROOT}/manifests/models.json" \
  "${DATA_ROOT}/manifests/wheels.json" \
  "${DATA_ROOT}/manifests/images.json"; do
  if [[ ! -e "${required}" ]]; then
    printf 'missing bootstrap asset: %s\n' "${required}" >&2
    exit 1
  fi
done

STAGING="$(mktemp -d "${TMPDIR:-/tmp}/lhmsb-bundle.XXXXXX")"
cleanup() {
  rm -rf "${STAGING}"
}
trap cleanup EXIT

mkdir -p \
  "${STAGING}/wheelhouse" \
  "${STAGING}/images" \
  "${STAGING}/models" \
  "${STAGING}/manifests" \
  "${STAGING}/datasets/releases"

git -C "${REPO_ROOT}" archive \
  --format=tar.gz \
  --output="${STAGING}/repository.tar.gz" \
  HEAD
git -C "${REPO_ROOT}" rev-parse HEAD >"${STAGING}/repository.commit"
cp -R "${DATA_ROOT}/wheelhouse/." "${STAGING}/wheelhouse/"
cp -R "${DATA_ROOT}/images/." "${STAGING}/images/"
cp -R "${DATA_ROOT}/models/." "${STAGING}/models/"
cp -R "${DATA_ROOT}/manifests/." "${STAGING}/manifests/"
cp -R \
  "${REPO_ROOT}/datasets/releases/software-vertical-v0.1.0" \
  "${STAGING}/datasets/releases/"
cp -R \
  "${REPO_ROOT}/datasets/releases/software-vertical-mem0-v0.2.0" \
  "${STAGING}/datasets/releases/"

python3 - "${STAGING}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = {}
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = path.relative_to(root).as_posix()
    if relative == "BUNDLE_MANIFEST.json":
        continue
    files[relative] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
payload = {
    "schema_version": 1,
    "credential_files_included": False,
    "files": files,
}
(root / "BUNDLE_MANIFEST.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

mkdir -p "$(dirname "${OUT}")"
python3 - "${STAGING}" "${OUT}" <<'PY'
import gzip
import pathlib
import tarfile
import sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
with output.open("wb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if path.is_file():
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
                else:
                    archive.addfile(info)
PY

python3 - "${OUT}" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
path.with_name(path.name + ".sha256").write_text(
    f"{digest}  {path.name}\n",
    encoding="utf-8",
)
PY

printf 'Offline bundle: %s\n' "${OUT}"
printf 'Checksum: %s.sha256\n' "${OUT}"
