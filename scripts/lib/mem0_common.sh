#!/usr/bin/env bash

mem0_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

mem0_print_command() {
  local argument
  printf 'DRY-RUN'
  for argument in "$@"; do
    printf ' %q' "${argument}"
  done
  printf '\n'
}

mem0_run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    mem0_print_command "$@"
    return 0
  fi
  "$@"
}

mem0_unknown_argument() {
  printf 'unknown argument: %s\n' "$1" >&2
  return 2
}

mem0_require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    printf '%s requires a value\n' "${option}" >&2
    return 2
  fi
}

mem0_compose() {
  local repo_root="$1"
  local env_file="$2"
  shift 2
  mem0_run docker compose \
    --env-file "${env_file}" \
    -f "${repo_root}/deploy/compose.mem0.yaml" \
    "$@"
}

mem0_configure_slurm_gpus() {
  local embedding_id="${LHMSB_EMBEDDING_GPU_ID:-}"
  local reranker_id="${LHMSB_RERANKER_GPU_ID:-}"
  local allocated="${SLURM_JOB_GPUS:-}"
  local -a gpu_ids=()

  if [[ -n "${allocated}" ]]; then
    IFS=',' read -r -a gpu_ids <<<"${allocated}"
    if ((${#gpu_ids[@]} < 2)); then
      printf 'at least two allocated Slurm GPUs are required: %s\n' \
        "${allocated}" >&2
      return 1
    fi
    embedding_id="${gpu_ids[0]//[[:space:]]/}"
    reranker_id="${gpu_ids[1]//[[:space:]]/}"
  elif [[ -n "${embedding_id}" || -n "${reranker_id}" ]]; then
    if [[ -z "${embedding_id}" || -z "${reranker_id}" ]]; then
      printf 'set both LHMSB_EMBEDDING_GPU_ID and LHMSB_RERANKER_GPU_ID\n' >&2
      return 1
    fi
  else
    printf 'SLURM_JOB_GPUS is empty; explicitly set both LHMSB GPU IDs\n' >&2
    return 1
  fi

  if [[ "${embedding_id}" == "${reranker_id}" ]]; then
    printf 'embedding and reranker GPU IDs must be distinct\n' >&2
    return 1
  fi
  export LHMSB_EMBEDDING_GPU_ID="${embedding_id}"
  export LHMSB_RERANKER_GPU_ID="${reranker_id}"
}

mem0_acquire_slurm_lock() {
  local data_root="$1"
  local lock_file="${data_root}/locks/mem0-slurm.lock"
  mkdir -p "$(dirname "${lock_file}")"
  exec 9>"${lock_file}"
  if ! flock -n 9; then
    printf 'another Mem0 Slurm job owns %s\n' "${lock_file}" >&2
    return 1
  fi
}

mem0_restore_archived_images() {
  local data_root="$1"
  local archive
  for archive in qdrant.tar tei.tar worker.tar; do
    if [[ ! -f "${data_root}/images/${archive}" ]]; then
      printf 'missing bootstrapped image archive: %s\n' \
        "${data_root}/images/${archive}" >&2
      return 1
    fi
    docker load --input "${data_root}/images/${archive}"
  done
  mem0_verify_runtime_images "${data_root}"
}

mem0_verify_runtime_images() {
  local data_root="$1"
  local manifest="${data_root}/manifests/images.json"
  local expected
  local qdrant_expected
  local tei_expected
  local worker_expected
  local actual

  expected="$(python3 - "${manifest}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
keys = ("qdrant_runtime", "tei_runtime", "worker")
values = [data.get(key) for key in keys]
if not all(isinstance(value, str) and value.startswith("sha256:") for value in values):
    raise SystemExit(f"runtime image IDs are missing from {path}")
print("\t".join(values))
PY
)"
  IFS=$'\t' read -r qdrant_expected tei_expected worker_expected \
    <<<"${expected}"

  actual="$(docker image inspect --format '{{.Id}}' \
    'lhmsb/qdrant:qualification')"
  if [[ "${actual}" != "${qdrant_expected}" ]]; then
    printf 'Qdrant runtime image ID mismatch: %s != %s\n' \
      "${actual}" "${qdrant_expected}" >&2
    return 1
  fi
  actual="$(docker image inspect --format '{{.Id}}' \
    'lhmsb/tei:qualification')"
  if [[ "${actual}" != "${tei_expected}" ]]; then
    printf 'TEI runtime image ID mismatch: %s != %s\n' \
      "${actual}" "${tei_expected}" >&2
    return 1
  fi
  actual="$(docker image inspect --format '{{.Id}}' "${worker_expected}")"
  if [[ "${actual}" != "${worker_expected}" ]]; then
    printf 'worker runtime image ID mismatch: %s != %s\n' \
      "${actual}" "${worker_expected}" >&2
    return 1
  fi

  export QDRANT_RUNTIME_IMAGE_ID="${qdrant_expected}"
  export TEI_RUNTIME_IMAGE_ID="${tei_expected}"
  export LHMSB_WORKER_IMAGE_DIGEST="${worker_expected}"
}

mem0_write_host_manifest() {
  local data_root="$1"
  local manifest="${data_root}/manifests/host.json"
  local docker_version
  local compose_version
  local gpu_inventory
  docker_version="$(docker --version)"
  compose_version="$(docker compose version)"
  gpu_inventory="$(nvidia-smi \
    --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
    --format=csv,noheader)"
  mkdir -p "$(dirname "${manifest}")"
  DOCKER_VERSION="${docker_version}" \
    COMPOSE_VERSION="${compose_version}" \
    GPU_INVENTORY="${gpu_inventory}" \
    python3 - "${manifest}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
gpus = [
    line.strip()
    for line in os.environ["GPU_INVENTORY"].splitlines()
    if line.strip()
]
if len(gpus) < 2:
    raise SystemExit(f"at least two NVIDIA GPUs are required, found {len(gpus)}")
payload = {
    "schema_version": 1,
    "docker": os.environ["DOCKER_VERSION"],
    "compose": os.environ["COMPOSE_VERSION"],
    "gpus": gpus,
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(path)
PY
}
