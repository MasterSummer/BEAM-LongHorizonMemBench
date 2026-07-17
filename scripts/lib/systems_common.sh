#!/usr/bin/env bash

# Shared shell contract for the schema-v2 multisystem workflow. Functions in
# this file do not print environment values, so provider credentials cannot
# accidentally leak into Slurm logs or dry-run output.

systems_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

systems_print_command() {
  local argument
  printf 'DRY-RUN'
  for argument in "$@"; do
    printf ' %q' "${argument}"
  done
  printf '\n'
}

systems_run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    systems_print_command "$@"
    return 0
  fi
  "$@"
}

systems_unknown_argument() {
  printf 'unknown argument: %s\n' "$1" >&2
  return 2
}

systems_require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" ]]; then
    printf '%s requires a value\n' "${option}" >&2
    return 2
  fi
}

systems_compose() {
  local repo_root="$1"
  local env_file="$2"
  shift 2
  systems_run docker compose \
    --project-name "${LHMSB_COMPOSE_PROJECT:-lhmsb-systems}" \
    --env-file "${env_file}" \
    -f "${repo_root}/deploy/compose.systems.yaml" \
    "$@"
}

systems_configure_gpus() {
  local embedding_id="${LHMSB_EMBEDDING_GPU_ID:-}"
  local reranker_id="${LHMSB_RERANKER_GPU_ID:-}"
  local allocated="${SLURM_JOB_GPUS:-}"
  local -a gpu_ids=()

  if [[ -n "${allocated}" ]]; then
    IFS=',' read -r -a gpu_ids <<<"${allocated}"
    if ((${#gpu_ids[@]} < 2)); then
      printf 'at least two allocated Slurm GPUs are required\n' >&2
      return 1
    fi
    embedding_id="${gpu_ids[0]//[[:space:]]/}"
    reranker_id="${gpu_ids[1]//[[:space:]]/}"
  elif [[ -z "${embedding_id}" || -z "${reranker_id}" ]]; then
    printf 'set SLURM_JOB_GPUS or both LHMSB_EMBEDDING_GPU_ID and LHMSB_RERANKER_GPU_ID\n' >&2
    return 1
  fi

  if [[ "${embedding_id}" == "${reranker_id}" ]]; then
    printf 'embedding and reranker GPU IDs must be distinct\n' >&2
    return 1
  fi
  export LHMSB_EMBEDDING_GPU_ID="${embedding_id}"
  export LHMSB_RERANKER_GPU_ID="${reranker_id}"
}

systems_acquire_slurm_lock() {
  local data_root="$1"
  local lock_file="${data_root}/locks/systems-slurm.lock"
  mkdir -p "$(dirname "${lock_file}")"
  exec 9>"${lock_file}"
  if ! flock -n 9; then
    printf 'another multisystem Slurm job owns %s\n' "${lock_file}" >&2
    return 1
  fi
}

systems_prepare_dirs() {
  local data_root="$1"
  mkdir -p \
    "${data_root}/datasets" \
    "${data_root}/models" \
    "${data_root}/qdrant" \
    "${data_root}/neo4j" \
    "${data_root}/history" \
    "${data_root}/wheelhouse" \
    "${data_root}/images" \
    "${data_root}/manifests" \
    "${data_root}/runs" \
    "${data_root}/logs" \
    "${data_root}/locks" \
    "${data_root}/bundles"
}

systems_restore_archived_images() {
  local data_root="$1"
  local archive
  for archive in qdrant.tar neo4j.tar tei.tar core-worker.tar mem0-worker.tar \
    amem-worker.tar memos-worker.tar; do
    if [[ ! -f "${data_root}/images/${archive}" ]]; then
      printf 'missing bootstrapped image archive: %s\n' \
        "${data_root}/images/${archive}" >&2
      return 1
    fi
    docker load --input "${data_root}/images/${archive}" >/dev/null
  done
}

systems_verify_runtime_images() {
  local data_root="$1"
  local manifest="${data_root}/manifests/images.json"
  python3 - "${manifest}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing image manifest: {path}")
data = json.loads(path.read_text(encoding="utf-8"))
required = (
    "qdrant_runtime",
    "neo4j_runtime",
    "tei_runtime",
    "core_worker",
    "mem0_worker",
    "amem_worker",
    "memos_worker",
)
for key in required:
    value = data.get(key)
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SystemExit(f"{key} is not an immutable sha256 image ID")
PY
}

systems_write_host_manifest() {
  local data_root="$1"
  local manifest="${data_root}/manifests/host.json"
  local gpu_inventory
  gpu_inventory="$(nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv,noheader)"
  GPU_INVENTORY="${gpu_inventory}" python3 - "${manifest}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
gpus = [line.strip() for line in os.environ["GPU_INVENTORY"].splitlines() if line.strip()]
if len(gpus) < 2:
    raise SystemExit(f"at least two NVIDIA GPUs are required, found {len(gpus)}")
if not all("A100" in item.upper() for item in gpus[:2]):
    raise SystemExit("the first two GPUs are not NVIDIA A100 devices")
payload = {"schema_version": 1, "gpus": gpus}
temporary = path.with_suffix(".json.tmp")
temporary.parent.mkdir(parents=True, exist_ok=True)
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

systems_require_live_secrets() {
  [[ -n "${OPENCODE_ZEN_API_KEY:-}" ]] || {
    printf 'OPENCODE_ZEN_API_KEY is required for live evaluation\n' >&2
    return 1
  }
  [[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
    printf 'DEEPSEEK_API_KEY is required for live evaluation\n' >&2
    return 1
  }
}
