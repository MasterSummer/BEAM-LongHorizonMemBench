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
