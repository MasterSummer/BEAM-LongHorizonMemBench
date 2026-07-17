#!/usr/bin/env bash

# Shared helpers for the canonical native multisystem workflow.  This file is
# intentionally independent of any container runtime: worker commands execute
# from one of the four host Python virtual environments and services bind only
# to loopback.

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

systems_load_env() {
  local env_file="$1"
  if [[ ! -f "${env_file}" ]]; then
    printf 'missing operator settings file: %s\n' "${env_file}" >&2
    return 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
}

systems_prepare_dirs() {
  local data_root="$1"
  mkdir -p \
    "${data_root}/datasets" \
    "${data_root}/env" \
    "${data_root}/history" \
    "${data_root}/locks" \
    "${data_root}/logs" \
    "${data_root}/manifests" \
    "${data_root}/models" \
    "${data_root}/neo4j" \
    "${data_root}/qdrant" \
    "${data_root}/runs" \
    "${data_root}/services" \
    "${data_root}/sources" \
    "${data_root}/venvs" \
    "${data_root}/wheelhouse"
}

systems_venv_python() {
  local data_root="$1"
  local environment="$2"
  case "${environment}" in
    core|mem0|amem|memos) printf '%s\n' "${data_root}/venvs/${environment}/bin/python" ;;
    *)
      printf 'unknown Python environment: %s\n' "${environment}" >&2
      return 2
      ;;
  esac
}

systems_run_cli() {
  local data_root="$1"
  local environment="$2"
  shift 2
  "$(systems_venv_python "${data_root}" "${environment}")" \
    -m lhmsb.qualification "$@"
}

systems_assert_lock_contract() {
  local repo_root="$1"
  local environment path
  for environment in core mem0 amem memos; do
    path="${repo_root}/deploy/locks/${environment}-requirements.txt"
    [[ -f "${path}" ]] || {
      printf 'missing tracked lock contract: %s\n' "${path}" >&2
      return 1
    }
    grep -q 'lock-status: bootstrap-contract' "${path}" || {
      printf 'invalid tracked lock contract: %s\n' "${path}" >&2
      return 1
    }
    grep -q -- '--require-hashes' "${path}" || {
      printf 'tracked lock does not require hashes: %s\n' "${path}" >&2
      return 1
    }
  done
}

systems_assert_generated_lock() {
  local data_root="$1"
  local environment="$2"
  local path="${data_root}/locks/${environment}-requirements.txt"
  [[ -s "${path}" ]] || {
    printf 'missing generated lock: %s\n' "${path}" >&2
    return 1
  }
  grep -q -- '--hash=sha256:' "${path}" || {
    printf 'generated lock has no distribution hashes: %s\n' "${path}" >&2
    return 1
  }
}

systems_select_a100_devices() {
  local allocated="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}"
  local embedding="${LHMSB_EMBEDDING_GPU_ID:-}"
  local reranker="${LHMSB_RERANKER_GPU_ID:-}"
  local -a devices=()
  if [[ -n "${allocated}" ]]; then
    allocated="${allocated//gpu:/}"
    IFS=',' read -r -a devices <<<"${allocated}"
    if ((${#devices[@]} < 2)); then
      printf 'at least two allocated A100 devices are required\n' >&2
      return 1
    fi
    embedding="${devices[0]//[[:space:]]/}"
    reranker="${devices[1]//[[:space:]]/}"
  fi
  if [[ -z "${embedding}" || -z "${reranker}" ]]; then
    printf 'set allocated GPUs or both native service GPU IDs\n' >&2
    return 1
  fi
  [[ "${embedding}" != "${reranker}" ]] || {
    printf 'embedding and reranker devices must be distinct\n' >&2
    return 1
  }
  export LHMSB_EMBEDDING_GPU_ID="${embedding}"
  export LHMSB_RERANKER_GPU_ID="${reranker}"
}

systems_configure_gpus() {
  systems_select_a100_devices
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  command -v nvidia-smi >/dev/null || {
    printf 'nvidia-smi is required for a live native run\n' >&2
    return 1
  }
  local inventory
  inventory="$(nvidia-smi --query-gpu=index,name --format=csv,noheader)"
  local device
  for device in "${LHMSB_EMBEDDING_GPU_ID}" "${LHMSB_RERANKER_GPU_ID}"; do
    printf '%s\n' "${inventory}" | awk -F, -v wanted="${device}" \
      '$1 ~ /^[[:space:]]*'"${device}"'[[:space:]]*$/ && toupper($2) ~ /A100/ {found=1} END {exit !found}' || {
        printf 'selected device %s is not an NVIDIA A100\n' "${device}" >&2
        return 1
      }
  done
}

systems_acquire_run_lock() {
  local data_root="$1"
  local run_name="${2:-${LHMSB_RUN_NAME:-systems-run}}"
  local lock_file="${data_root}/locks/run-${run_name}.lock"
  mkdir -p "$(dirname "${lock_file}")"
  exec 9>"${lock_file}"
  if ! flock -n 9; then
    printf 'another job owns run identity %s\n' "${run_name}" >&2
    return 1
  fi
}

systems_acquire_slurm_lock() {
  systems_acquire_run_lock "$1" "slurm-${SLURM_JOB_ID:-manual}"
}

systems_write_host_manifest() {
  local data_root="$1"
  local manifest="${data_root}/manifests/host.json"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    systems_print_command nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
      --format=csv --noheader
    return 0
  fi
  local inventory
  inventory="$(nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader)"
  GPU_INVENTORY="${inventory}" \
  EMBEDDING_GPU="${LHMSB_EMBEDDING_GPU_ID}" \
  RERANKER_GPU="${LHMSB_RERANKER_GPU_ID}" \
    python3 - "${manifest}" <<'PY'
import json
import os
import sys
from pathlib import Path

rows = [line.strip() for line in os.environ["GPU_INVENTORY"].splitlines() if line.strip()]
selected = (os.environ["EMBEDDING_GPU"], os.environ["RERANKER_GPU"])
by_index = {row.split(",", 1)[0].strip(): row for row in rows}
for device in selected:
    row = by_index.get(device)
    if row is None or "A100" not in row.upper():
        raise SystemExit(f"selected device {device} is not an NVIDIA A100")
payload = {
    "schema_version": 1,
    "selected_devices": {"embedding": selected[0], "reranker": selected[1]},
    "gpus": rows,
}
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

systems_write_runtime_env() {
  local data_root="$1"
  local path="${data_root}/manifests/runtime.env"
  local service_root="${data_root}/services/${LHMSB_SERVICE_INSTANCE:-manual}"
  export LHMSB_RUNTIME_MANIFEST_PATH="${data_root}/manifests/native-runtime.json"
  export LHMSB_MODEL_BUNDLE_MANIFEST_PATH="${data_root}/manifests/model-bundle.json"
  mkdir -p "$(dirname "${path}")"
  cat >"${path}.tmp" <<EOF
LHMSB_DATA_ROOT=$(printf '%q' "${data_root}")
LHMSB_RUNTIME_MANIFEST_PATH=$(printf '%q' "${LHMSB_RUNTIME_MANIFEST_PATH}")
LHMSB_MODEL_BUNDLE_MANIFEST_PATH=$(printf '%q' "${LHMSB_MODEL_BUNDLE_MANIFEST_PATH}")
LHMSB_QDRANT_URL=$(printf '%q' "${LHMSB_QDRANT_URL:-http://127.0.0.1:6333}")
LHMSB_NEO4J_URI=$(printf '%q' "${LHMSB_NEO4J_URI:-bolt://127.0.0.1:7687}")
LHMSB_EMBEDDING_URL=$(printf '%q' "${LHMSB_EMBEDDING_URL:-http://127.0.0.1:8080}")
LHMSB_RERANKER_URL=$(printf '%q' "${LHMSB_RERANKER_URL:-http://127.0.0.1:8081}")
LHMSB_SERVICE_ROOT=$(printf '%q' "${service_root}")
LHMSB_NEO4J_PASSWORD=$(printf '%q' "${LHMSB_NEO4J_PASSWORD:-}")
EOF
  mv "${path}.tmp" "${path}"
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
