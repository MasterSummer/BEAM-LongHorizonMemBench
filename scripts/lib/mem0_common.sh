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
