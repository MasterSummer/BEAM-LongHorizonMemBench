#!/usr/bin/env bash
set -euo pipefail

# Create a private official-DeepSeek credential pool without printing any key.
# The existing DEEPSEEK_API_KEY from the operator .env is retained as slot 0;
# additional keys are read silently from the controlling terminal.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
KEY_FILE="${LHMSB_DEEPSEEK_KEY_FILE:-${HOME}/.config/memhorizon/deepseek.keys}"

[[ -f "${ENV_FILE}" ]] || {
  printf 'missing operator settings file: %s\n' "${ENV_FILE}" >&2
  exit 1
}
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
[[ -n "${DEEPSEEK_API_KEY:-}" ]] || {
  printf 'DEEPSEEK_API_KEY is missing from %s\n' "${ENV_FILE}" >&2
  exit 1
}

mkdir -p "$(dirname "${KEY_FILE}")"
chmod 700 "$(dirname "${KEY_FILE}")"
temporary="${KEY_FILE}.tmp.$$"
trap 'rm -f "${temporary}"' EXIT
umask 077
printf '%s\n' "${DEEPSEEK_API_KEY}" >"${temporary}"

while true; do
  key=""
  IFS= read -r -s -p "Additional official DeepSeek API key (empty to finish): " key
  printf '\n'
  [[ -n "${key}" ]] || break
  if ! grep -Fqx -- "${key}" "${temporary}"; then
    printf '%s\n' "${key}" >>"${temporary}"
  fi
done
unset key DEEPSEEK_API_KEY

mv "${temporary}" "${KEY_FILE}"
trap - EXIT
chmod 600 "${KEY_FILE}"
count="$(awk 'NF {count++} END {print count + 0}' "${KEY_FILE}")"
printf 'Configured %s DeepSeek API key(s) in %s\n' "${count}" "${KEY_FILE}"
