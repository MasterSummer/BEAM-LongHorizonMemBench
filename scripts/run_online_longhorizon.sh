#!/usr/bin/env bash
set -euo pipefail

# Canonical container-free workflow for the 20-episode × 5-condition online
# long-horizon matrix. One Slurm array task owns one episode, starts one set of
# native services, and runs the five conditions sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/systems_common.sh"
source "${SCRIPT_DIR}/lib/systems_services.sh"
REPO_ROOT="$(systems_repo_root)"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
ENV_FILE="${LHMSB_ENV_FILE:-${REPO_ROOT}/.env}"
RUN_NAME="${LHMSB_ONLINE_RUN_NAME:-memhorizon-online-final}"
DATASET="${LHMSB_SYSTEM_DATASET:-${DATA_ROOT}/datasets/software_longitudinal_v014}"
CONFIG="${LHMSB_SYSTEM_CONFIG:-${REPO_ROOT}/configs/experiments/systems_controlled_gpt_only_longitudinal_v014_shengsuanyun_writer.yaml}"
MODE=""
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-}"
N_EPISODES=20
SYSTEM_CONDITIONS="${LHMSB_ONLINE_SYSTEM_CONDITIONS:-}"
DRY_RUN=0
ALLOW_DIRTY=0

usage() {
  cat <<'EOF'
Usage: scripts/run_online_longhorizon.sh MODE [options]

Modes:
  models     list routed GPT-5 model IDs without printing the API key
  plan       freeze the balanced online episode matrix
  task       run all five conditions for one episode task
  aggregate  build the final online report and integrity validation

Options:
  --data-root PATH
  --env-file PATH
  --dataset PATH
  --config PATH
  --run-name NAME
  --task-index N
  --n-episodes N
  --system-conditions CSV
  --allow-dirty   development planning only; never use for final results
  --dry-run
EOF
}

if (($#)); then
  MODE="$1"
  shift
fi
while (($#)); do
  case "$1" in
    --data-root) systems_require_value "$1" "${2:-}" || exit 2; DATA_ROOT="$2"; shift 2 ;;
    --env-file) systems_require_value "$1" "${2:-}" || exit 2; ENV_FILE="$2"; shift 2 ;;
    --dataset) systems_require_value "$1" "${2:-}" || exit 2; DATASET="$2"; shift 2 ;;
    --config) systems_require_value "$1" "${2:-}" || exit 2; CONFIG="$2"; shift 2 ;;
    --run-name) systems_require_value "$1" "${2:-}" || exit 2; RUN_NAME="$2"; shift 2 ;;
    --task-index) systems_require_value "$1" "${2:-}" || exit 2; TASK_INDEX="$2"; shift 2 ;;
    --n-episodes) systems_require_value "$1" "${2:-}" || exit 2; N_EPISODES="$2"; shift 2 ;;
    --system-conditions) systems_require_value "$1" "${2:-}" || exit 2; SYSTEM_CONDITIONS="$2"; shift 2 ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) systems_unknown_argument "$1" || exit $? ;;
  esac
done

case "${MODE}" in
  models|plan|task|aggregate) ;;
  *) usage >&2; exit 2 ;;
esac
[[ "${N_EPISODES}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'n-episodes must be a positive integer\n' >&2
  exit 2
}
RUN_DIR="${DATA_ROOT}/runs/online/${RUN_NAME}"
CORE_PYTHON="${DATA_ROOT}/venvs/core/bin/python"
PYTHONPATH_VALUE="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_python() {
  local environment="$1"
  shift
  PYTHONPATH="${PYTHONPATH_VALUE}" \
    "${DATA_ROOT}/venvs/${environment}/bin/python" "$@"
}

if [[ "${MODE}" == "models" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' \
      'DRY-RUN query ShengSuanYun /models and print only model IDs containing gpt-5'
    exit 0
  fi
  systems_load_env "${ENV_FILE}"
  [[ -n "${SHENGSUANYUN_API_KEY:-}" ]] || {
    printf 'SHENGSUANYUN_API_KEY is unset\n' >&2
    exit 1
  }
  python3 <<'PY'
import json
import os
import urllib.request

request = urllib.request.Request(
    "https://router.shengsuanyun.com/api/v1/models",
    headers={"Authorization": f"Bearer {os.environ['SHENGSUANYUN_API_KEY']}"},
)
with urllib.request.urlopen(request, timeout=30) as response:
    data = json.load(response)
ids = sorted(
    str(item.get("id", ""))
    for item in data.get("data", [])
    if "gpt-5" in str(item.get("id", "")).lower()
)
print("\n".join(ids) if ids else "NO_GPT5_MODELS")
PY
  exit 0
fi

if [[ "${MODE}" == "plan" ]]; then
  command=(
    "${CORE_PYTHON}" -m lhmsb.longhorizon.online_matrix plan
    --dataset "${DATASET}" --config "${CONFIG}" --out "${RUN_DIR}"
    --n-episodes "${N_EPISODES}"
  )
  if [[ -n "${SYSTEM_CONDITIONS}" ]]; then
    IFS=',' read -r -a online_conditions <<<"${SYSTEM_CONDITIONS}"
    for online_condition in "${online_conditions[@]}"; do
      command+=(--system-condition "${online_condition}")
    done
  fi
  [[ "${ALLOW_DIRTY}" == "1" ]] && command+=(--allow-dirty)
  if [[ "${DRY_RUN}" == "1" ]]; then
    systems_print_command env PYTHONPATH="${PYTHONPATH_VALUE}" "${command[@]}"
  else
    PYTHONPATH="${PYTHONPATH_VALUE}" "${command[@]}"
  fi
  exit 0
fi

if [[ "${MODE}" == "aggregate" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    systems_print_command env PYTHONPATH="${PYTHONPATH_VALUE}" "${CORE_PYTHON}" \
      -m lhmsb.longhorizon.online_matrix aggregate --run-dir "${RUN_DIR}"
  else
    run_python core -m lhmsb.longhorizon.online_matrix aggregate --run-dir "${RUN_DIR}"
  fi
  exit 0
fi

[[ "${TASK_INDEX}" =~ ^[0-9]+$ ]] || {
  printf 'task mode requires --task-index or SLURM_ARRAY_TASK_ID\n' >&2
  exit 2
}
if [[ "${DRY_RUN}" == "1" ]]; then
  systems_print_command systems_start_all_services "${DATA_ROOT}"
  systems_print_command "${CORE_PYTHON}" -m lhmsb.longhorizon.online_matrix task \
    --run-dir "${RUN_DIR}" --task-index "${TASK_INDEX}"
  systems_print_command "${CORE_PYTHON}" -m lhmsb.longhorizon.online_cli \
    --dataset '<matrix dataset>' --config '<matrix config>' \
    --episode-id '<task episode>' --condition '<five conditions>' \
    --steps-per-session 16 --out "${RUN_DIR}/results/task-${TASK_INDEX}"
  exit 0
fi

systems_load_env "${ENV_FILE}"
systems_select_deepseek_api_key
systems_configure_gpus
export LHMSB_DATA_ROOT="${DATA_ROOT}" LHMSB_REPO_ROOT="${REPO_ROOT}"
export LHMSB_SERVICE_INSTANCE="${RUN_NAME}-${SLURM_ARRAY_JOB_ID:-manual}-${TASK_INDEX}"
export MEMOS_BASE_PATH="${RUN_DIR}/runtime/${LHMSB_SERVICE_INSTANCE}/memos"
systems_acquire_run_lock "${DATA_ROOT}" "${LHMSB_SERVICE_INSTANCE}"
mapfile -t required_secrets < <(
  python3 - "${RUN_DIR}/MATRIX.json" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name in data.get("required_secret_env", []):
    print(name)
PY
)
for secret_name in "${required_secrets[@]}"; do
  [[ -n "${!secret_name:-}" ]] || {
    printf 'required secret environment variable is unset: %s\n' "${secret_name}" >&2
    exit 1
  }
done
systems_start_all_services "${DATA_ROOT}"
cleanup() {
  local status="$?"
  trap - EXIT INT TERM USR1
  systems_stop_all_services "${DATA_ROOT}" || true
  exit "${status}"
}
trap cleanup EXIT INT TERM USR1
systems_write_runtime_env "${DATA_ROOT}"

task_json="$(run_python core -m lhmsb.longhorizon.online_matrix task \
  --run-dir "${RUN_DIR}" --task-index "${TASK_INDEX}")"
episode_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["episode_id"])' <<<"${task_json}")"
matrix_values="$(python3 - "${RUN_DIR}/MATRIX.json" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data["dataset"])
print(data["config"])
print(data["policy_profile"])
print(data["repository"]["code_commit"])
PY
)"
mapfile -t matrix_fields <<<"${matrix_values}"
DATASET="${matrix_fields[0]}"
CONFIG="${matrix_fields[1]}"
POLICY_PROFILE="${matrix_fields[2]}"
CODE_COMMIT="${matrix_fields[3]}"
TASK_OUT="${RUN_DIR}/results/task-$(printf '%04d' "${TASK_INDEX}")"
mkdir -p "${TASK_OUT}"

mapfile -t online_cells < <(
  python3 - "${RUN_DIR}/MATRIX.json" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
environments = {
    "workspace_only": "core",
    "flat_retrieval": "core",
    "mem0": "mem0",
    "amem": "amem",
    "memos": "memos",
}
for name in data["conditions"]:
    condition = "workspace_only" if name == "workspace_only" else "memory"
    backend = "none" if name == "workspace_only" else name
    print(environments[name], condition, backend)
PY
)
for cell in "${online_cells[@]}"; do
  read -r environment condition backend <<<"${cell}"
  result_id="$(run_python core -c \
    'from lhmsb.longhorizon.online_cli import online_result_id; import sys; print(online_result_id(episode_id=sys.argv[1], condition=sys.argv[2], backend=None if sys.argv[3] == "none" else sys.argv[3], policy_profile=sys.argv[4]))' \
    "${episode_id}" "${condition}" "${backend}" "${POLICY_PROFILE}")"
  result_path="${TASK_OUT}/${result_id}.json"
  if [[ -s "${result_path}" ]] && python3 - "${result_path}" "${CODE_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
repository = payload.get("repository", {})
raise SystemExit(
    0
    if repository.get("code_commit") == sys.argv[2]
    and repository.get("code_dirty") is False
    else 1
)
PY
  then
    printf 'resume: verified result already exists: %s\n' "${result_id}"
    continue
  fi
  arguments=(
    -m lhmsb.longhorizon.online_cli
    --dataset "${DATASET}" --config "${CONFIG}"
    --episode-id "${episode_id}" --policy-profile "${POLICY_PROFILE}"
    --condition "${condition}" --env-file "${ENV_FILE}"
    --data-root "${DATA_ROOT}" --steps-per-session 16
    --runtime-root "${RUN_DIR}/runtime/${LHMSB_SERVICE_INSTANCE}"
    --environment-steps-per-action 1 --out "${TASK_OUT}"
  )
  [[ "${backend}" != "none" ]] && arguments+=(--backend "${backend}")
  run_python "${environment}" "${arguments[@]}"
done
touch "${TASK_OUT}/DONE"
