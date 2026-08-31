#!/usr/bin/env bash
set -euo pipefail

# One entry point for validating, resuming, executing, and finalizing the
# contribution-linked AAAI rebuttal experiment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="run"
BACKEND="slurm"
DATA_ROOT="${LHMSB_DATA_ROOT:-/data/lhmsb}"
RUN_NAME="${LHMSB_ONLINE_RUN_NAME:-memhorizon-aaai-rebuttal}"
SOURCE_PLAN="${LHMSB_REBUTTAL_SOURCE_PLAN:-${CONTROL_REPO}/tmp/server_evidence_20260728/main}"
CALIBRATION_DIR="${LHMSB_REBUTTAL_CALIBRATION_DIR:-${CONTROL_REPO}/tmp/server_evidence_20260728/calibration}"
POLICY_REPO=""
DATASET=""
CONFIG=""
ENV_FILE="${LHMSB_ENV_FILE:-${CONTROL_REPO}/.env}"
MAX_PARALLEL=1
DRY_RUN=0
SKIP_TESTS=0
SKIP_REBIND=0
LEGACY_RESULTS=()

usage() {
  cat <<'EOF'
Usage: scripts/run_aaai_rebuttal_experiment.sh [MODE] [options]

Modes:
  run       validate + prepare + execute (default)
  validate  run every zero-API preflight gate
  prepare   install the frozen matrix and import compatible saved results
  execute   run only tasks that still have missing result cells
  finalize  aggregate and zero-API audit a complete 100-cell matrix

Options:
  --backend local|slurm
  --data-root PATH
  --run-name NAME
  --source-plan PATH
  --calibration-dir PATH
  --legacy-results PATH     repeat for multiple saved batches
  --policy-repo PATH        clean checkout at the matrix commit; auto-created if omitted
  --dataset PATH
  --config PATH
  --env-file PATH
  --max-parallel N          Slurm array concurrency (default: 1)
  --skip-tests              skip the local zero-API pytest gate
  --skip-rebind             keep dataset/config paths stored in the source matrix
  --dry-run                 prepare and validate, but print paid execution commands
EOF
}

if (($#)) && [[ "$1" != --* ]]; then
  MODE="$1"
  shift
fi
while (($#)); do
  case "$1" in
    --backend) BACKEND="${2:?missing value for --backend}"; shift 2 ;;
    --data-root) DATA_ROOT="${2:?missing value for --data-root}"; shift 2 ;;
    --run-name) RUN_NAME="${2:?missing value for --run-name}"; shift 2 ;;
    --source-plan) SOURCE_PLAN="${2:?missing value for --source-plan}"; shift 2 ;;
    --calibration-dir) CALIBRATION_DIR="${2:?missing value for --calibration-dir}"; shift 2 ;;
    --legacy-results) LEGACY_RESULTS+=("${2:?missing value for --legacy-results}"); shift 2 ;;
    --policy-repo) POLICY_REPO="${2:?missing value for --policy-repo}"; shift 2 ;;
    --dataset) DATASET="${2:?missing value for --dataset}"; shift 2 ;;
    --config) CONFIG="${2:?missing value for --config}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?missing value for --env-file}"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="${2:?missing value for --max-parallel}"; shift 2 ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-rebind) SKIP_REBIND=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${MODE}" in run|validate|prepare|execute|finalize) ;; *) usage >&2; exit 2 ;; esac
case "${BACKEND}" in local|slurm) ;; *) printf 'backend must be local or slurm\n' >&2; exit 2 ;; esac
[[ "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'max-parallel must be a positive integer\n' >&2
  exit 2
}

RUN_DIR="${DATA_ROOT}/runs/online/${RUN_NAME}"
AUDIT_DIR="${RUN_DIR}/rebuttal-audits"
if [[ -x "${CONTROL_REPO}/.venv/bin/python" ]]; then
  CONTROL_PYTHON="${CONTROL_REPO}/.venv/bin/python"
elif [[ -x "${DATA_ROOT}/venvs/core/bin/python" ]]; then
  CONTROL_PYTHON="${DATA_ROOT}/venvs/core/bin/python"
else
  CONTROL_PYTHON="$(command -v python3)"
fi

control_python() {
  PYTHONPATH="${CONTROL_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}" "${CONTROL_PYTHON}" "$@"
}

require_clean_audit_repo() {
  [[ "${DRY_RUN}" == "1" ]] && return
  local dirty
  dirty="$(git -C "${CONTROL_REPO}" status --porcelain)"
  [[ -z "${dirty}" ]] || {
    printf 'audit repository is dirty; commit the experiment code before paid execution: %s\n' \
      "${CONTROL_REPO}" >&2
    exit 1
  }
}

matrix_value() {
  control_python - "${RUN_DIR}/MATRIX.json" "$1" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

ensure_policy_repo() {
  local expected_commit
  expected_commit="$(control_python - "${SOURCE_PLAN}/MATRIX.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["repository"]["code_commit"])
PY
)"
  if [[ -z "${POLICY_REPO}" ]]; then
    POLICY_REPO="${DATA_ROOT}/worktrees/memhorizon-policy-${expected_commit:0:12}"
    if [[ ! -d "${POLICY_REPO}/.git" && ! -f "${POLICY_REPO}/.git" ]]; then
      if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY-RUN git -C %q worktree add --detach %q %q\n' \
          "${CONTROL_REPO}" "${POLICY_REPO}" "${expected_commit}"
      else
        mkdir -p "$(dirname "${POLICY_REPO}")"
        git -C "${CONTROL_REPO}" worktree add --detach "${POLICY_REPO}" "${expected_commit}"
      fi
    fi
  fi
  if [[ -d "${POLICY_REPO}" ]]; then
    local actual_commit dirty
    actual_commit="$(git -C "${POLICY_REPO}" rev-parse HEAD)"
    dirty="$(git -C "${POLICY_REPO}" status --porcelain)"
    [[ "${actual_commit}" == "${expected_commit}" ]] || {
      printf 'policy repo commit mismatch: %s != %s\n' "${actual_commit}" "${expected_commit}" >&2
      exit 1
    }
    [[ -z "${dirty}" ]] || {
      printf 'policy repo is dirty: %s\n' "${POLICY_REPO}" >&2
      exit 1
    }
  elif [[ "${DRY_RUN}" != "1" ]]; then
    printf 'policy repo does not exist: %s\n' "${POLICY_REPO}" >&2
    exit 1
  fi
  DATASET="${DATASET:-${DATA_ROOT}/datasets/software_longitudinal_v014}"
  CONFIG="${CONFIG:-${POLICY_REPO}/configs/experiments/systems_controlled_gpt_only_longitudinal_v014_shengsuanyun_writer.yaml}"
}

run_zero_api_validation() {
  mkdir -p "${AUDIT_DIR}"
  control_python -m lhmsb.longhorizon.rebuttal_validation \
    --calibration-dir "${CALIBRATION_DIR}" \
    --out "${AUDIT_DIR}/calibration_validation.json"
  control_python -m lhmsb.longhorizon.known_fault_validation \
    --out-dir "${AUDIT_DIR}/known-fault"
  if [[ -f "${DATASET}/MANIFEST.json" && -f "${RUN_DIR}/tasks.jsonl" ]]; then
    control_python -m lhmsb.longhorizon.construction_validation \
      --dataset "${DATASET}" --run-dir "${RUN_DIR}" \
      --out-dir "${AUDIT_DIR}/construction"
  elif [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN skip C2 reconstruction ledger because dataset is unavailable: %s\n' \
      "${DATASET}"
  else
    printf 'C2 validation dataset or frozen task ledger is missing\n' >&2
    exit 1
  fi
  if [[ "${SKIP_TESTS}" != "1" ]]; then
    PYTHONPATH="${CONTROL_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${CONTROL_PYTHON}" -m pytest -q \
      "${CONTROL_REPO}/tests/longhorizon/test_known_fault_validation.py" \
      "${CONTROL_REPO}/tests/longhorizon/test_construction_validation.py" \
      "${CONTROL_REPO}/tests/longhorizon/test_rebuttal_manifest.py" \
      "${CONTROL_REPO}/tests/longhorizon/test_rebuttal_validation.py" \
      "${CONTROL_REPO}/tests/longhorizon/test_online_matrix.py"
  fi
}

prepare_run() {
  local arguments=(
    -m lhmsb.longhorizon.rebuttal_manifest
    --run-dir "${RUN_DIR}"
    --source-plan "${SOURCE_PLAN}"
  )
  local root
  for root in "${LEGACY_RESULTS[@]}"; do
    arguments+=(--legacy-results "${root}")
  done
  if [[ "${SKIP_REBIND}" != "1" ]]; then
    arguments+=(--dataset "${DATASET}" --config "${CONFIG}")
  fi
  control_python "${arguments[@]}"
}

missing_indices_csv() {
  control_python - "${RUN_DIR}/EXPERIMENT_MANIFEST.json" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(",".join(str(item) for item in data["missing_task_indices"]))
PY
}

run_task() {
  local task_index="$1"
  local command=(
    "${POLICY_REPO}/scripts/run_online_longhorizon.sh" task
    --data-root "${DATA_ROOT}"
    --env-file "${ENV_FILE}"
    --run-name "${RUN_NAME}"
    --task-index "${task_index}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    LHMSB_REPO_ROOT="${POLICY_REPO}" "${command[@]}"
  fi
}

execute_missing() {
  local indices
  indices="$(missing_indices_csv)"
  if [[ -z "${indices}" ]]; then
    printf 'No missing online result cells; proceeding to finalization.\n'
    finalize_run
    return
  fi
  if [[ "${BACKEND}" == "local" ]]; then
    local task_index
    while IFS= read -r task_index; do
      [[ -n "${task_index}" ]] && run_task "${task_index}"
    done < <(tr ',' '\n' <<<"${indices}")
    [[ "${DRY_RUN}" == "1" ]] || finalize_run
    return
  fi
  local array_spec="${indices}%${MAX_PARALLEL}"
  local export_values="ALL,LHMSB_REPO_ROOT=${POLICY_REPO},LHMSB_DATA_ROOT=${DATA_ROOT},LHMSB_ONLINE_RUN_NAME=${RUN_NAME},LHMSB_ENV_FILE=${ENV_FILE}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY-RUN sbatch --array=%q --export=%q %q\n' \
      "${array_spec}" "${export_values}" \
      "${POLICY_REPO}/deploy/slurm/online_longhorizon_array.sbatch"
    printf 'DRY-RUN submit finalize job with afterok dependency on the array job\n'
    return
  fi
  command -v sbatch >/dev/null || { printf 'sbatch is unavailable\n' >&2; exit 1; }
  local array_job finalize_job
  array_job="$(sbatch --parsable --array="${array_spec}" --export="${export_values}" \
    "${POLICY_REPO}/deploy/slurm/online_longhorizon_array.sbatch")"
  finalize_job="$(sbatch --parsable --dependency="afterok:${array_job}" \
    --export="ALL,LHMSB_CONTROL_REPO=${CONTROL_REPO},LHMSB_POLICY_REPO=${POLICY_REPO},LHMSB_DATA_ROOT=${DATA_ROOT},LHMSB_ONLINE_RUN_NAME=${RUN_NAME},LHMSB_ENV_FILE=${ENV_FILE},LHMSB_SYSTEM_DATASET=${DATASET},LHMSB_REBUTTAL_SOURCE_PLAN=${SOURCE_PLAN}" \
    "${CONTROL_REPO}/deploy/slurm/aaai_rebuttal_finalize.sbatch")"
  printf 'Submitted missing-task array job %s and finalize job %s.\n' \
    "${array_job}" "${finalize_job}"
}

finalize_run() {
  local completion
  completion="$(control_python -m lhmsb.longhorizon.rebuttal_manifest --run-dir "${RUN_DIR}")"
  control_python - "${completion}" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
if not data.get("complete"):
    raise SystemExit(f"matrix remains incomplete: {data.get('missing_result_count')} cells")
PY
  LHMSB_DATA_ROOT="${DATA_ROOT}" LHMSB_ONLINE_RUN_NAME="${RUN_NAME}" \
    LHMSB_REPO_ROOT="${POLICY_REPO}" LHMSB_ENV_FILE="${ENV_FILE}" \
    "${POLICY_REPO}/scripts/run_online_longhorizon.sh" aggregate \
      --data-root "${DATA_ROOT}" --run-name "${RUN_NAME}"
  control_python -m lhmsb.longhorizon.online_audit \
    --dataset "${DATASET}" --run-dir "${RUN_DIR}/results" \
    --out "${AUDIT_DIR}/terminal_audit.json"
  local expected_count
  expected_count="$(matrix_value expected_result_count)"
  control_python -m lhmsb.longhorizon.rebuttal_validation \
    --terminal-audit "${AUDIT_DIR}/terminal_audit.json" \
    --expected-result-count "${expected_count}" \
    --out "${AUDIT_DIR}/terminal_audit_validation.json"
  control_python -m lhmsb.longhorizon.terminal_audit_summary \
    --audit "${AUDIT_DIR}/terminal_audit.json" \
    --out-dir "${AUDIT_DIR}/terminal-summary"
  printf 'Complete: %s\n' "${RUN_DIR}"
}

case "${MODE}" in
  validate)
    ensure_policy_repo
    prepare_run
    run_zero_api_validation
    ;;
  prepare)
    ensure_policy_repo
    prepare_run
    ;;
  execute)
    require_clean_audit_repo
    ensure_policy_repo
    execute_missing
    ;;
  finalize)
    require_clean_audit_repo
    ensure_policy_repo
    finalize_run
    ;;
  run)
    require_clean_audit_repo
    ensure_policy_repo
    prepare_run
    run_zero_api_validation
    execute_missing
    ;;
esac
