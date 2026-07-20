#!/usr/bin/env bash

# Lifecycle manager for loopback-only native Qdrant, Neo4j, and two TEI
# processes. Each run owns isolated state, ports, PID identities, and logs.

systems_service_root() {
  local data_root="$1"
  printf '%s\n' "${data_root}/services/${LHMSB_SERVICE_INSTANCE:-manual}"
}

systems_allocate_ports() {
  local data_root="$1"
  local root
  root="$(systems_service_root "${data_root}")"
  local output="${root}/ports.json"
  mkdir -p "${root}" "${data_root}/locks"
  exec 8>"${data_root}/locks/ports.lock"
  flock 8
  python3 - "${data_root}/services" "${output}" <<'PY'
import json
import socket
import sys
from pathlib import Path

services_root = Path(sys.argv[1])
output = Path(sys.argv[2])
used: set[int] = set()
for path in services_root.glob("*/ports.json"):
    if path == output:
        continue
    try:
        used.update(int(value) for value in json.loads(path.read_text()).values())
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        continue

names = ("qdrant_http", "qdrant_grpc", "neo4j_bolt", "neo4j_http", "embedding", "reranker")
chosen: dict[str, int] = {}
for name in names:
    for port in range(20000, 50000):
        if port in used:
            continue
        with socket.socket() as handle:
            try:
                handle.bind(("127.0.0.1", port))
            except OSError:
                continue
        chosen[name] = port
        used.add(port)
        break
    else:
        raise SystemExit("no free loopback ports available")

temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(chosen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
PY
  flock -u 8
  systems_export_ports "${data_root}"
}

systems_export_ports() {
  local data_root="$1"
  local ports
  ports="$(python3 - "$(systems_service_root "${data_root}")/ports.json" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
names = ("qdrant_http", "qdrant_grpc", "neo4j_bolt", "neo4j_http", "embedding", "reranker")
print("\t".join(str(data[name]) for name in names))
PY
  )"
  IFS=$'\t' read -r \
    LHMSB_QDRANT_HTTP_PORT LHMSB_QDRANT_GRPC_PORT \
    LHMSB_NEO4J_BOLT_PORT LHMSB_NEO4J_HTTP_PORT \
    LHMSB_EMBEDDING_PORT LHMSB_RERANKER_PORT <<<"${ports}"
  export LHMSB_QDRANT_HTTP_PORT LHMSB_QDRANT_GRPC_PORT
  export LHMSB_NEO4J_BOLT_PORT LHMSB_NEO4J_HTTP_PORT
  export LHMSB_EMBEDDING_PORT LHMSB_RERANKER_PORT
  export LHMSB_QDRANT_URL="http://127.0.0.1:${LHMSB_QDRANT_HTTP_PORT}"
  export LHMSB_NEO4J_URI="bolt://127.0.0.1:${LHMSB_NEO4J_BOLT_PORT}"
  export LHMSB_EMBEDDING_URL="http://127.0.0.1:${LHMSB_EMBEDDING_PORT}"
  export LHMSB_RERANKER_URL="http://127.0.0.1:${LHMSB_RERANKER_PORT}"
}

systems_process_start_time() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  awk '{print $22}' "/proc/${pid}/stat"
}

systems_record_pid() {
  local data_root="$1"
  local name="$2"
  local pid="$3"
  local start_time
  start_time="$(systems_process_start_time "${pid}")"
  local root
  root="$(systems_service_root "${data_root}")"
  PID_VALUE="${pid}" START_TIME="${start_time}" NAME_VALUE="${name}" \
    python3 - "${root}/${name}.pid.json" <<'PY'
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = {
    "name": os.environ["NAME_VALUE"],
    "pid": int(os.environ["PID_VALUE"]),
    "proc_start_time": os.environ["START_TIME"],
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

systems_pid_identity() {
  local pid_file="$1"
  [[ -s "${pid_file}" ]] || return 1
  local identity
  identity="$(python3 - "${pid_file}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"{int(data['pid'])}\t{data['proc_start_time']}")
PY
  )"
  local pid recorded current
  IFS=$'\t' read -r pid recorded <<<"${identity}"
  current="$(systems_process_start_time "${pid}" 2>/dev/null || true)"
  [[ -n "${current}" && "${current}" == "${recorded}" ]] || return 1
  printf '%s\n' "${pid}"
}

systems_stop_service() {
  local data_root="$1"
  local name="$2"
  local pid_file
  pid_file="$(systems_service_root "${data_root}")/${name}.pid.json"
  local pid
  pid="$(systems_pid_identity "${pid_file}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]]; then
    kill -TERM "${pid}" 2>/dev/null || true
    local attempt
    for attempt in {1..50}; do
      systems_pid_identity "${pid_file}" >/dev/null 2>&1 || break
      sleep 0.2
    done
    if systems_pid_identity "${pid_file}" >/dev/null 2>&1; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${pid_file}"
}

systems_stop_all_services() {
  local data_root="$1"
  systems_stop_service "${data_root}" reranker
  systems_stop_service "${data_root}" embedding
  systems_stop_service "${data_root}" neo4j
  systems_stop_service "${data_root}" qdrant
  rm -f "$(systems_service_root "${data_root}")/ports.json"
}

systems_wait_http() {
  local name="$1"
  local url="$2"
  local attempt
  for attempt in {1..60}; do
    if curl --fail --silent --show-error --max-time 2 "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  printf '%s health check timed out: %s\n' "${name}" "${url}" >&2
  return 1
}

systems_wait_neo4j() {
  local attempt
  for attempt in {1..60}; do
    if env -i \
      HOME="${HOME}" PATH="${LHMSB_JAVA_HOME}/bin:/usr/bin:/bin" \
      JAVA_HOME="${LHMSB_JAVA_HOME}" \
      "${LHMSB_NEO4J_HOME}/bin/cypher-shell" \
      -a "${LHMSB_NEO4J_URI}" -u neo4j -p "${LHMSB_NEO4J_PASSWORD}" \
      'RETURN 1;' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  printf 'Neo4j health check timed out\n' >&2
  return 1
}

systems_health_services() {
  systems_wait_http qdrant "${LHMSB_QDRANT_URL}/healthz"
  systems_wait_neo4j
  systems_wait_http embedding "${LHMSB_EMBEDDING_URL}/health"
  systems_wait_http reranker "${LHMSB_RERANKER_URL}/health"
}

systems_start_qdrant() {
  local data_root="$1"
  local root
  root="$(systems_service_root "${data_root}")"
  local state="${data_root}/qdrant/${LHMSB_SERVICE_INSTANCE:-manual}"
  mkdir -p "${state}" "${root}/logs"
  (
    cd "${state}"
    exec env -i HOME="${HOME}" PATH="/usr/bin:/bin" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
      QDRANT__SERVICE__HOST=127.0.0.1 \
      QDRANT__SERVICE__HTTP_PORT="${LHMSB_QDRANT_HTTP_PORT}" \
      QDRANT__SERVICE__GRPC_PORT="${LHMSB_QDRANT_GRPC_PORT}" \
      QDRANT__STORAGE__STORAGE_PATH="${state}" \
      "${LHMSB_QDRANT_BIN}"
  ) >"${root}/logs/qdrant.log" 2>&1 &
  systems_record_pid "${data_root}" qdrant "$!"
}

systems_start_neo4j() {
  local data_root="$1"
  local root
  root="$(systems_service_root "${data_root}")"
  local state="${data_root}/neo4j/${LHMSB_SERVICE_INSTANCE:-manual}"
  local configuration="${state}/conf"
  mkdir -p \
    "${configuration}" "${state}/data" "${state}/logs" "${state}/run" \
    "${state}/import" "${state}/plugins" "${root}/logs"
  cat >"${configuration}/neo4j.conf" <<EOF
server.default_listen_address=127.0.0.1
server.bolt.listen_address=127.0.0.1:${LHMSB_NEO4J_BOLT_PORT}
server.http.listen_address=127.0.0.1:${LHMSB_NEO4J_HTTP_PORT}
server.directories.data=${state}/data
server.directories.logs=${state}/logs
server.directories.run=${state}/run
server.directories.import=${state}/import
server.directories.plugins=${state}/plugins
server.memory.heap.initial_size=1g
server.memory.heap.max_size=4g
server.memory.pagecache.size=2g
EOF
  local password_file="${state}/initial-password"
  if [[ ! -s "${password_file}" ]]; then
    printf 'lhmsb-%s\n' "$(printf '%s' "${LHMSB_SERVICE_INSTANCE:-manual}" | sha256sum | cut -c1-24)" \
      >"${password_file}"
    chmod 600 "${password_file}"
    env -i HOME="${HOME}" PATH="${LHMSB_JAVA_HOME}/bin:/usr/bin:/bin" \
      JAVA_HOME="${LHMSB_JAVA_HOME}" NEO4J_CONF="${configuration}" \
      "${LHMSB_NEO4J_HOME}/bin/neo4j-admin" dbms set-initial-password \
      "$(<"${password_file}")" >/dev/null
  fi
  export LHMSB_NEO4J_PASSWORD
  LHMSB_NEO4J_PASSWORD="$(<"${password_file}")"
  export LHMSB_NEO4J_PASSWORD
  env -i HOME="${HOME}" PATH="${LHMSB_JAVA_HOME}/bin:/usr/bin:/bin" \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 JAVA_HOME="${LHMSB_JAVA_HOME}" \
    NEO4J_CONF="${configuration}" \
    "${LHMSB_NEO4J_HOME}/bin/neo4j" console \
    >"${root}/logs/neo4j.log" 2>&1 &
  systems_record_pid "${data_root}" neo4j "$!"
}

systems_start_tei() {
  local data_root="$1"
  local name="$2"
  local gpu="$3"
  local model_dir="$4"
  local port="$5"
  local root
  root="$(systems_service_root "${data_root}")"
  mkdir -p "${root}/logs"
  env -i HOME="${HOME}" PATH="/usr/bin:/bin" LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CUDA_VISIBLE_DEVICES="${gpu}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${LHMSB_TEI_BIN}" \
    --model-id "${model_dir}" --hostname 127.0.0.1 --port "${port}" \
    --dtype float16 >"${root}/logs/${name}.log" 2>&1 &
  systems_record_pid "${data_root}" "${name}" "$!"
}

systems_start_all_services() {
  local data_root="$1"
  # MemOS otherwise defaults to ``$PWD/.memos`` and dirties the clean source
  # checkout before the immutable run identity is recorded.
  export MEMOS_BASE_PATH="${MEMOS_BASE_PATH:-${data_root}/memos}"
  export LHMSB_MEMOS_TOKENIZER_PATH="${LHMSB_MEMOS_TOKENIZER_PATH:-character}"
  mkdir -p "${MEMOS_BASE_PATH}"
  systems_stop_all_services "${data_root}"
  systems_allocate_ports "${data_root}"
  systems_start_qdrant "${data_root}"
  systems_start_neo4j "${data_root}"
  systems_start_tei "${data_root}" embedding "${LHMSB_EMBEDDING_GPU_ID}" \
    "${LHMSB_EMBEDDING_MODEL_DIR}" "${LHMSB_EMBEDDING_PORT}"
  systems_start_tei "${data_root}" reranker "${LHMSB_RERANKER_GPU_ID}" \
    "${LHMSB_RERANKER_MODEL_DIR}" "${LHMSB_RERANKER_PORT}"
  if ! systems_health_services; then
    systems_stop_all_services "${data_root}"
    return 1
  fi
}
