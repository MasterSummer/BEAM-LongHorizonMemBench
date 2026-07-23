# Mem0 A100 服务器迁移与 Qualification 工作流

本文档是当前 Mem0 vertical slice 的操作手册。目标是在一台至少配备
两张 NVIDIA A100 的 Linux 服务器上，从冻结数据运行 Controlled 轨道，
并产出可验证的 memory write、retrieval、visible、causal use、behavior
和 drift 指标。服务器入口默认使用
`configs/experiments/mem0_controlled_zen.yaml`；当前 Controlled-Zen run
不包含 `mem0_native`。Live preflight、smoke 和 qualification
只在服务器上执行，不在本工作站执行。

当前阶段只启用 Mem0。Letta、Graphiti、Hindsight 和 MemOS 保留为后续系统，
不能混入本次 run identity 或结果目录。

## 1. 冻结的实验矩阵

正式数据是一个 16-session Software episode。运行矩阵包含三个 policy：

- `claude-opus-4-8` → OpenCode Zen；
- `deepseek-v4-pro` → 官方 DeepSeek API；
- `gpt-5.6-sol` → OpenCode Zen。

每个 policy 运行三个原子 condition：

| Condition | Memory 配置 | Policy 可见内容 |
|---|---|---|
| `workspace_only` | 无 memory | workspace |
| `oracle_current_state` | evaluator 当前最小 state | workspace + oracle state |
| `mem0_controlled` | policy model 同时作为 Mem0 extraction LLM；本地 BGE-M3 embedding | native-order 和 common-rerank 两个配对 readout |

因此共有 9 个可恢复的原子 task，并产生 12 个 condition/readout 结果：
Controlled task 的一次相同写入会生成 native order 与 common reranker 两个
读取分支。

本地 GPU 分配固定为：

```text
GPU 0 -> BAAI/bge-m3 embedding TEI
GPU 1 -> BAAI/bge-reranker-v2-m3 reranker TEI
```

三个 policy model 通过 provider API 调用，不在 A100 上本地推理。Claude
Opus 4.8 和 GPT-5.6 Sol 通过 OpenCode Zen，DeepSeek V4 Pro 通过官方
DeepSeek API。多于两张 GPU 不会自动扩大当前矩阵并行度；当前实现保持
task 顺序执行，以确保首轮 qualification 容易审计。

## 2. 版本与数据锁

关键冻结项如下：

| 组件 | 锁定值 |
|---|---|
| Mem0 | `mem0ai==2.0.12` |
| Mem0 source commit | `42cf18c4e6adb448e981aa1c7b55c1602b0cb670` |
| Mem0 wheel SHA-256 | `6b7e1afa466f6e14dd34b5e9222c159a69fad38f8d787e73adbf91dbb29e73e2` |
| BGE-M3 revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| BGE reranker revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Mem0 dataset release | `software-vertical-mem0-v0.2.0` |
| Dataset archive SHA-256 | `4a455e1a16cc66fa7c218ba48543174426ec710989a301de3fa61f694c170380` |
| Legacy v0.1 archive SHA-256 | `c1b35c1a554c2ad8d1e1f895a563a6bc5a67979b54b8857ce287468c2efe8130` |
| Qdrant image tag | `v1.15.4`，bootstrap 后转为 digest |
| TEI image tag | `1.8.0`，bootstrap 后转为 digest |

不要在正式 run 中临时替换模型 ID、embedding、reranker、Mem0 prompt、
`candidate_k=20` 或 `visible_k=5`。任何改变都应使用新配置和新 run name。

## 3. 服务器前置条件

服务器需要：

- Linux、Git、Python 3、`uv`；
- Docker Engine、Docker Compose v2；
- NVIDIA driver、`nvidia-smi` 和 NVIDIA Container Toolkit；
- 至少两张 Docker 可见的 NVIDIA GPU；
- 一个持久、可写的数据盘，默认挂载为 `/data/lhmsb`；
- 能访问 OCI registry、PyPI 和 Hugging Face 的 bootstrap 网络；
- 正式运行时能访问 OpenCode Zen 和 DeepSeek，或对应的显式
  `OPENCODE_ZEN_BASE_URL` / `DEEPSEEK_BASE_URL`；
- OpenCode Zen 和 DeepSeek 已为当前账号提供配置中声明的精确 model ID。

如果 provider 不提供某个精确 model ID，preflight 会失败。不要静默改用
邻近模型；应先修改并重新冻结实验配置。

## 4. 服务器目录

代码仓库和大文件数据根分开：

```text
<repo>/
├── configs/
├── datasets/releases/
├── deploy/
├── docker/
├── scripts/
├── src/
└── .env                         # 仅服务器持有，不提交

/data/lhmsb/
├── datasets/
│   ├── software_v1/
│   └── software_mem0_v2/
├── models/
│   ├── bge-m3/
│   └── bge-reranker-v2-m3/
├── wheelhouse/
├── images/
├── qdrant/                      # Qdrant 持久存储；task 以 collection 隔离
├── history/
│   └── preflight/
├── hf-cache/
├── manifests/
│   ├── host.json
│   ├── images.json
│   ├── models.json
│   └── wheels.json
├── runs/
│   ├── preflight/latest.json
│   └── mem0/<run-name>/
└── bundles/
```

正式 task 的 Mem0 history SQLite 位于：

```text
/data/lhmsb/runs/mem0/<run-name>/cells/tasks/<task-id>/store/history.sqlite
```

后续系统采用同一约定：

```text
configs/systems/<system>/
src/lhmsb/adapters/<system>_qualification.py
deploy/compose.<system>.yaml
/data/lhmsb/runs/<system>/<run-name>/
```

但只有对应系统完成独立 qualification 后，才应加入跨系统矩阵。

## 5. 推荐的在线迁移

### 5.1 拉取精确代码

在服务器上：

```bash
read -r -p 'Repository URL: ' LHMSB_REPOSITORY_URL
git clone "${LHMSB_REPOSITORY_URL}" BEAM-LongHorizonMemBench
cd BEAM-LongHorizonMemBench
git fetch --all --tags
read -r -p 'Published release commit SHA: ' LHMSB_RELEASE_SHA
test -n "${LHMSB_RELEASE_SHA}"
git checkout --detach "${LHMSB_RELEASE_SHA}"
test "$(git rev-parse HEAD)" = "${LHMSB_RELEASE_SHA}"
git status --short
```

不要以可移动的 `main` 名称直接启动正式实验。合并后先记录并发布精确 commit，
再把它填入 `LHMSB_RELEASE_SHA`。正式运行要求 `git status --short` 为空。记录：

```bash
git rev-parse HEAD
```

使用能够访问 Docker daemon 的非 root 账户执行 bootstrap。脚本会把该账户
的 UID/GID 写入 `.env`，worker container 以相同身份访问
`/data/lhmsb`，避免 bind mount 因固定容器 UID 而不可写。

首次使用默认 data root 时，先由管理员创建目录并交给该非 root 账户：

```bash
sudo install -d -m 0750 -o "$(id -un)" -g "$(id -gn)" /data/lhmsb
test -w /data/lhmsb
```

如果不能使用 `/data`，选择一个持久且当前账户可写的路径，并在后续所有
命令中传入相同的 `--data-root`。bootstrap 会在下载前显式拒绝不可写路径，
不会半途留下只完成一部分的资产目录。

### 5.2 配置 provider credentials

```bash
cp .env.example .env
chmod 600 .env
```

编辑 `.env`，至少填写：

```dotenv
OPENCODE_ZEN_API_KEY=...
OPENCODE_ZEN_BASE_URL=https://opencode.ai/zen
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

当前凭据只包括 OpenCode Zen 和官方 DeepSeek。若修改 endpoint，只修改相应
`*_BASE_URL`。密钥不会写入 run manifest、report 或离线 bundle。

### 5.3 一次性 bootstrap

```bash
scripts/bootstrap_server.sh \
  --data-root /data/lhmsb \
  --env-file .env
```

该命令会：

1. 校验并解压 v0.1 与 v0.2 数据；
2. 导出锁定依赖并建立 wheelhouse；
3. 校验 Mem0 wheel；
4. 下载两个固定 revision 的 BGE 模型并逐文件 hash；
5. 拉取 Qdrant、TEI 和 Python 镜像并解析 OCI digest；
6. 为 Qdrant/TEI 建立确定性本地归档别名，记录平台 image ID，并构建固定
   代码 commit 的非 root worker image；
7. 保存镜像归档和 image/model/wheel manifests；
8. 使用 `configs/experiments/mem0_controlled_zen.yaml` 运行不产生 provider
   调用的 repository-only preflight。

bootstrap 完成后，`.env` 中的 registry digest 与 runtime image ID 字段会被
写入解析后的值。
脚本会把 `.env` 权限固定为 `0600`。host manifest 同时记录 GPU UUID、
显存、driver version 和 compute capability。

### 5.4 Live preflight

```bash
scripts/preflight_mem0.sh \
  --data-root /data/lhmsb \
  --env-file .env
```

该步骤启动 Qdrant、embedding 和 reranker，并按固定顺序检查：

- repo、配置、两个冻结 release 和信息防火墙；
- Docker/Compose、两张 GPU、镜像 digest 和模型文件 hash；
- Qdrant 隔离生命周期；
- BGE-M3 输出维度必须为 1024；
- reranker 排序；
- OpenCode Zen 与 DeepSeek 的认证、精确模型身份和结构化输出；
- `mem0ai==2.0.12`；
- 三个 Controlled policy 各自完成一次真实 Mem0
  write → inventory → history → search；
- Mem0 内部 LLM 与 embedding 调用均被 trace；
- Controlled-Zen 配置和完整 trace/prompt contract。

Live preflight 会产生少量真实 provider 调用。成功标志：

```bash
jq '.ok, .stopped_at' /data/lhmsb/runs/preflight/latest.json
```

期望输出为 `true` 和 `null`。

### 5.5 四 session live smoke

```bash
scripts/run_mem0_smoke.sh \
  --data-root /data/lhmsb \
  --env-file .env \
  --run-name smoke-$(git rev-parse --short HEAD)
```

Smoke 会在服务器上确定性生成并冻结 4-session fixture，然后运行同一
9-task、12-result-cell 配置。它用于发现 provider、Mem0 或 trace 的集成
问题，不进入论文正式结果。

### 5.6 十六 session qualification

```bash
RUN_NAME=mem0-q1-$(git rev-parse --short HEAD)

scripts/run_mem0_qualification.sh \
  --data-root /data/lhmsb \
  --env-file .env \
  --run-name "${RUN_NAME}"
```

脚本会执行 `plan` 和 `run-matrix --keep-going`；矩阵完整时继续执行
`validate`。独立 task 发生失败时，其余 task 仍会继续，随后脚本以非零
状态停止，保留 partial report 供诊断。

## 6. Slurm 工作流

集群允许 compute node 运行 Docker 时，正式入口是单个 qualification job。
它会在同一个 allocation 中依次执行 image restore、host/GPU manifest、完整
live preflight、plan、matrix 和 validate，因此不会发生 preflight/qualification
并发竞态，也不会复用其他节点的硬件身份。

前提是仓库、`.env` 和 `/data/lhmsb` 对 compute node 可见，并且 5.3 的
bootstrap 已在这个共享 data root 生成 `images/{qdrant,tei,worker}.tar`。
job 会把这些归档加载到实际分配节点的 Docker daemon，校验加载后的 Qdrant、
TEI 与 worker image ID，再以 `pull_policy: never` 启动；因此不依赖 RepoDigest
能否被 `docker load` 恢复，也不会静默访问 registry。login node 与 compute
node 不需要共用本地镜像缓存。每个 job 使用独立的 Compose project 和
Qdrant namespace，并在退出时执行 `docker compose down`。共享的 host manifest
与 preflight report 由 `/data/lhmsb/locks/mem0-slurm.lock` 串行保护；若已有
另一个 Mem0 Slurm job，新的 job 会立即失败而不是并发污染状态。

```bash
export LHMSB_DATA_ROOT=/data/lhmsb
export LHMSB_ENV_FILE="$PWD/.env"
export LHMSB_RUN_NAME=mem0-q1-$(git rev-parse --short HEAD)

QUALIFICATION_JOB_ID="$(sbatch --parsable deploy/slurm/mem0_qualification.sbatch)"
printf 'qualification job: %s\n' "${QUALIFICATION_JOB_ID}"
```

默认资源是 `gpu:a100:2`。脚本从 `SLURM_JOB_GPUS` 取得两个全局 GPU ID，
分别传给 embedding 与 reranker；preflight 会拒绝重复 ID、未知 ID 或非 A100
设备。如果集群 GRES 名称不同，在提交时覆盖：

```bash
export LHMSB_SLURM_GRES="gpu:a100"
sbatch --gres="${LHMSB_SLURM_GRES}:2" \
  deploy/slurm/mem0_qualification.sbatch
```

如果集群不导出可用的 `SLURM_JOB_GPUS`，必须同时显式导出
`LHMSB_EMBEDDING_GPU_ID` 与 `LHMSB_RERANKER_GPU_ID`，且二者不同。独立的
`mem0_preflight.sbatch` 仅用于诊断；不要再把它与 qualification 无依赖地
并行提交。确需先单独诊断时，等待它成功后再提交 qualification：

```bash
PREFLIGHT_JOB_ID="$(sbatch --parsable deploy/slurm/mem0_preflight.sbatch)"
sbatch --dependency="afterok:${PREFLIGHT_JOB_ID}" \
  deploy/slurm/mem0_qualification.sbatch
```

qualification job 仍会在自己的节点重新执行一次完整 preflight，这是有意的
节点身份校验。Slurm 只负责调度；数据集、task identity、worker CLI 和报告
格式与 Compose 路径完全相同。

## 7. 恢复和重试

### 7.1 中断后恢复

使用相同 commit、`.env`、data root 和 run name 重新执行：

```bash
scripts/run_mem0_qualification.sh \
  --data-root /data/lhmsb \
  --env-file .env \
  --run-name "${RUN_NAME}"
```

已完成且 identity/hash 有效的 task result 会直接复用；task 内部的
session write、alignment 和 continuation cell 也会逐层复用。

### 7.2 查看失败 task

```bash
RUN_DIR=/data/lhmsb/runs/mem0/${RUN_NAME}

jq -r '
  select(.result.status != "complete")
  | [.result.task_id, .result.status, .result.error_class, .result.error_message]
  | @tsv
' "${RUN_DIR}"/results/*.json

jq -r \
  '[.task_index, .task_id, .policy_profile_id, .condition] | @tsv' \
  "${RUN_DIR}/tasks.jsonl"
```

### 7.3 重试一个失败 task

```bash
export LHMSB_LIVE_QUALIFICATION=1

docker compose --env-file .env -f deploy/compose.mem0.yaml run --rm worker \
  run-task \
  --run-dir "/data/lhmsb/runs/mem0/${RUN_NAME}" \
  --task-index <zero-based-index> \
  --force

docker compose --env-file .env -f deploy/compose.mem0.yaml run --rm worker \
  aggregate \
  --run-dir "/data/lhmsb/runs/mem0/${RUN_NAME}"

docker compose --env-file .env -f deploy/compose.mem0.yaml run --rm worker \
  validate \
  --report "/data/lhmsb/runs/mem0/${RUN_NAME}/report" \
  --json "/data/lhmsb/runs/mem0/${RUN_NAME}/validation.json"
```

`--force` 只授权重新执行该 task result；已有且 hash 正确的内部 cell 仍会
恢复。若代码、配置、数据、镜像、模型或硬件 identity 改变，系统会拒绝
续跑，此时必须新建 run name。

## 8. 输出与验收

Run 根目录包含：

```text
run_manifest.json
run_config.yaml
tasks.jsonl
results/
cells/tasks/<task-id>/
matrix-status.json
validation.json
report/
```

`report/` 中固定输出：

| 文件 | 用途 |
|---|---|
| `run_manifest.json` | 代码、数据、配置、镜像、模型、硬件及所有 artifact hash |
| `tasks.jsonl` | 9 个原子 task |
| `task_results.jsonl` | task 完整可移植结果及实际 store bytes |
| `memory_events.jsonl` | Mem0 ADD/UPDATE/DELETE/NONE/observed delta |
| `memory_inventory.jsonl` | 每个 checkpoint 的 `N_write`、`N_live` 和对象清单 |
| `retrieval_trace.jsonl` | candidate → retrieved → reranked 顺序和延迟 |
| `sceu_results.jsonl` | visible memory、action、checker、drift |
| `interventions.jsonl` | leave-one-out/replacement 的因果使用判断 |
| `api_usage.jsonl` | policy、Mem0 internal LLM、embedding、reranker 调用 |
| `metrics.json` | 全矩阵聚合指标及 numerator/denominator |
| `metrics_by_cell.json` | 每个 policy × condition × readout 的完整指标及 numerator/denominator |
| `summary.json` | task、trace 和 API call 数量 |
| `scorecard.csv` / `scorecard.md` | policy × condition × readout 行为与 drift 对比 |

最终检查：

```bash
RUN_DIR=/data/lhmsb/runs/mem0/${RUN_NAME}

jq . "${RUN_DIR}/matrix-status.json"
jq . "${RUN_DIR}/validation.json"
jq . "${RUN_DIR}/report/summary.json"
```

正式成功需要：

- `matrix-status.json` 中 `complete: true`；
- `missing_results: 0`；
- `non_complete_results: 0`；
- `validation.json` 中 `ok: true`；
- `summary.json` 中 `n_tasks: 9`；
- report artifact hash 与 trace ordering 全部通过。

## 9. 指标对应关系

### 9.1 Workspace 边际价值

- `mean_behavior_score`
- `behavior_correct_rate`
- `mem0_controlled_native_gain_beyond_workspace`
- `mem0_controlled_common_rerank_gain_beyond_workspace`
- 两个对应的 `*_oracle_gap_closed`
- `mem0_gain_beyond_workspace` / `oracle_gap_closed`（两个 Controlled
  readout cell 的宏平均，只作总览）
- `common_rerank_behavior_delta`

### 9.2 写入、handoff 与 state maintenance

- `write_coverage`
- `write_selectivity`
- `current_state_storage_precision`
- `current_state_storage_recall`
- `current_state_storage_f1`
- `stale_state_retention_rate`
- `duplicate_live_memory_rate`
- `update_delete_responsiveness`
- `write_to_continuation_alignment`
- `memory_write_count`（每个 final checkpoint 的平均累计 `N_write`）
- `live_memory_count`（每个 final checkpoint 的平均 `N_live`）
- `memory_write_count_total` / `live_memory_count_total`（审计总量）

### 9.3 Retrieval、visible 与 causal use

- `candidate_recall`
- `retrieval_precision` / `retrieval_recall` / `retrieval_f1`
- `retrieval_false_positive_rate`
- `retrieval_timeliness`
- `candidate_shortfall_rate`
- `visible_sufficiency`
- `visible_contamination`
- `stale_retrieval_rate`
- `retrieved_but_not_visible_rate`
- `visible_without_detected_unique_causal_effect_rate`（旧报告中的
  `visible_but_not_causally_used_rate` 仅作为兼容别名；无检测效应不等于未使用）
- `unique_causal_effect_rate`（canonical；稳定 intervention 改变 action 或
  checker 的比例）
- `causal_memory_use_rate`（兼容别名）
- `beneficial_intervention_rate`
- `harmful_intervention_rate`
- `ambiguous_intervention_rate`
- `unstable_intervention_rate`
- `leave_one_memory_out_action_flip_rate`

### 9.4 State evolution 与 long-horizon drift

- `state_conflict_resolution_accuracy`
- `stale_state_action_rate`
- `constraint_loss_rate`
- `current_plan_deviation_rate`
- `local_over_global_rate`
- `matched_early_late_behavioral_decay`
- `aggregate_drift_rate`

`state_conflict_resolution_accuracy` 的分母只包含真实冲突点：scope conflict、
valid update，以及已经存在 invalidated alternative 的 late matched-branch。
early matched baseline 不进入该分母。

这四个 drift component 分别对应：

1. 仍有效的约束逐渐失去行为影响；
2. 当前计划偏离已更新的全局计划；
3. 旧状态或已撤销分支重新支配动作；
4. 局部子目标错误覆盖全局目标。

### 9.5 成本与可靠性

- policy/internal LLM 的 input、output、cached、reasoning tokens；
- policy retry 与 terminal failure；
- embedding call、input count 和 latency；
- reranker call、candidate pair 和 latency；
- write、retrieval、rerank、policy latency；
- `qdrant_store_bytes`；
- `history_store_bytes`。

`qdrant_store_bytes` 是每个 task 独立 collection 在 task 完成后创建的
Qdrant 压缩 snapshot 大小；测量后 snapshot 会删除。
`history_store_bytes` 是关闭 Mem0 后 SQLite 主文件加仍存在的 WAL/SHM
sidecar 字节数。两个值是观测量，不是从 memory count 估算。

按 policy 或 condition 分析时，优先直接读取 `metrics_by_cell.json`；
`task_results.jsonl`、`sceu_results.jsonl` 和 `scorecard.csv` 保留相同的
`policy_profile_id/condition/readout` 分组键供逐记录审计。`metrics.json`
是全矩阵汇总，不应被误解为单一 track 的结果。

RQ5 的 scale 变量始终是 `memory_write_count` 和 `live_memory_count`，不是
token 数。当前单 episode qualification 验证计数和选择性契约；正式的
多 memory-count scaling curve 仍属于下一阶段 pilot。

## 10. 结果回传与封存

建议同时保存 report 和完整可恢复 run：

```bash
rsync -a \
  server:/data/lhmsb/runs/mem0/${RUN_NAME}/ \
  runs/server/mem0/${RUN_NAME}/

sha256sum \
  runs/server/mem0/${RUN_NAME}/report/run_manifest.json \
  runs/server/mem0/${RUN_NAME}/report/metrics.json
```

不要只复制 `scorecard.csv`。论文审计至少需要 report 的全部 JSONL、两个
manifest、validation 结果和 task cells。

## 11. 可选离线依赖 bundle

在已经完成 online bootstrap 的机器上：

```bash
scripts/build_offline_bundle.sh \
  --data-root /data/lhmsb \
  --out /data/lhmsb/bundles/lhmsb-mem0-qualification.tar.gz
```

Bundle 包含代码归档、commit、wheelhouse、OCI image archives、两个 BGE
模型、manifests 和两个冻结数据 release，并带确定性 SHA-256 sidecar。
它不包含 `.env` 或任何 credential。即使依赖通过 bundle 离线迁移，
正式 qualification 仍需要 OpenCode Zen 和 DeepSeek endpoint 的网络访问。

## 12. Mem0 通过后的 decision gate

1. 冻结 Mem0 的 raw run 和 validated report；
2. 检查三条 policy 及 Controlled native/common-rerank readout 的可区分性；
3. 审计至少一条完整
   `stored → candidate → retrieved → visible → causal use → behavior`
   链；
4. 决定是否扩大 episode 数和 memory-count 梯度；
5. 下一 memory system 待定；基于 Mem0 结果和届时的系统调查单独完成
   selection、design 和 qualification，不在本阶段预先指定；
6. 在下一系统被明确选择并独立 qualification 通过之前，不启动跨系统
   pilot。
