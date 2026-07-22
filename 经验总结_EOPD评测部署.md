# EOPD 评测 / 部署 经验总结（Lessons Learned）

> 适用：在新 VM / apex 等机器上复现 EOPD 论文 Table 2（6 个数学基准：MATH500 / AMC23 / Minerva / OlympiadBench / AIME24 / AIME25）。
> 本文件汇总协作规范、提速、已知 bug 与修复、环境约定、诊断姿势，供后续终端/他人直接照做。

---

## 1. 协作规范（最重要）

- **指令必须写成仓库内 `指令_*.txt` 再提交 git**，远程机器 `git pull` 后 `bash 指令_*.txt` 执行。
  - 包括**诊断命令、修复补丁的 bash 步骤**也要走 txt，不只训练/评测启动流程。
  - 简单的单行确认命令可仍在聊天里给；但训练 / 评测 / 同步 / 诊断 / 修复一律走 txt。
- **脚本要加进度条（tqdm）**，让用户能安心等待：
  - 长耗时脚本（评测生成、评分）必须能实时显示进度（如 `benchmarks x/6` + 每 benchmark 百分比）。
  - 已落地：`score_avg_pass_at_k.py` 有双层 tqdm（外层 benchmark 计数 + 内层逐题）。生成脚本靠 `launched shard 0..7` 日志 + 分片 log。
- 改动任何脚本后**提交并 push**，再让远程 `git pull`，不要只在聊天里贴改完的代码。

---

## 2. 评测提速：多卡用「数据并行」而非张量并行

- **1.7B 这种小模型用 `tensor_parallel_size>1` 反而更慢**（通信开销盖过计算）。正确提速 = 数据并行。
- 做法：`NUM_SHARDS=N`（= 同时启动的独立 vLLM 实例数，各占 1 卡、各 `TP=1`），把 prompts 切 N 份并发生成，约 N 倍速。
  - VM（8 卡）：`NUM_SHARDS=8`；apex（1 卡）：`NUM_SHARDS=1` 或 `auto`。
  - 实现：`generate_offline_vllm.py` 的 `--num_shards/--shard_id/--merge`；`eval_six_benchmarks.sh` 每 benchmark 启 N 个进程 + 合并。
- **单看某个 shard 日志速度仍 ~824 toks/s 是正常的**——加速来自 N 并发，不是单实例变快。验证 8 卡是否生效看 `nvidia-smi`（应 8 张卡都有占用、8 个 python 进程）。

---

## 3. 已知 Bug 与修复（都已在仓库，记录备查）

| 问题 | 根因 | 修复 | commit |
|---|---|---|---|
| 评分合并出的 merged parquet 全不存在，最后评分读文件崩溃 | `generate_offline_vllm.py` 里 `--model_path/--data_path` 标成 `required=True`，而 `--merge` 模式不传这俩，`parse_args()` 在进 merge 分支前报错退出，`merge_shards` 从未被调用（脚本没检查合并退出码，静默跳过） | 这两个参数改为非必填；生成模式内再加校验 | `44bd877` |
| 评分时单条坏数据拖垮整轮 | `compute_score` 异常未捕获；`responses`/`ground_truth` 类型未兜底 | 逐条 `try/except` 记 0 分；`responses` None/字符串/空 安全跳过 | `21a8492` |
| 评分刷大量 `We did not manage to extract a prediction` WARNING | math_verify 对无法解析的答案打 WARNING | 把 `math_verify`/`verl` logger 及 root 设为 `ERROR`，只保留最终表 | `db1dc5e` |
| 评测脚本依赖已 broken 的 `verl.trainer.main_generation` | 该 fork 的 vLLM async rollout 生成路径不可用 | 改用离线 vLLM（`generate_offline_vllm.py`，直接 `vllm.LLM` 同步生成） | — |
| 训练存盘的 HF 权重只有 config/tokenizer，无 `model.safetensors` | 训练自动导出 HF 权重失败 | 手动 `python -m verl.model_merger merge --backend fsdp` 合并 FSDP 分片 → `actor/huggingface/` | — |
| 评分无进度条、干等焦虑 | 脚本只有末尾打印 | 加 tqdm 双层进度条（带 tqdm 缺失兜底） | `13cad5b` |

**分片合并顺序还原**：每个 shard 写 `sub["__idx__"] = row_idx`（全局行号），`merge_shards` 按 `__idx__` 排序后丢弃该列，保证合并后顺序与原始一致。

---

## 4. 环境与机器约定

- **必须用 eopd 环境的 python**（`/opt/conda/envs/eopd/bin/python3` 或 `conda activate eopd`），否则缺 DTensor / math_verify，整批报错。脚本已做自动回退到 eopd 绝对路径。
- **eopd 环境必须装 `vllm`**（离线评测依赖）；评测前用预检确认 `vllm / math_verify / transformers / pandas` 都在。
- 多卡容器里设 `NCCL_P2P_DISABLE=1`，避免 NCCL peer access 不可用（仅略慢，正确性不受影响）。
- **VM**：8 卡，路径 `REPO=/inspire/.../CacheOPD/eopd-baseline`，数据 `/inspire/.../dk/data`，ckpt `/root/ckpts/EOPD`，评测 `NUM_SHARDS=8`。
- **apex**：SSH 走代理；EOPD 评测已跑完可作参照；1 卡机器 `NUM_SHARDS=1`。

---

## 5. 评测产物与重跑策略（避免浪费时间）

- 生成很慢（6 基准合计 ~1.5h+，其中 OlympiadBench 2126 题是大头），**生成产物 `eval_gen/<BENCH>.parquet.shard*.parquet` 要保留**。
- 若只评分/合并失败，**只重跑那一步，不要重生成**：
  - 合并：`generate_offline_vllm.py --merge --input_glob ".../<B>.parquet.shard*.parquet" --output_path ".../<B>.parquet"`
  - 评分：`score_avg_pass_at_k.py --input B=.../<B>.parquet ...`（直接读 merged，写 `eval_results/*_scores.{txt,json}`）
- 评分读 `responses` 列 + `reward_model.ground_truth`，用 `math_verify` 抽 `\boxed{}` 比对算 Avg@8 / Pass@8。

---

## 6. 诊断与「安心等待」的正确姿势

- **看 GPU/进程状态**：
  - `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv` 看几张卡在算。
  - `pgrep -af <脚本名>` 看进程在不在；`ps -o pid,etime,time,pcpu -p <PID>` 两次对比，`TIME`（累计 CPU 时间）在涨 = 在持续推进。
- **无进度条时别误判卡死**：评分是**单线程 CPU** 跑 math_verify，OlympiadBench 1.7 万条回答要几分钟~十几分钟，期间终端静默、CPU 吃满是正常的。`pgrep` 返回空 = 结束。
- **`/proc/<PID>/fd` 看不到 parquet 是正常的**：parquet 被 `read_parquet` 瞬间读进内存后立刻关句柄，长时间的计算循环不持有文件，所以 fd 里没 parquet ≠ 结束。
- **抓完整报错的正确做法**（避免只看到 `Traceback` 第一行）：用 `timeout` + 重定向到文件，防止卡死且绝不丢 stderr：
  ```bash
  timeout 300 /opt/conda/envs/eopd/bin/python3 xxx.py ... > /tmp/run.txt 2>&1; echo "EXIT=$?"; cat /tmp/run.txt
  ```
- 看到 `Traceback` 但无正文 / 进程像卡住：多半是 import 卡住或输出被 `tee` 丢了 stderr，按上面方式落盘再读。

---

## 7. 与论文 Table 2 对照的注意点

- **EOPD vs OPD 的 Pass@8 差值**才是论文 Table 2 的核心结论（论文 ~+2.4 @1.7B）。两边用同一份数据，差值可比。
- **当前分数偏低主要是训练不足**（VM EOPD step 330 / apex step 174，远少于论文的 3 epoch），不是评测 bug。评测协议与论文一致（temp=1.0, top_p=0.8, max 8192, k=8, zero-shot, "Please reason step by step, and put your final answer within \boxed{}."）。
- 对照时看 `actor_huggingface_scores.txt` 末尾 6 行 + `MEAN (simple)`，以及 json 里的 `mean_avg_at_k` / `mean_pass_at_k`。

---

## 8. 关键文件速查

- `examples/on_policy_distillation/generate_offline_vllm.py` — 离线 vLLM 生成 + 分片合并
- `examples/on_policy_distillation/eval_six_benchmarks.sh` — 6 基准评测主流程（多卡数据并行）
- `examples/on_policy_distillation/score_avg_pass_at_k.py` — Avg@8 / Pass@8 评分（带进度条 + 健壮 + 静默 WARNING）
- `指令_vm_全流程.txt` — VM 完整流程（EOPD 训练→转换→评测 + OPD 训练→转换→评测 → Table 2）
- `指令_vm_重跑评测.txt` — 停单卡→拉新→8 卡重评测
- `指令_vm_修评分.txt` — 只重评分（不重生成）
- `指令_vm_诊断合并.txt` / `指令_vm_重合并并评分.txt` — 合并失败诊断 / 重合并+评分
