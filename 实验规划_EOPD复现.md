# EOPD 复现实验规划

> 目标：在六个数学推理 benchmark 上复现 EOPD 的表现，使用 **Qwen3-1.7B-Base** 学生模型 + **Qwen3-8B** 教师模型。
> 复现范围：**OPD baseline + EOPD** 对照（对应论文 Table 2 的 Qwen3-1.7B-Base / MATH 训练数据这一行）。
> teacher top-k：**k=16**（论文 Appendix F 设定）。

---

## 一、从论文（eopd.pdf）提取的关键信息

### 1.1 模型设定（论文 §5.1）
- 学生模型：Qwen3-1.7B-Base（论文还包含 0.6B 和 4B，本次只做 1.7B）
- 教师模型：**Qwen3-8B**，**关闭 thinking mode**（论文明确说明）
- 1.7B 学生使用的训练数据：**MATH** 数据集
- 教师对学生分布的高熵 token 占比约 18.5%

### 1.2 六个评测 benchmark 与指标
- 数据集：MATH500、AMC23、Minerva、OlympiadBench、AIME24、AIME25
- 采样参数：temperature = **1.0**，top-p = **0.8**，max response length = **8192** tokens
- 每个问题采样 **8 条**回答，报告两个指标：
  - **Avg@8**：每个问题 8 条回答中正确的平均比例，再跨问题取均值（反映平均推理质量）
  - **Pass@8**：8 条中至少 1 条正确的问题占比（反映最佳情况解题能力）

### 1.3 EOPD 超参数（论文 §5.1、Appendix F）
- τ（熵阈值）= **0.8**：仅在教师分布高熵的 token 上引入 forward KL
- α（soft-KD 系数）= **1.0**
- top-k = **16**：forward KL 只在教师 top-16 token 上计算（兼顾累积概率质量与显存）
- 训练 batch B = **128**，mini-batch B_mini = **32**，每个训练迭代 4 步梯度更新

### 1.4 Table 2 对照目标（Qwen3-1.7B-Base，训练数据 MATH）

| Benchmark | 指标 | OPD (baseline) | EOPD (目标) |
|---|---|---|---|
| MATH500 | Avg@8 / Pass@8 | 67.76 / 84.80 | 68.73 / 87.60 |
| AMC23 | Avg@8 / Pass@8 | 39.06 / 70.00 | 41.88 / 75.00 |
| Minerva | Avg@8 / Pass@8 | 29.83 / 47.06 | 30.15 / 50.74 |
| OlympiadBench | Avg@8 / Pass@8 | 30.09 / 51.56 | 30.28 / 51.11 |
| AIME24 | Avg@8 / Pass@8 | 8.33 / 20.00 | 10.42 / 23.33 |
| AIME25 | Avg@8 / Pass@8 | 6.25 / 16.67 | 5.83 / 16.67 |

预期：EOPD 在 6 个集中 5 个优于 OPD（OlympiadBench 基本持平，AIME25 在该规模下噪声较大）。

---

## 二、已克隆仓库（WLS04/EOPD）的现状

仓库实际是 **verl 的一个 fork**，EOPD 的实现已经包含在内：

- **损失函数**：`verl/trainer/ppo/core_algos.py` 第 1025–1348 行，`compute_policy_loss_on_policy_distill`
  - 总损失 = **clipped reverse KL**（这就是 OPD）+ **entropy-gated forward KL**（额外项即 EOPD 的核心）
  - 通过 `soft_kd_entropy_threshold`（=τ=0.8）对高熵 token 加 mask，只有高熵处才加 forward KL
  - 配置键与论文一一对应：`soft_kd_entropy_threshold`→τ，`soft_kd_loss_coef`→α，`ref.topk_logits`→k
- **训练器**：`verl/trainer/ppo/on_policy_distill_trainer.py` 中的 `OnPolicyDistillTrainer`，把教师模型作为 ref/teacher worker 加载
- **README 设定**：`examples/on_policy_distillation/README.md` 给出了关键配置，但其中引用的启动脚本 `on_policy_it.sh` **在仓库里并不存在** → 需要自行新建
- **数据预处理**：`examples/data_preprocess/` 下已有
  - `math500_test.py`、`aime24_test.py`、`aime25_test.py`、`math_dataset.py`、`gsm8k.py`
  - ⚠️ **AMC23 / Minerva / OlympiadBench 三个 benchmark 没有预处理脚本**，需自行补充（完整复现 Table 2 的前置条件）

### 已实现 vs 待办
| 项 | 状态 |
|---|---|
| EOPD / OPD 损失实现 | ✅ 已在 core_algos.py |
| OnPolicyDistillTrainer | ✅ 已有 |
| 启动脚本 on_policy_it.sh | ❌ 缺失，需新建 |
| MATH 训练数据预处理 | ⚠️ math_dataset.py 存在，需确认参数 |
| MATH500 / AIME24 / AIME25 评测预处理 | ✅ 已有 |
| AMC23 / Minerva / OlympiadBench 评测预处理 | ❌ 缺失，需新建 |

---

## 三、详细执行步骤

### Phase 0 — 环境准备
```bash
cd /mnt/d/Desktop/eopd-baseline
pip install -e .                       # 安装 verl 本体
# 按后端选择其一
pip install -r requirements-cuda.txt   # 或 requirements-sglang.txt / requirements-npu.txt
```
验证：`python -c "import verl; print(verl.__version__)"`。
本仓库 EOPD 依赖 Ray + FSDP/SGLang（或 vLLM）rollout 后端。

### Phase 1 — 数据准备
**训练数据（MATH）**
```bash
python examples/data_preprocess/math_dataset.py   # 先读其 argparse 再补 --local_dir data/math
```

**评测数据（6 个 benchmark）**
- 已有脚本：`math500_test.py`、`aime24_test.py`、`aime25_test.py`（对应 MATH500 / AIME24 / AIME25）
- 需新建：`amc23`、`minerva`、`olympiadbench` 三个预处理脚本
  - 下载对应原始数据集，转换为与现有脚本一致的 `{"prompt", "reward_model": {"ground_truth"}}` schema
  - 可参考 `math500_test.py` 的写法复制修改

### Phase 2 — 模型下载
```bash
# 学生
hf download Qwen/Qwen3-1.7B-Base  --local-dir /models/Qwen3-1.7B-Base
# 教师（务必关闭 thinking mode，论文 §5.1）
hf download Qwen/Qwen3-8B          --local-dir /models/Qwen3-8B
```
注意：论文模型为 **Qwen3** 系列；你提到的“Qwen-1.7B”请确认即 Qwen3-1.7B-Base（与要复现的 Table 2 行一致）。

### Phase 3 — 训练配置（两个变体）
仓库缺失启动脚本，需为每个变体写一份启动命令/配置。基础设定来自 `examples/on_policy_distillation/README.md` + 论文 §5.1：

| 配置键 | 取值 | 含义 |
|---|---|---|
| `algorithm.adv_estimator` | `on_policy` | on-policy 优势估计 |
| `actor_rollout_ref.teacher_model.path` | `/models/Qwen3-8B` | 教师模型 |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `on_policy_distill` | EOPD/OPD 损失 |
| `actor_rollout_ref.actor.policy_loss.soft_kd_student_full_vocab` | `True` | 全词表蒸馏 |
| `actor_rollout_ref.ref.topk_logits` | `16` | **k=16（论文）** |
| `trainer.trainer_class` | `OnPolicyDistillTrainer` | 训练器 |
| batch / mini-batch | `128` / `32` | B、B_mini（每迭代 4 步梯度） |
| rollout temp / top-p / max-len | `1.0` / `0.8` / `8192` | 生成参数 |

**EOPD（目标变体）** 额外加：
```
actor_rollout_ref.actor.policy_loss.soft_kd_entropy_threshold=0.8   # τ
actor_rollout_ref.actor.policy_loss.soft_kd_loss_coef=1.0           # α
```

**OPD baseline（对照变体）** 同上加：
```
actor_rollout_ref.actor.policy_loss.soft_kd_entropy_threshold=100   # mask 恒为空 → 纯 OPD
```
原理：见 `core_algos.py:1210-1213`，当 `teacher_entropy >= 100` 恒为 False 时，`soft_kd_mask` 全零，`soft_kd_loss`→0，仅剩 clipped reverse KL，即 OPD。

启动方式（参考 `recipe/*/run_*.sh` 的 GPU/并行块）：
```bash
python -m verl.trainer.main_ppo <base_config>.yaml \
    trainer.trainer_class=OnPolicyDistillTrainer \
    algorithm.adv_estimator=on_policy \
    actor_rollout_ref.teacher_model.path=/models/Qwen3-8B \
    actor_rollout_ref.actor.policy_loss.loss_mode=on_policy_distill \
    actor_rollout_ref.actor.policy_loss.soft_kd_student_full_vocab=True \
    actor_rollout_ref.ref.topk_logits=16 \
    actor_rollout_ref.actor.policy_loss.soft_kd_entropy_threshold=0.8 \
    actor_rollout_ref.actor.policy_loss.soft_kd_loss_coef=1.0
```

### Phase 4 — 训练
- EOPD 与 OPD 作为两次独立任务运行
- 关注 `actor/soft_kd_*` 指标（`core_algos.py:1301`）：
  - EOPD 应观察到非零的 `soft_kd_token_ratio`（高熵 token 被蒸馏）
  - OPD 该比例恒为 0

### Phase 5 — 评测（Avg@8 / Pass@8）
对每个 benchmark，每条 prompt 生成 **8 条**回答（temperature=1.0, top_p=0.8, max_tokens=8192），然后：
- **Avg@8** = 各 prompt 的（8 条中正确比例）的均值
- **Pass@8** = 至少 1 条正确的 prompt 占比

答案抽取/评分使用 `verl/utils/reward_score/math_verify.py`（与论文可验证奖励设定一致）。
可参考 `tests/special_e2e/run_r1_distill_qwen_aime24_eval.sh` 改写成循环 k=8 并计算两个指标的脚本。

### Phase 6 — 与 Table 2 核对
见上方 1.4 对照表。若 EOPD ≈ OPD 处处持平，说明 forward-KL 项未生效，检查：
1. `soft_kd_entropy_threshold` 是否被覆盖
2. `teacher_topk_log_probs` 是否真正流入损失（`core_algos.py:1174`）

---

## 四、风险与阻塞项

1. **启动脚本缺失**：README 引用的 `on_policy_it.sh` 不在仓库中，需自行编写启动命令/配置。
2. **三个评测预处理缺失**：AMC23 / Minerva / OlympiadBench 无预处理脚本，必须新建才能完整复现 Table 2。
3. **Qwen 与 Qwen3 的命名**：确认使用的是 Qwen3-1.7B-Base（论文对应行）。
4. **top-k 取值**：本次选定 k=16（论文）。若结果与 Table 2 偏差较大，可回退尝试 README 示例的 k=32。
5. **教师 thinking mode**：必须关闭，否则分布与论文不符。

---

## 五、建议的下一步
1. 先确认环境与后端（CUDA / SGLang / vLLM）及模型路径；
2. 补齐 AMC23 / Minerva / OlympiadBench 三个评测预处理脚本；
3. 编写缺失的启动脚本（分别支持 OPD 与 EOPD 两个变体）；
4. 用小规模（少量 step / 少量数据）冒烟测试，确认 `soft_kd_*` 指标符合预期后再跑全量。
