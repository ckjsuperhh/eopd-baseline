# EOPD 代码导航

这份文档用于回答两个问题：一次 EOPD 训练从哪里启动、算法逻辑最终落在哪些文件。仓库以 `verl` 为训练框架，`examples/on_policy_distillation/` 保留方法示例，`scripts/eopd/` 则把示例整合为可迁移的复现流程。

## 先看哪里

| 目标 | 首先阅读 | 说明 |
| --- | --- | --- |
| 直接复现 EOPD | `scripts/eopd/run_eopd.sh` | 可配置模型、数据、GPU、checkpoint 和论文主要超参的包装入口。 |
| 理解最小方法配置 | `examples/on_policy_distillation/on_policy_it.sh` | 真正发起 `python -m verl.trainer.main_ppo` 的 EOPD/OPD 示例。 |
| 理解训练框架 | `verl/trainer/main_ppo.py` | Hydra 配置、Ray 初始化、worker 和 trainer 选择的总入口。 |
| 理解 OPD/EOPD 训练循环 | `verl/trainer/ppo/on_policy_distill_trainer.py` | `RayPPOTrainer` 的蒸馏特化；固定教师作为 ref policy。 |
| 理解损失公式 | `verl/trainer/ppo/core_algos.py` | `compute_policy_loss_on_policy_distill` 实现 reverse-KL、PPO clip 及熵门控 forward-KL。 |
| 追踪教师/学生 logits | `verl/workers/fsdp_workers.py`、`verl/workers/actor/dp_actor.py` | 教师熵与 top-k logits 的产生、传递，以及学生端损失调用。 |
| 准备数据和评测 | `scripts/eopd/data_preprocess/`、`scripts/eopd/run_eval.sh` | 统一数据布局，运行六个数学基准并汇总。 |

## 训练调用链

```text
scripts/eopd/run_eopd.sh
  -> examples/on_policy_distillation/on_policy_it.sh
     -> python -m verl.trainer.main_ppo
        -> verl/trainer/main_ppo.py::run_ppo
           -> Ray TaskRunner 创建 actor / rollout / teacher(ref) worker
           -> OnPolicyDistillTrainer（由 trainer.trainer_class 选择）
              -> 学生通过 vLLM rollout 生成 response
              -> 教师在同一 response 上计算 log-prob、entropy、top-k logits
              -> 学生重算 log-prob 和对应 top-k logits
              -> core_algos.py::compute_policy_loss_on_policy_distill
              -> actor 更新、保存 checkpoint、记录指标
```

`main_ppo.py` 是框架级入口，而不是 EOPD 专用脚本：它读取 Hydra 配置、启动 Ray，并根据 `trainer.trainer_class` 选择训练器。EOPD 的启动配置写入 `+trainer.trainer_class=OnPolicyDistillTrainer`，因此在 `main_ppo.py` 中实例化 `OnPolicyDistillTrainer`。

## EOPD 的核心数据流

1. **学生 rollout**：vLLM 使用当前学生权重，从 prompt 生成 response；训练对象始终是学生。
2. **教师打分**：`OnPolicyDistillTrainer` 强制启用 reference policy，并将 `teacher_model.path` 指定的教师作为该 worker；教师在学生生成的 token 上计算 `ref_log_prob`。EOPD 模式还要求教师给出每个 token 的 entropy 和 top-k logits。
3. **学生重算**：actor worker 对同一序列计算带梯度的学生 log-prob，并在教师 top-k token 上取学生 log-prob。
4. **损失与更新**：`compute_policy_loss_on_policy_distill` 用 `ref_log_prob - old_log_prob` 构造密集的教师优势，配合 PPO clip 得到 OPD 的 reverse-KL 项；若提供 top-k logits，则仅在教师熵不低于阈值 `tau` 的 response token 上加入 forward-KL（soft KD）项。

因此，OPD 与 EOPD 的关键区别不是 rollout 引擎，而是教师信号：

- **OPD**：教师 token log-prob 驱动的 clipped reverse-KL；不请求 `ref.topk_logits`。
- **EOPD**：在 OPD 基础上，请求教师 top-k logits，并以教师 entropy 门控 forward-KL；脚本默认 `topk=16`、`tau=0.8`、系数 `alpha=1.0`。

## 关键文件及职责

### 方法示例与配置

- `examples/on_policy_distillation/on_policy_it.sh`：论文风格的模型、训练长度、采样及 EOPD/OPD 分支配置；适合核对方法参数。
- `examples/on_policy_distillation/eval_six_benchmarks.sh`：批量生成并评测六个数学基准。
- `examples/on_policy_distillation/generate_offline_vllm.py`：评测阶段的 vLLM 离线生成器。
- `examples/on_policy_distillation/score_avg_pass_at_k.py`：计算各 benchmark 的 Avg@k / Pass@k 汇总。
- `verl/trainer/config/ppo_trainer.yaml`：`main_ppo` 的基础 Hydra 配置；训练前应先了解其中 actor、rollout、ref、trainer 的层级。

### `verl` 框架与 EOPD 实现

- `verl/trainer/main_ppo.py`：创建资源池、Ray worker 和训练器；`TaskRunner.run` 根据配置选择 `OnPolicyDistillTrainer`。
- `verl/trainer/ppo/on_policy_distill_trainer.py`：EOPD 专用训练器。它继承 `RayPPOTrainer`，确保教师 ref worker 存在并在训练循环中使用教师信号。
- `verl/trainer/ppo/ray_trainer.py`：通用分布式 PPO 训练循环，负责 rollout、batch 组织、教师/学生 log-prob 调度、日志及 checkpoint。
- `verl/trainer/ppo/core_algos.py`：`@register_policy_loss("on_policy_distill")` 注册 EOPD 的核心损失；这是修改公式、熵门控或 token 权重时最重要的文件。
- `verl/workers/fsdp_workers.py`：FSDP worker 的教师前向路径，将 `ref_entropy` 和教师 top-k 信息放入 batch。
- `verl/workers/actor/dp_actor.py`：学生前向、top-k 对齐、`ref_entropy` 检查，并调用已注册的 policy loss。
- `verl/workers/actor/megatron_actor.py`、`verl/workers/megatron_workers.py`：使用 Megatron 后端时对应的实现；本地标准复现默认走 FSDP2。
- `verl/rollout/` 与 `verl/workers/rollout/`：rollout 后端适配层；实验脚本通过 `actor_rollout_ref.rollout.name=vllm` 选择 vLLM。

## `scripts/` 补充了什么

`examples/on_policy_distillation` 是算法示例；`scripts` 不重复实现 EOPD，而是解决“在另一台机器怎样稳定复现”的工程问题。

| 路径 | 功能 |
| --- | --- |
| `scripts/eopd/run_eopd.sh` | EOPD 训练入口：激活环境、处理 FlashAttention fallback、统一数据和 checkpoint 路径，并转调 example。 |
| `scripts/eopd/run_opd.sh` | 复用 EOPD 启动器，仅切到 `METHOD=opd`，保证对照条件一致。 |
| `scripts/eopd/run_eval.sh` | 六基准评测包装器，统一 vLLM、GPU、离线缓存和模型路径约定。 |
| `scripts/eopd/data_preprocess/run_all_preprocess.sh` | 下载/转换 DAPO-Math-17k、六个测试集和 smoke 子集到统一 parquet 布局。 |
| `scripts/eopd/data_preprocess/*.py` | 每个训练集或测试集的独立下载和格式转换逻辑。 |
| `scripts/eopd/eopd_monitor.sh` | 轮询训练日志，记录进度及常见致命错误。 |
| `scripts/eopd/flash_attn_shim/` | 旧 glibc 或无法安装 FlashAttention 时的 `bert_padding` 兼容层；真实 FlashAttention 可用时不会启用。 |
| `scripts/install_vllm_sglang_mcore.sh` | 安装 vLLM、SGLang、Megatron 及相关 Python 依赖。 |
| `scripts/converter_hf_to_mcore.py`、`scripts/legacy_model_merger.py` | HF、FSDP、Megatron checkpoint 的转换/合并。 |
| `scripts/diagnose.py`、`scripts/rollout_viewer.py` | 分别用于环境诊断和交互式检查 rollout 结果。 |

## 推荐阅读顺序

1. 读本文件和 `examples/on_policy_distillation/README.md`，建立角色与术语。
2. 读 `examples/on_policy_distillation/on_policy_it.sh`，看清 EOPD 与 OPD 的配置差异。
3. 跟进 `verl/trainer/main_ppo.py`，再读 `on_policy_distill_trainer.py`。
4. 阅读 `core_algos.py` 的 `compute_policy_loss_on_policy_distill`，确认 reverse-KL、PPO clip 和 entropy-gated forward-KL 的实现。
5. 需要排查张量来源时，再看 `fsdp_workers.py` 与 `dp_actor.py`。
6. 真正运行实验时使用 `scripts/eopd/run_eopd.sh`、`run_eval.sh` 和数据预处理脚本，而不是手工复制命令。

## 修改方法时的定位建议

- **改 EOPD 损失或 token 选择规则**：从 `core_algos.py` 开始，同时检查 `dp_actor.py` 是否已把所需张量传入。
- **改教师信号（熵、top-k logits）**：检查 `fsdp_workers.py` 的 reference 前向和 `on_policy_distill_trainer.py` 的 batch 流转。
- **改 rollout 或并行资源**：从启动脚本的 `actor_rollout_ref.rollout.*`、`trainer.n_gpus_per_node` 开始，随后进入 `verl` 的 rollout worker。
- **改数据、评测或复现可移植性**：优先修改 `scripts/eopd/`；保持 `examples/` 用作贴近原始方法的参考实现。

相关操作说明见 `EOPD复现.md`；该文档描述“代码在哪里”，复现文档描述“怎样运行”。
