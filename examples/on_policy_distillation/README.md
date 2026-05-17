## On-Policy Distillation

After installing `verl`, run `examples/on_policy_distillation/on_policy_it.sh`.
Prepare the required datasets by preprocessing in `examples/data_preprocess`.

### On-Policy Distillation Settings (from `on_policy_it.sh`)

- `algorithm.adv_estimator=on_policy`: enables on-policy advantage estimation.
- `actor_rollout_ref.teacher_model.path=Qwen/Qwen3-8B`: teacher model used for distillation.
- `actor_rollout_ref.actor.policy_loss.loss_mode=on_policy_distill`: use on-policy distillation loss.
- `actor_rollout_ref.actor.policy_loss.soft_kd_student_full_vocab=True`: distill against full-vocab teacher distribution.
- `actor_rollout_ref.ref.topk_logits=32`: use top-k teacher logits for FKL.
- `trainer.trainer_class=OnPolicyDistillTrainer`: uses the on-policy distillation trainer.