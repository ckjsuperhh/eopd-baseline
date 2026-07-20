#!/usr/bin/env bash
# 训练错误监控器：每 60 秒扫描训练日志，记录进度并告警致命错误。
# 在独立 tmux 会话中常驻运行即可（例如：tmux new -d -s mon 'bash scripts/eopd/eopd_monitor.sh'）。
#
# 可调环境变量：
#   LOG   训练日志路径（默认 $HOME/eopd_train.log）
#   MON   监控日志路径（默认 $HOME/eopd_monitor.log）
#   INTERVAL  轮询间隔秒（默认 60）
#
# 监控的致命错误关键字：
#   OutOfMemory / CUDA out of memory / Traceback / Error executing /
#   HFValidationError / Expandable / RayTaskError
set -e

LOG="${LOG:-$HOME/eopd_train.log}"
MON="${MON:-$HOME/eopd_monitor.log}"
INTERVAL="${INTERVAL:-60}"

echo "monitor started $(date) (watching $LOG)" > "$MON"
while true; do
  ts=$(date +%H:%M:%S)
  err=$(grep -E "OutOfMemory|Traceback|CUDA out of memory|Error executing|HFValidationError|Expandable|RayTaskError" "$LOG" 2>/dev/null | tail -1)
  step=$(grep -oE "Training Progress:[ ]*[0-9]+%" "$LOG" 2>/dev/null | tail -1)
  stepm=$(grep -oE "step:[0-9]+ -" "$LOG" 2>/dev/null | tail -1)
  if [ -n "$err" ]; then
    echo "[$ts] ERROR: $err" >> "$MON"
  fi
  echo "[$ts] ${step:-?} ${stepm:-no-step-yet}" >> "$MON"
  sleep "$INTERVAL"
done
