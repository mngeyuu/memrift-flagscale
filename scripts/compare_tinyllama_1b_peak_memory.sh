#!/bin/bash
# ============================================
# TinyLlama 1.1B 显存峰值对比：Baseline LoRA vs MemRift+LoRA
# ============================================
# 依次运行 Baseline（FlagScale_baseline）与 MemRift（本仓库），
# 从日志中解析显存峰值并打印对比表。
#
# 用法（在 FlagScale 仓库根目录执行）:
#   ./scripts/compare_tinyllama_1b_peak_memory.sh
#
# 可选环境变量:
#   BASELINE_ROOT   - FlagScale_baseline 仓库路径，默认 ../FlagScale_baseline
#   TRAIN_ITERS     - 训练步数（两边一致），默认 5（快速对比）
#   SKIP_PREPARE    - 对 MemRift：1=跳过准备压缩权重（若已存在）
#   WAIT_SEC        - 每次训练后等待日志/进程的最长时间（秒），默认 600
#
set -e

FLAGSCALE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_ROOT="${BASELINE_ROOT:-$FLAGSCALE_ROOT/../FlagScale_baseline}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"
TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$FLAGSCALE_ROOT/memrift_weights/tinyllama_1b_level18}"

# 实际训练输出在 runner 的 exp_dir 下，解析时用
BASELINE_EXP_DIR="$BASELINE_ROOT/outputs/tinyllama_lora_baseline"
MEMRIFT_EXP_DIR="$FLAGSCALE_ROOT/outputs/memrift_example"
LOG_BASELINE="$BASELINE_EXP_DIR/logs/host_0_localhost.output"
LOG_MEMRIFT="$MEMRIFT_EXP_DIR/logs/host_0_localhost.output"

echo "============================================"
echo "TinyLlama 1.1B 显存峰值对比"
echo "============================================"
echo "FLAGSCALE_ROOT:  $FLAGSCALE_ROOT"
echo "BASELINE_ROOT:   $BASELINE_ROOT"
echo "TRAIN_ITERS:     $TRAIN_ITERS"
echo "============================================"

# 等待训练结束：轮询 pid 或日志中的峰值行，最多等 WAIT_SEC 秒
WAIT_SEC="${WAIT_SEC:-600}"
wait_for_training_log() {
  local log="$1"
  local pid_file="$2"
  local sec=0
  while [ $sec -lt "$WAIT_SEC" ]; do
    if [ -f "$log" ] && grep -qE "整轮 peak \(max_memory_allocated\)|max allocated:" "$log" 2>/dev/null; then
      return 0
    fi
    if [ -f "$pid_file" ]; then
      pid=$(cat "$pid_file" 2>/dev/null)
      if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        return 0
      fi
    fi
    sleep 10
    sec=$((sec + 10))
  done
  return 1
}

# 解析峰值：优先整轮 peak 行，否则取 report_memory 的 max allocated
parse_peak_from_log() {
  local log="$1"
  local peak=""
  if [ -f "$log" ]; then
    # 整轮 peak: "整轮 peak (max_memory_allocated)             6980.4 MB"（取最后一行的数字）
    peak=$(grep "整轮 peak (max_memory_allocated)" "$log" 2>/dev/null | tail -1 | sed -n 's/.*peak (max_memory_allocated)[^0-9]*\([0-9][0-9.]*\) MB.*/\1/p')
    if [ -z "$peak" ]; then
      # report_memory: "... | max allocated: XXXX.XXX | ..."
      peak=$(grep "max allocated:" "$log" 2>/dev/null | tail -1 | sed -n 's/.*max allocated: \([0-9][0-9.]*\).*/\1/p')
    fi
  fi
  echo "$peak"
}

# ---------- 1) 运行 Baseline LoRA ----------
if [ ! -d "$BASELINE_ROOT" ]; then
  echo "错误: BASELINE_ROOT 不存在: $BASELINE_ROOT"
  echo "请设置: export BASELINE_ROOT=/path/to/FlagScale_baseline"
  exit 1
fi

echo ""
echo ">> [1/2] 运行 Baseline LoRA (${TRAIN_ITERS} iters)..."
cd "$BASELINE_ROOT"
TRAIN_ITERS="$TRAIN_ITERS" TINYLLAMA_MODEL="$TINYLLAMA_MODEL" \
  python run.py \
  --config-path=examples/tinyllama_lora/conf \
  --config-name=train \
  action=run \
  train.model.tokenizer_path="$TINYLLAMA_MODEL" \
  train.model.tokenizer_model="$TINYLLAMA_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  "+train.system.baseline_profile_memory=true" \
  "+train.system.log_memory_to_tensorboard=true" \
  "+train.system.record_memory_history=true" \
  "+train.system.memory_snapshot_path=baseline_memory_snapshot.pickle" \
  2>&1 | tee "$FLAGSCALE_ROOT/scripts/.compare_baseline_stdout.log" || true
cd - >/dev/null
LOG_BASELINE="$BASELINE_EXP_DIR/logs/host_0_localhost.output"
echo ">> 等待 Baseline 训练完成（最多 ${WAIT_SEC}s）..."
wait_for_training_log "$LOG_BASELINE" "$BASELINE_EXP_DIR/logs/pids/host_0_localhost.pid" || true

# ---------- 2) 运行 MemRift+LoRA ----------
echo ""
echo ">> [2/2] 运行 MemRift+LoRA (${TRAIN_ITERS} iters)..."
# 清空旧 checkpoint，避免 use_distributed_optimizer 改为 false 后加载旧 optimizer 报错，且从 iter 0 跑满
rm -rf "$MEMRIFT_EXP_DIR/checkpoints" 2>/dev/null || true
if [ "$SKIP_PREPARE" != "1" ] || [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  if [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
    echo ">> 准备 MemRift 压缩权重..."
    cd "$FLAGSCALE_ROOT"
    python -m flagscale.compress.memrift.offline_comp.prepare_weight \
      --model "$TINYLLAMA_MODEL" \
      --outdir "$MEMRIFT_WEIGHT_DIR" \
      --level 18
  fi
fi

cd "$FLAGSCALE_ROOT"
TRAIN_ITERS="$TRAIN_ITERS" TINYLLAMA_MODEL="$TINYLLAMA_MODEL" \
  SKIP_PREPARE="$SKIP_PREPARE" MEMRIFT_WEIGHT_DIR="$MEMRIFT_WEIGHT_DIR" \
  python run.py \
  --config-path=examples/memrift/conf \
  --config-name=train_mock \
  action=run \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
  train.model.tokenizer_path="$TINYLLAMA_MODEL" \
  train.model.tokenizer_model="$TINYLLAMA_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  train.trainer.lr_warmup_iters=2 \
  train.system.use_distributed_optimizer=true \
  train.system.memrift_activation_enable=true \
  "+train.system.log_memory_to_tensorboard=true" \
  "+train.system.record_memory_history=true" \
  "+train.system.memory_snapshot_path=memrift_memory_snapshot.pickle" \
  2>&1 | tee "$FLAGSCALE_ROOT/scripts/.compare_memrift_stdout.log" || true
cd - >/dev/null
LOG_MEMRIFT="$MEMRIFT_EXP_DIR/logs/host_0_localhost.output"
echo ">> 等待 MemRift 训练完成（最多 ${WAIT_SEC}s）..."
wait_for_training_log "$LOG_MEMRIFT" "$MEMRIFT_EXP_DIR/logs/pids/host_0_localhost.pid" || true

# ---------- 3) 解析并打印对比 ----------
PEAK_BASELINE=$(parse_peak_from_log "$LOG_BASELINE")
PEAK_MEMRIFT=$(parse_peak_from_log "$LOG_MEMRIFT")

echo ""
echo "============================================"
echo "  显存峰值对比 (TinyLlama 1.1B, ${TRAIN_ITERS} iters)"
echo "============================================"
printf "  %-28s %12s\n" "配置" "峰值 (MB)"
echo "  ----------------------------------------"
printf "  %-28s %12s\n" "Baseline LoRA (整网权重常驻)" "${PEAK_BASELINE:-N/A}"
printf "  %-28s %12s\n" "MemRift+LoRA (压缩+按需加载)" "${PEAK_MEMRIFT:-N/A}"
echo "  ----------------------------------------"

if [ -n "$PEAK_BASELINE" ] && [ -n "$PEAK_MEMRIFT" ]; then
  DIFF=$(awk "BEGIN { printf \"%.1f\", $PEAK_MEMRIFT - $PEAK_BASELINE }" 2>/dev/null || echo "N/A")
  if [ "$DIFF" != "N/A" ]; then
    printf "  %-28s %+12s\n" "差值 (MemRift - Baseline)" "${DIFF}"
  fi
fi

echo ""
echo "  详细日志:"
echo "    Baseline: $LOG_BASELINE"
echo "    MemRift:  $LOG_MEMRIFT"
echo "============================================"
