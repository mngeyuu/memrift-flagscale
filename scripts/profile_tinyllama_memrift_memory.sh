#!/bin/bash
# ============================================
# TinyLlama MemRift 训练 · 内存 Profiling
# ============================================
# 在跑 TinyLlama MemRift 训练时开启显存统计与（可选）PyTorch Profiler，
# 用于查看 allocated/peak、TensorBoard 曲线和 trace。
#
# 用法（在仓库根目录 FlagScale 下执行）:
#   ./scripts/profile_tinyllama_memrift_memory.sh
#
# 可选环境变量: 同 run_tinyllama_memrift_train.sh
#   TINYLLAMA_MODEL, MEMRIFT_WEIGHT_DIR, TRAIN_ITERS, SKIP_PREPARE
# 额外:
#   ENABLE_PYTORCH_PROFILER - 设为 1 时同时开 PyTorch Profiler（trace 写入 tensorboard_dir）
#   PROFILE_STEP_START / PROFILE_STEP_END - Profiler 起止 step，默认 5 与 10
#
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$ROOT_DIR/memrift_weights/tinyllama_1b_level18}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
ENABLE_PYTORCH_PROFILER="${ENABLE_PYTORCH_PROFILER:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-10}"

CONFIG_PATH="examples/memrift/conf"
EXP_DIR="${EXP_DIR:-$ROOT_DIR/outputs/memrift_example}"
# 内存快照写入路径（训练进程 cwd 为仓库根目录）
MEMORY_SNAPSHOT_PATH="${MEMORY_SNAPSHOT_PATH:-memrift_memory_snapshot.pickle}"

echo "============================================"
echo "TinyLlama MemRift 训练 · 内存 Profiling"
echo "============================================"
echo "ROOT_DIR:            $ROOT_DIR"
echo "MEMRIFT_WEIGHT_DIR:  $MEMRIFT_WEIGHT_DIR"
echo "TRAIN_ITERS:         $TRAIN_ITERS"
echo "log_memory_to_tensorboard: true"
echo "record_memory_history:      true"
echo "memory_snapshot_path:       $MEMORY_SNAPSHOT_PATH"
echo "ENABLE_PYTORCH_PROFILER:    $ENABLE_PYTORCH_PROFILER"
echo "============================================"

# 若未跳过且无压缩权重，先准备权重（与 run_tinyllama_memrift_train.sh 一致）
if [ "$SKIP_PREPARE" != "1" ] && [ ! -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  echo ">> 正在生成 MemRift 压缩权重..."
  python -m flagscale.compress.memrift.offline_comp.prepare_weight \
    --model "$TINYLLAMA_MODEL" \
    --outdir "$MEMRIFT_WEIGHT_DIR" \
    --level 18
fi

# 构建 override：开启内存相关与可选 profiler（+ 表示添加 config 中未定义的键）
# 显式开启激活压缩，测试 TinyLlama 1.1B 训练显存峰值
# 训练在远端后台跑（nohup），脚本启动后即返回；看结果用 tail -f .../host_0_localhost.output
OVERRIDES=(
  "train.system.memrift_compressed_weight_dir=$MEMRIFT_WEIGHT_DIR"
  "train.model.tokenizer_path=$TINYLLAMA_MODEL"
  "train.model.tokenizer_model=$TINYLLAMA_MODEL"
  "train.trainer.train_iters=$TRAIN_ITERS"
  "train.system.memrift_activation_enable=true"
  "+train.system.log_memory_to_tensorboard=true"
  "+train.system.record_memory_history=true"
  "+train.system.memory_snapshot_path=$MEMORY_SNAPSHOT_PATH"
)

if [ "$ENABLE_PYTORCH_PROFILER" = "1" ]; then
  OVERRIDES+=(
    "+train.system.profile=true"
    "+train.system.use_pytorch_profiler=true"
    "+train.system.profile_step_start=$PROFILE_STEP_START"
    "+train.system.profile_step_end=$PROFILE_STEP_END"
  )
fi

echo ">> 启动训练（内存 profiling 已开启，后台运行；看日志: tail -f $EXP_DIR/logs/host_0_localhost.output）..."
python run.py \
  --config-path="$CONFIG_PATH" \
  --config-name=train_mock \
  action=run \
  "${OVERRIDES[@]}"

echo ""
echo ">> 查看内存结果:"
echo "  1) 日志中的显存: 见 $EXP_DIR/logs/host_0_localhost.output 中 report_memory 的 max allocated / max reserved (MB)"
echo "  2) TensorBoard 曲线: tensorboard --logdir=$EXP_DIR/tensorboard   (mem-allocated-bytes, mem-max-allocated-bytes 等)"
echo "  3) 内存快照: $ROOT_DIR/$MEMORY_SNAPSHOT_PATH (可用 PyTorch 官方工具可视化)"
if [ "$ENABLE_PYTORCH_PROFILER" = "1" ]; then
  echo "  4) PyTorch Profiler trace: 同上 tensorboard，在 PyTorch Profiler 面板查看"
fi
