#!/bin/bash
# ============================================
# 显存对比：memrift_demo 纯 LoRA / MemRift+LoRA vs FlagScale 纯 LoRA / MemRift+LoRA
# 抓取基线、峰值、显存组成
# ============================================
# 用法: ./scripts/compare_memrift_lora_memory.sh
# 环境变量: TRAIN_ITERS(default 5), SKIP_PREPARE(1=跳过压缩权重), CUDA_VISIBLE_DEVICES
#           CONDA_ENV(如 myc-flagscale) 激活后跑 memrift_demo
set -e

FLAGSCALE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_ROOT="${MEMRIFT_DEMO_ROOT:-/share/project/mengyc/code/memrift_demo}"
TRAIN_ITERS="${TRAIN_ITERS:-5}"
SKIP_PREPARE="${SKIP_PREPARE:-1}"
TINYLLAMA_MODEL="${TINYLLAMA_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MEMRIFT_WEIGHT_DIR="${MEMRIFT_WEIGHT_DIR:-$FLAGSCALE_ROOT/memrift_weights/tinyllama_1b_level18}"
OUT_JSON="$FLAGSCALE_ROOT/scripts/compare_memrift_memory_result.json"

echo "============================================"
echo "显存对比: memrift_demo vs FlagScale"
echo "============================================"
echo "FLAGSCALE_ROOT: $FLAGSCALE_ROOT"
echo "DEMO_ROOT:      $DEMO_ROOT"
echo "TRAIN_ITERS:    $TRAIN_ITERS"
echo "OUT_JSON:       $OUT_JSON"
echo "============================================"

mkdir -p "$(dirname "$OUT_JSON")"
RESULTS="{}"

# jq 不可用时用 Python 合并 JSON
_jq_merge() {
  local key="$1"
  local val="$2"
  if command -v jq >/dev/null 2>&1; then
    echo "$RESULTS" | jq --argjson d "$val" --arg k "$key" '.[$k] = $d'
  else
    python3 -c "
import json
r=json.loads(r'''$RESULTS''')
r['$key']=json.loads(r'''$val''')
print(json.dumps(r,ensure_ascii=False))
" 2>/dev/null || echo "$RESULTS"
  fi
}

# --- 解析 memrift_demo profile 输出 ---
parse_demo_profile() {
  local log="$1"
  local baseline="" peak="" fwd_act="" grad_misc=""
  [ -f "$log" ] || return
  # 稳态基线 (权重+LoRA+优化器)         2207.3 MB → 取最后一处 “数字 MB”
  baseline=$(grep "稳态基线" "$log" 2>/dev/null | tail -1 | sed -n 's/.*稳态基线[^0-9]*\([0-9][0-9.]*\) *MB.*/\1/p')
  # 整轮 peak     3571.8 MB → 取 “peak” 后的数字
  peak=$(grep "整轮 peak" "$log" 2>/dev/null | tail -1 | sed -n 's/.*整轮 peak[^0-9]*\([0-9][0-9.]*\) *MB.*/\1/p')
  fwd_act=$(grep "前向激活增量" "$log" 2>/dev/null | tail -1 | sed -n 's/.*增量 \+\([0-9][0-9.]*\) MB.*/\1/p')
  grad_misc=$(grep "反向临时+梯度" "$log" 2>/dev/null | tail -1 | sed -n 's/.*反向临时+梯度[^0-9]*\([0-9][0-9.]*\) *MB.*/\1/p')
  echo "{\"baseline\":\"$baseline\",\"peak\":\"$peak\",\"fwd_act\":\"$fwd_act\",\"grad_misc\":\"$grad_misc\"}"
}

# --- 解析 FlagScale report_memory ---
parse_flagscale_memory() {
  local log="$1"
  local baseline="" peak=""
  [ -f "$log" ] || return
  baseline=$(grep "baseline, before first forward" "$log" 2>/dev/null | tail -1 | sed -n 's/.*| allocated: \([0-9][0-9.]*\) | max allocated.*/\1/p')
  peak=$(grep "after.*iterations" "$log" 2>/dev/null | tail -1 | sed -n 's/.*max allocated: \([0-9][0-9.]*\).*/\1/p')
  echo "{\"baseline\":\"$baseline\",\"peak\":\"$peak\"}"
}

# --- 1) memrift_demo 纯 LoRA ---
echo ""
echo ">> [1/4] memrift_demo 纯 LoRA (无 hook/weight)..."
DEMO_PURE_LOG="$DEMO_ROOT/.compare_pure_lora.log"
if [ -d "$DEMO_ROOT" ]; then
  cd "$DEMO_ROOT"
  if [ -n "$CONDA_ENV" ]; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV" || true
  fi
  pip install -q humanize 2>/dev/null || true
  WANDB_MODE=disabled python train_pool1.py \
    --model "$TINYLLAMA_MODEL" \
    --finetune_type lora \
    --profile_memory \
    --round 3 \
    --max_length 512 \
    2>&1 | tee "$DEMO_PURE_LOG" || true
  DEMO_PURE=$(parse_demo_profile "$DEMO_PURE_LOG")
  RESULTS=$(_jq_merge "memrift_demo_pure_lora" "$DEMO_PURE")
  cd - >/dev/null
else
  echo "  跳过: DEMO_ROOT 不存在 $DEMO_ROOT"
fi

# --- 2) memrift_demo MemRift+LoRA ---
echo ""
echo ">> [2/4] memrift_demo MemRift+LoRA (hook + weight + activation)..."
DEMO_MEMRIFT_LOG="$DEMO_ROOT/.compare_memrift_lora.log"
if [ -d "$DEMO_ROOT" ] && [ -f "$MEMRIFT_WEIGHT_DIR/index.json" ]; then
  cd "$DEMO_ROOT"
  if [ -n "$CONDA_ENV" ]; then
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null && conda activate "$CONDA_ENV" || true
  fi
  pip install -q humanize 2>/dev/null || true
  WANDB_MODE=disabled python train_pool1.py \
    --model "$TINYLLAMA_MODEL" \
    --outdir "$MEMRIFT_WEIGHT_DIR" \
    --finetune_type lora \
    --hook --weight --activation --weight_async --act_async \
    --profile_memory \
    --round 3 \
    --max_length 512 \
    2>&1 | tee "$DEMO_MEMRIFT_LOG" || true
  DEMO_MEMRIFT=$(parse_demo_profile "$DEMO_MEMRIFT_LOG")
  RESULTS=$(_jq_merge "memrift_demo_memrift_lora" "$DEMO_MEMRIFT")
  cd - >/dev/null
else
  echo "  跳过: 需要 DEMO_ROOT 与 $MEMRIFT_WEIGHT_DIR/index.json"
fi

# --- 3) FlagScale MemRift+LoRA ---
echo ""
echo ">> [3/4] FlagScale MemRift+LoRA..."
rm -rf "$FLAGSCALE_ROOT/outputs/memrift_example/checkpoints"/* 2>/dev/null || true
cd "$FLAGSCALE_ROOT"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
SKIP_PREPARE="$SKIP_PREPARE" TRAIN_ITERS="$TRAIN_ITERS" \
  python run.py \
  --config-path=examples/memrift/conf \
  --config-name=train_mock \
  action=run \
  train.system.memrift_compressed_weight_dir="$MEMRIFT_WEIGHT_DIR" \
  train.model.tokenizer_path="$TINYLLAMA_MODEL" \
  train.model.tokenizer_model="$TINYLLAMA_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  train.trainer.lr_warmup_iters=2 \
  train.system.logging.log_interval=1 \
  "+train.system.memrift_profile_memory=true" \
  train.system.memrift_print_debug=true \
  2>&1 | tee "$FLAGSCALE_ROOT/scripts/.compare_flagscale_memrift.log" || true

echo ">> 等待 FlagScale MemRift 完成..."
FS_MEMRIFT_LOG="$FLAGSCALE_ROOT/outputs/memrift_example/logs/host_0_localhost.output"
FS_MEMRIFT_BACKUP="$FLAGSCALE_ROOT/scripts/.compare_flagscale_memrift_host.output"
[ -f "$FS_MEMRIFT_LOG" ] && cp "$FS_MEMRIFT_LOG" "$FS_MEMRIFT_BACKUP"
if [ -f "$FS_MEMRIFT_BACKUP" ]; then
  FS_MEMRIFT=$(parse_flagscale_memory "$FS_MEMRIFT_BACKUP")
else
  FS_MEMRIFT=$(parse_flagscale_memory "$FLAGSCALE_ROOT/scripts/.compare_flagscale_memrift.log")
fi
# 解析 sm_gpu / LoRA / other (用 MemRift 备份避免被步骤4覆盖)
FS_LOG="${FS_MEMRIFT_BACKUP:-$FLAGSCALE_ROOT/scripts/.compare_flagscale_memrift.log}"
[ -f "$FS_LOG" ] || FS_LOG="$FLAGSCALE_ROOT/scripts/.compare_flagscale_memrift.log"
FS_SM=$(grep "sm_gpu (常驻" "$FS_LOG" 2>/dev/null | tail -1 | sed -n 's/.*\([0-9][0-9.]*\) MB.*/\1/p')
FS_LORA=$(grep "LoRA 参数" "$FS_LOG" 2>/dev/null | tail -1 | sed -n 's/.*\([0-9][0-9.]*\) MB.*/\1/p')
FS_OTHER=$(grep "其它 (优化器" "$FS_LOG" 2>/dev/null | tail -1 | sed -n 's/.*\([0-9][0-9.]*\) MB.*/\1/p')
RESULTS=$(_jq_merge "flagscale_memrift" "$FS_MEMRIFT")
# 补充 sm_gpu/lora/other（需再次合并）
if command -v jq >/dev/null 2>&1; then
  RESULTS=$(echo "$RESULTS" | jq '.flagscale_memrift.sm_gpu_mb = "'"${FS_SM:-}"'" | .flagscale_memrift.lora_mb = "'"${FS_LORA:-}"'" | .flagscale_memrift.other_mb = "'"${FS_OTHER:-}"'"' 2>/dev/null || echo "$RESULTS")
else
  RESULTS=$(python3 -c "
import json
r=json.loads('''$RESULTS''')
r.setdefault('flagscale_memrift',{})['sm_gpu_mb']='${FS_SM:-}'
r.setdefault('flagscale_memrift',{})['lora_mb']='${FS_LORA:-}'
r.setdefault('flagscale_memrift',{})['other_mb']='${FS_OTHER:-}'
print(json.dumps(r,ensure_ascii=False))
")
fi

# --- 4) FlagScale 纯 LoRA ---
echo ""
echo ">> [4/4] FlagScale 纯 LoRA (memrift_enable=false, init_model_with_meta=false)..."
rm -rf "$FLAGSCALE_ROOT/outputs/memrift_example/checkpoints"/* 2>/dev/null || true
cd "$FLAGSCALE_ROOT"
python run.py \
  --config-path=examples/memrift/conf \
  --config-name=train_mock \
  action=run \
  train.system.memrift_enable=false \
  "+train.system.init_model_with_meta_device=false" \
  train.model.tokenizer_path="$TINYLLAMA_MODEL" \
  train.model.tokenizer_model="$TINYLLAMA_MODEL" \
  train.trainer.train_iters="$TRAIN_ITERS" \
  train.trainer.lr_warmup_iters=2 \
  train.system.logging.log_interval=1 \
  2>&1 | tee "$FLAGSCALE_ROOT/scripts/.compare_flagscale_pure.log" || true

echo ">> 等待 FlagScale 纯 LoRA 完成..."
FS_PURE_LOG="$FLAGSCALE_ROOT/outputs/memrift_example/logs/host_0_localhost.output"
if [ -f "$FS_PURE_LOG" ]; then
  FS_PURE=$(parse_flagscale_memory "$FS_PURE_LOG")
else
  FS_PURE=$(parse_flagscale_memory "$FLAGSCALE_ROOT/scripts/.compare_flagscale_pure.log")
fi
RESULTS=$(_jq_merge "flagscale_pure_lora" "$FS_PURE")
cd - >/dev/null

# --- 汇总打印 ---
echo "$RESULTS" > "$OUT_JSON"
echo ""
echo "============================================"
echo "  显存对比汇总 (TinyLlama 1.1B, ${TRAIN_ITERS} iters)"
echo "============================================"
printf "  %-28s %12s %12s %12s\n" "配置" "基线(MB)" "峰值(MB)" "peak-baseline"
echo "  --------------------------------------------------------------------"
for k in memrift_demo_pure_lora memrift_demo_memrift_lora flagscale_pure_lora flagscale_memrift; do
  if command -v jq >/dev/null 2>&1; then
    b=$(echo "$RESULTS" | jq -r ".\"$k\".baseline // \"N/A\"")
    p=$(echo "$RESULTS" | jq -r ".\"$k\".peak // \"N/A\"")
  else
    b=$(echo "$RESULTS" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('$k',{}).get('baseline','N/A'))" 2>/dev/null || echo "N/A")
    p=$(echo "$RESULTS" | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('$k',{}).get('peak','N/A'))" 2>/dev/null || echo "N/A")
  fi
  d="N/A"
  if [ "$b" != "N/A" ] && [ "$p" != "N/A" ] && [ -n "$b" ] && [ -n "$p" ]; then
    d=$(awk "BEGIN { printf \"%.0f\", $p - $b }" 2>/dev/null || echo "N/A")
  fi
  n="$k"
  [ "$k" = "memrift_demo_pure_lora" ] && n="memrift_demo 纯 LoRA"
  [ "$k" = "memrift_demo_memrift_lora" ] && n="memrift_demo MemRift+LoRA"
  [ "$k" = "flagscale_pure_lora" ] && n="FlagScale 纯 LoRA"
  [ "$k" = "flagscale_memrift" ] && n="FlagScale MemRift+LoRA"
  printf "  %-28s %12s %12s %12s\n" "$n" "$b" "$p" "$d"
done
echo "  --------------------------------------------------------------------"
echo ""
echo "  详细 JSON: $OUT_JSON"
echo "  FlagScale MemRift 组成: sm_gpu=${FS_SM:-N/A} LoRA=${FS_LORA:-N/A} other=${FS_OTHER:-N/A} MB"
echo "============================================"
