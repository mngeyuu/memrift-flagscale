# MemRift Metric Test Suite

This directory contains a reproducible metric-test suite for the assessment items in
`zhibiao.md`.

The consolidated Chinese terminal-output and acceptance guide is available at
[`docs/MemRift_四项指标终端验收输出.md`](../../docs/MemRift_四项指标终端验收输出.md).

The suite is organized by model:

- `llama8b/`: industry-mainstream model, defaulting to LLaMA-3.1-8B.
- `aquila/`: Zhiyuan/self-developed model, defaulting to Aquila2-7B.
- `qwen3_8b/`: Qwen3-8B acceptance suite.

Each model directory has the same four scripts:

- `accuracy_loss.sh`: compares the final training `lm loss` of Pure LoRA and
  MemRift+LoRA; relative loss degradation must be at most 1%.
- `compression_ratio.sh`: checks compressed weight size against BF16 reference size.
- `train_context_gain.sh`: checks training context length gain against pure LoRA.
- `load_time_reduction.sh`: checks inference weight disk-read time reduction.

All training scripts are single-GPU and use MemRift weight + activation async mode:

- `tensor_model_parallel_size=1`
- `pipeline_model_parallel_size=1`
- `context_parallel_size=1`
- `memrift_weight_enable=true`
- `memrift_activation_enable=true`
- `memrift_weight_async=true`
- `memrift_act_async=true`

## Data

Training uses Alpaca by default. The expected Megatron indexed dataset prefix is:

```bash
/share/project/mengyc/data/alpaca_megatron/alpaca_text_document
```

The files must exist:

```bash
/share/project/mengyc/data/alpaca_megatron/alpaca_text_document.bin
/share/project/mengyc/data/alpaca_megatron/alpaca_text_document.idx
```

Override with:

```bash
ALPACA_DATA_PATH=/path/to/alpaca_text_document bash scripts/metrics/llama8b/accuracy_loss.sh
```

## Model Defaults

LLaMA 8B defaults:

```bash
MODEL_PATH=/share/project/mengyc/models/Llama-3.1-8B
MEMRIFT_WEIGHT_DIR=/share/project/mengyc/code/memrift-flagscale/memrift_weights/llama31_8b_level18
CONFIG_NAME=train_llama31_8b_mock
```

Aquila defaults:

```bash
MODEL_PATH=/share/project/mengyc/models/Aquila2-7B
MEMRIFT_WEIGHT_DIR=/share/project/mengyc/code/memrift-flagscale/memrift_weights/aquila2_7b_level18
CONFIG_NAME=train_aquila2_7b_mock
```

Override any of these as environment variables.

If `MEMRIFT_WEIGHT_DIR/index.json` is missing, the LLaMA 8B metric scripts prepare
compressed weights automatically into:

```bash
/share/project/mengyc/code/memrift-flagscale/memrift_weights/llama31_8b_level18
```

The command used by the scripts is:

```bash
python3 -m flagscale.compress.memrift.offline_comp.prepare_weight \
  --model "$MODEL_PATH" \
  --outdir "$MEMRIFT_WEIGHT_DIR" \
  --level "${MEMRIFT_PREPARE_LEVEL:-18}"
```

Set `PREPARE_MEMRIFT_WEIGHTS=0` to fail instead of preparing weights. The default
LLaMA 8B model path is:

```bash
/share/project/mengyc/models/Llama-3.1-8B
```

## Run

Run LLaMA 8B metrics:

```bash
cd /share/project/mengyc/code/memrift-flagscale
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/run_all_metrics.sh
```

For the context-length metric, the default search cap is `8192`. If both pure LoRA
and MemRift reach that cap, the result is cap-limited and does not prove the real
maximum context boundary. Raise the cap to test the `>= 20%` gain criterion:

```bash
CUDA_VISIBLE_DEVICES=0 CONTEXT_SEARCH_CAP=12288 \
  bash scripts/metrics/llama8b/train_context_gain.sh
```

Or run them individually:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/accuracy_loss.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/compression_ratio.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/train_context_gain.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/load_time_reduction.sh
```

Every individual metric script prints its own final one-item table with the
measured value, criterion, and `METRIC RESULT: PASS/FAIL`. If the test exits
before producing a fresh `result.json`, the final table reports `MISSING`
instead of silently displaying an older result.

The final block also prints the baseline value, MemRift value, four-decimal
benefit calculation, and the explicit threshold comparison, for example:

```text
(26.7783s - 17.8519s) / 26.7783s × 100% = 33.3346%;
33.3346% >= 30.00% => 符合指标 (PASS)
```

The terminal header follows the acceptance-table terminology: software name,
model category (`业界主流模型` or `智源自研模型`), test scenario, measurement
scope, measured values, acceptance relation, and final decision. The relations
shown are storage saving `>= 30%`, relative final `lm loss` degradation `<= 1%`,
`L1 >= 1.2 * L0` for context length, and `T1 <= 0.7 * T0` for load time.

The current context script measures the training-side maximum runnable length;
it does not yet test the inference-side maximum context. The current load-time
script measures weight-file disk-read time and excludes model instantiation and
first forward. Both scope limitations are printed in the terminal result so the
display does not overstate what was measured.

Run Aquila metrics:

```bash
cd /share/project/mengyc/code/memrift-flagscale
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/run_all_metrics.sh
```

Or run them individually:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/accuracy_loss.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/compression_ratio.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/train_context_gain.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/load_time_reduction.sh
```

Run Qwen3-8B metrics:

```bash
cd /share/project/mengyc/code/memrift-flagscale
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/qwen3_8b/run_all_metrics.sh
```

## Fast Smoke Runs

Use shorter iterations/context lengths before full runs:

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_ITERS=1 BASE_SEQ_LEN=1024 TARGET_SEQ_LEN=1229 \
  bash scripts/metrics/llama8b/train_context_gain.sh
```

Use `ACTION=dryrun` to validate Hydra/FlagScale command generation without launching training:

```bash
ACTION=dryrun TRAIN_ITERS=1 BASE_SEQ_LEN=128 TARGET_SEQ_LEN=160 \
  bash scripts/metrics/aquila/train_context_gain.sh
```

## Output

Each script writes a `result.json` under:

```bash
output/metrics/<model>/<metric>/result.json
```

Examples:

```bash
output/metrics/llama8b/accuracy_loss/result.json
output/metrics/aquila/compression_ratio/result.json
```

Each `run_all_metrics.sh` always ends with a compact acceptance table showing the
measured values, thresholds, per-item `PASS`/`FAIL`, the overall result, and the
path to `summary.json`. It continues to the remaining metrics after one metric
fails, then exits nonzero if any criterion failed or a fresh result is missing.
This makes the final terminal frame suitable for an acceptance recording.

To display a summary again from existing result files without rerunning GPU jobs:

```bash
python3 scripts/metrics/metrics_summary.py \
  --model-key qwen3_8b \
  --model-name Qwen3-8B \
  --summary-out output/metrics/qwen3_8b/summary.json \
  output/metrics/qwen3_8b/{compression_ratio,load_time_reduction,accuracy_loss,train_context_gain}/result.json
```

## Criteria

The current pass/fail criteria are:

- Loss degradation: `(memrift_final_lm_loss - pure_lora_final_lm_loss) /
  pure_lora_final_lm_loss <= 1%`.
- Compression ratio: storage saving `>= 30%` versus BF16 reference bytes.
- Training context gain: MemRift context length `>= 20%` longer than pure LoRA baseline.
- Load-time reduction: MemRift compressed-weight disk read time `>= 30%`
  lower than baseline raw model weight disk read time. This script reads weight
  files only; it does not instantiate the model or run first forward.

These criteria map directly to `zhibiao.md`.
