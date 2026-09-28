# MemRift 第三方验收测试手册

本文档用于指导第三方在已准备好的训练镜像中，从零部署
`memrift-flagscale`，自行下载模型、准备数据、生成 MemRift 压缩权重，并复现
`zhibiao.md` 中定义的验收指标。

本手册默认第三方已经具备可用训练镜像。本文不负责驱动、系统库、PyTorch
后端和训练框架的安装，只负责在镜像内完成环境检查、模型/数据准备、测试脚本运行和结果判定。

## 1. 验收范围

验收对象：

- 代码仓库：`memrift-flagscale`
- 业界主流模型：`LLaMA-3.1-8B` base 模型
- 智源自研模型：`Aquila2-7B` base/预训练模型
- 训练数据：Alpaca，经 tokenizer 转换为 Megatron indexed dataset
- 测试方式：单卡纯 LoRA 与 MemRift+LoRA 对比
- 指标脚本：`scripts/metrics/llama8b/` 与 `scripts/metrics/aquila/`

不在本文验收范围内：

- 训练镜像构建
- 芯片驱动安装
- PyTorch 后端安装
- 多机多卡性能调优
- 使用 chat、instruct、SFT、RLHF 模型替代 base 模型后的效果对比

## 2. 指标与通过标准

`zhibiao.md` 中的指标被拆解为四个可自动化测试项。两个模型均需分别测试。

| 指标 | 测试项 | 脚本 | 通过标准 |
| --- | --- | --- | --- |
| 技术指标 1.1 | 精度损耗 | `accuracy_loss.sh` | GSM8K CoT、HellaSwag 各自相对纯 LoRA 的 accuracy 下降 `<= 1%` |
| 技术指标 1.1 | 压缩比例 | `compression_ratio.sh` | 压缩权重存储节省 `>= 30%` |
| 技术指标 1.2 | 训练上下文长度 | `train_context_gain.sh` | MemRift+LoRA 最大可运行上下文长度比纯 LoRA 提升 `>= 20%` |
| 技术指标 1.2 | 推理权重读盘时间 | `load_time_reduction.sh` | MemRift 压缩权重读盘耗时相对原始模型权重读盘耗时降低 `>= 30%` |

通过判定以各 `result.json` 中的 `pass: true` 为准。

## 3. 目录规划

第三方机器上的路径可以不同，但建议统一使用以下环境变量。后续命令均以这些变量为准。

```bash
export WORKDIR=/workspace/memrift_delivery
export REPO_ROOT=$WORKDIR/memrift-flagscale
export MODEL_ROOT=$WORKDIR/models
export DATA_ROOT=$WORKDIR/data
export WEIGHT_ROOT=$WORKDIR/memrift_weights
export METRICS_OUTPUT_ROOT=$WORKDIR/output/metrics
```

开发环境示例路径如下，仅用于对照，不要求第三方机器存在这些路径：

```bash
export WORKDIR=/share/project/mengyc
export REPO_ROOT=/share/project/mengyc/code/memrift-flagscale
export MODEL_ROOT=/share/project/mengyc/models
export DATA_ROOT=/share/project/mengyc/data
```

建议第三方在容器内创建 `env.sh`：

```bash
cat > env.sh <<'EOF'
export WORKDIR=/workspace/memrift_delivery
export REPO_ROOT=$WORKDIR/memrift-flagscale
export MODEL_ROOT=$WORKDIR/models
export DATA_ROOT=$WORKDIR/data
export WEIGHT_ROOT=$WORKDIR/memrift_weights
export METRICS_OUTPUT_ROOT=$WORKDIR/output/metrics

export LLAMA_MODEL_PATH=$MODEL_ROOT/Llama-3.1-8B
export AQUILA_MODEL_PATH=$MODEL_ROOT/Aquila2-7B

export ALPACA_RAW_JSON=$DATA_ROOT/alpaca_raw/alpaca_data.json
export ALPACA_JSONL=$DATA_ROOT/alpaca_raw/alpaca_text.jsonl
export LLAMA_ALPACA_DATA_PATH=$DATA_ROOT/alpaca_megatron_llama/alpaca_text_document
export AQUILA_ALPACA_DATA_PATH=$DATA_ROOT/alpaca_megatron_aquila/alpaca_text_document

export LLAMA_MEMRIFT_WEIGHT_DIR=$WEIGHT_ROOT/llama31_8b_level18
export AQUILA_MEMRIFT_WEIGHT_DIR=$WEIGHT_ROOT/aquila2_7b_level18

export MEMRIFT_PREPARE_LEVEL=18
export WANDB_MODE=offline
export HF_HUB_DISABLE_XET=1
EOF

source env.sh
mkdir -p "$WORKDIR" "$MODEL_ROOT" "$DATA_ROOT" "$WEIGHT_ROOT" "$METRICS_OUTPUT_ROOT"
```

## 4. 交付物清单

| 类型 | 内容 | 是否由交付方提供 | 是否由第三方准备 | 说明 |
| --- | --- | --- | --- | --- |
| 代码 | `memrift-flagscale` | 是 | 部署到 `$REPO_ROOT` | 本手册所有命令在仓库根目录执行 |
| 训练镜像 | 芯片适配后的训练镜像 | 否 | 是 | 本手册只做镜像内环境检查 |
| 模型 | `LLaMA-3.1-8B` base | 否 | 是 | 不使用 Instruct |
| 模型 | `Aquila2-7B` base/预训练权重 | 否 | 是 | 不使用 Chat/SFT/RLHF |
| 数据 | Alpaca 原始数据 | 否 | 是 | 需转换成 JSONL 再转换成 Megatron indexed dataset |
| 压缩权重 | MemRift compressed weights | 否 | 自动生成或手动生成 | 默认压缩等级 `18` |
| 输出 | 指标 JSON | 自动生成 | 自动生成 | 位于 `$METRICS_OUTPUT_ROOT` |

## 5. 训练镜像环境检查

以下命令均在训练镜像容器内执行。

```bash
source env.sh
cd "$REPO_ROOT"
```

检查 Python：

```bash
which python
python --version
```

检查 PyTorch 与设备后端：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available() if hasattr(torch, "cuda") else False)
print("cuda device count:", torch.cuda.device_count() if hasattr(torch, "cuda") else 0)
if hasattr(torch, "cuda") and torch.cuda.is_available():
    print("cuda device 0:", torch.cuda.get_device_name(0))
PY
```

国产芯片环境可按实际平台补充检查：

```bash
python - <<'PY'
mods = ["torch_npu", "torch_musa", "torch_xpu"]
for m in mods:
    try:
        __import__(m)
        print(m, "import ok")
    except Exception as e:
        print(m, "not available:", type(e).__name__)
PY
```

设备命令按平台选择执行：

```bash
nvidia-smi || true
npu-smi info || true
mthreads-gmi || true
```

检查 FlagScale 与 MemRift 可导入：

```bash
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/flagscale:$REPO_ROOT/flagscale/train:$REPO_ROOT/flagscale/train/megatron:${PYTHONPATH:-}

python - <<'PY'
import flagscale
print("flagscale import ok")

import flagscale.compress.memrift
print("memrift import ok")
PY
```

检查关键依赖：

```bash
python - <<'PY'
import transformers
import zstandard
print("transformers ok")
print("zstandard ok")
PY
```

检查磁盘空间：

```bash
df -h "$WORKDIR"
du -sh "$REPO_ROOT" || true
du -sh "$MODEL_ROOT" || true
du -sh "$DATA_ROOT" || true
```

## 6. 代码部署

将代码部署到 `$REPO_ROOT`。示例：

```bash
source env.sh
cd "$WORKDIR"

# 示例：从代码包解压
# tar -xf memrift-flagscale.tar.gz
# mv memrift-flagscale "$REPO_ROOT"

# 示例：从 Git 仓库获取
# git clone <repo-url> "$REPO_ROOT"

cd "$REPO_ROOT"
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/flagscale:$REPO_ROOT/flagscale/train:$REPO_ROOT/flagscale/train/megatron:${PYTHONPATH:-}
```

确认指标脚本存在：

```bash
ls scripts/metrics/llama8b
ls scripts/metrics/aquila
ls examples/memrift/conf
```

## 7. 模型版本要求

本验收默认使用 base/预训练模型，不使用 chat、instruct、SFT 或 RLHF 后的对话模型。

| 模型类别 | 默认模型 | 版本要求 | 对应脚本 |
| --- | --- | --- | --- |
| 业界主流模型 | `LLaMA-3.1-8B` | base，非 Instruct | `scripts/metrics/llama8b/` |
| 智源自研模型 | `Aquila2-7B` | base/预训练权重，非 Chat/SFT | `scripts/metrics/aquila/` |

不得直接替换为：

- `LLaMA-3.1-8B-Instruct`
- `LLaMA-3.2-3B-Instruct`
- Aquila Chat / Aquila SFT / Aquila RLHF 等对话微调版本

如果第三方因权限或平台原因替换模型版本，必须在验收报告中记录：

- 模型 ID
- 下载源
- revision / commit
- 权重文件列表
- `config.json`
- `tokenizer_config.json`
- 是否包含 `chat_template`
- 文件校验和

替换模型后的结果只能作为平台适配测试，不作为本手册默认验收指标的等价复现。

## 8. 模型下载与检查

第三方可根据自身网络条件从 HuggingFace、ModelScope 或内网镜像下载模型。

ModelScope 示例：

```bash
source env.sh
mkdir -p "$MODEL_ROOT"

modelscope download \
  --model <LLaMA-3.1-8B-base模型ID> \
  --local_dir "$LLAMA_MODEL_PATH"

modelscope download \
  --model <Aquila2-7B-base模型ID> \
  --local_dir "$AQUILA_MODEL_PATH"
```

HuggingFace 示例：

```bash
source env.sh
mkdir -p "$MODEL_ROOT"

huggingface-cli download <LLaMA-3.1-8B-base模型ID> \
  --local-dir "$LLAMA_MODEL_PATH" \
  --local-dir-use-symlinks False

huggingface-cli download <Aquila2-7B-base模型ID> \
  --local-dir "$AQUILA_MODEL_PATH" \
  --local-dir-use-symlinks False
```

模型完整性检查：

```bash
source env.sh

python - <<'PY'
import json
import os
from pathlib import Path

models = [
    ("llama", "LLAMA_MODEL_PATH"),
    ("aquila", "AQUILA_MODEL_PATH"),
]

bad_words = ["instruct", "chat", "sft", "rlhf"]

for name, env in models:
    p = Path(os.environ[env])
    print(f"\n[{name}] {p}")

    if not p.exists():
        raise SystemExit(f"model path not found: {p}")

    lower_path = str(p).lower()
    for word in bad_words:
        if word in lower_path:
            raise SystemExit(f"{name}: model path looks like non-base model: {p}")

    config_path = p / "config.json"
    tokenizer_config_path = p / "tokenizer_config.json"

    if not config_path.exists():
        raise SystemExit(f"{name}: missing config.json")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    print("  model_type:", config.get("model_type"))
    print("  architectures:", config.get("architectures"))

    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        chat_template = tokenizer_config.get("chat_template")
        print("  chat_template:", "present" if chat_template else "absent")
        if chat_template:
            raise SystemExit(f"{name}: tokenizer_config.json contains chat_template; verify this is base model")
    else:
        print("  tokenizer_config.json: missing")

    weight_files = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    print("  weight_files:", len(weight_files))
    if not weight_files:
        raise SystemExit(f"{name}: no weight files found")

print("\nBase model checks passed.")
PY
```

tokenizer 检查：

```bash
source env.sh

python - <<'PY'
import os
from transformers import AutoTokenizer

for name, env in [
    ("llama", "LLAMA_MODEL_PATH"),
    ("aquila", "AQUILA_MODEL_PATH"),
]:
    path = os.environ[env]
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    ids = tok("MemRift verification.", return_tensors=None)["input_ids"]
    print(name, "vocab_size=", getattr(tok, "vocab_size", None), "sample_tokens=", len(ids))
PY
```

## 9. Alpaca 数据下载与 JSONL 准备

第三方需自行下载 Alpaca 原始数据。下载后建议放置为：

```bash
$ALPACA_RAW_JSON
```

即：

```bash
source env.sh
mkdir -p "$DATA_ROOT/alpaca_raw"
ls -lh "$ALPACA_RAW_JSON"
```

Alpaca 原始 JSON 通常为一个 list，每条样本包含：

- `instruction`
- `input`
- `output`

先将其转换为每行一个 `{"text": ...}` 的 JSONL 文件：

```bash
source env.sh

python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["ALPACA_RAW_JSON"])
dst = Path(os.environ["ALPACA_JSONL"])
dst.parent.mkdir(parents=True, exist_ok=True)

data = json.loads(src.read_text(encoding="utf-8"))

def format_record(x):
    instruction = (x.get("instruction") or "").strip()
    inp = (x.get("input") or "").strip()
    output = (x.get("output") or "").strip()

    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )

    return (
        "Below is an instruction that describes a task.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )

with dst.open("w", encoding="utf-8") as f:
    for item in data:
        text = format_record(item)
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print(f"wrote {dst}")
PY
```

检查 JSONL：

```bash
wc -l "$ALPACA_JSONL"
head -n 1 "$ALPACA_JSONL"
```

## 10. 转换为 Megatron Indexed Dataset

指标训练脚本不直接读取 Alpaca JSON 或 JSONL，而是读取 Megatron indexed dataset
前缀。也就是说，最终必须得到：

```bash
${ALPACA_DATA_PATH}.bin
${ALPACA_DATA_PATH}.idx
```

由于 LLaMA 与 Aquila 的 tokenizer 可能不同，推荐分别转换两份数据：

```bash
$LLAMA_ALPACA_DATA_PATH.bin
$LLAMA_ALPACA_DATA_PATH.idx

$AQUILA_ALPACA_DATA_PATH.bin
$AQUILA_ALPACA_DATA_PATH.idx
```

先定位训练镜像中的 Megatron 数据预处理脚本：

```bash
find "$REPO_ROOT" -name preprocess_data.py
find /workspace -name preprocess_data.py 2>/dev/null | head
```

设置预处理脚本路径：

```bash
export MEGATRON_PREPROCESS=/path/to/Megatron-LM/tools/preprocess_data.py
python "$MEGATRON_PREPROCESS" --help | head -n 80
```

不同训练镜像中的 `preprocess_data.py` 参数可能略有差异。以下命令为常见 Megatron-LM
用法，实际以镜像中的 `--help` 为准。

转换 LLaMA 数据：

```bash
source env.sh
mkdir -p "$(dirname "$LLAMA_ALPACA_DATA_PATH")"

python "$MEGATRON_PREPROCESS" \
  --input "$ALPACA_JSONL" \
  --output-prefix "$LLAMA_ALPACA_DATA_PATH" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$LLAMA_MODEL_PATH" \
  --json-keys text \
  --append-eod \
  --workers 8
```

转换 Aquila 数据：

```bash
source env.sh
mkdir -p "$(dirname "$AQUILA_ALPACA_DATA_PATH")"

python "$MEGATRON_PREPROCESS" \
  --input "$ALPACA_JSONL" \
  --output-prefix "$AQUILA_ALPACA_DATA_PATH" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$AQUILA_MODEL_PATH" \
  --json-keys text \
  --append-eod \
  --workers 8
```

如果镜像中的预处理工具不支持 `HuggingFaceTokenizer` 或 `--tokenizer-model`，
请根据 `python "$MEGATRON_PREPROCESS" --help` 修改 tokenizer 参数，但必须保证：

- LLaMA 数据使用 LLaMA 模型目录中的 tokenizer；
- Aquila 数据使用 Aquila 模型目录中的 tokenizer；
- 输出前缀分别写入 `$LLAMA_ALPACA_DATA_PATH` 和 `$AQUILA_ALPACA_DATA_PATH`；
- 输出文件后缀为 `.bin` 和 `.idx`。

数据产物检查：

```bash
source env.sh

python - <<'PY'
from pathlib import Path
import os

for name, env in [
    ("llama alpaca", "LLAMA_ALPACA_DATA_PATH"),
    ("aquila alpaca", "AQUILA_ALPACA_DATA_PATH"),
]:
    p = Path(os.environ[env])
    print(f"[{name}] prefix={p}")
    for suffix in [".bin", ".idx"]:
        f = Path(str(p) + suffix)
        print(" ", f, f.stat().st_size if f.exists() else "MISSING")
        if not f.exists() or f.stat().st_size == 0:
            raise SystemExit(f"invalid dataset file: {f}")
PY
```

## 11. 国产芯片和非 CUDA 环境适配

当前指标脚本默认以 CUDA 命名为主，常见默认项包括：

- `CUDA_VISIBLE_DEVICES`
- `LOAD_DEVICE=cuda:0`
- `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
- 训练脚本中通过 Hydra 注入 `++experiment.envs.CUDA_VISIBLE_DEVICES`

如果第三方使用非 CUDA 国产芯片，需要根据训练镜像实际情况确认以下内容：

| 项 | CUDA 默认 | 国产芯片适配说明 |
| --- | --- | --- |
| 设备可见性变量 | `CUDA_VISIBLE_DEVICES=0` | 替换为厂商对应变量，如 `ASCEND_VISIBLE_DEVICES`、`MUSA_VISIBLE_DEVICES` 等 |
| load-time 设备名 | `LOAD_DEVICE=cuda:0` | 替换为镜像实际支持的 device，如 `npu:0`、`musa:0` 或兼容别名 |
| PyTorch 后端 | `torch.cuda` | 确认 `torch_npu`、`torch_musa` 或其他后端可用 |
| MemRift 扩展 | CUDA 路径 | 确认对应平台是否已有等价实现 |
| 脚本环境注入 | `++experiment.envs.CUDA_VISIBLE_DEVICES` | 如平台不识别该变量，需在脚本或运行命令中增加厂商变量 |

CUDA 示例：

```bash
export CUDA_VISIBLE_DEVICES=0
export LOAD_DEVICE=cuda:0
```

Ascend 示例，具体以训练镜像实际设备名为准：

```bash
export ASCEND_VISIBLE_DEVICES=0
export LOAD_DEVICE=npu:0
```

MUSA 示例，具体以训练镜像实际设备名为准：

```bash
export MUSA_VISIBLE_DEVICES=0
export LOAD_DEVICE=musa:0
```

如果平台不支持当前 MemRift 依赖的 CUDA-only 扩展或异步拷贝路径，则训练上下文和加载时间指标无法作为完整验收结果，只能作为平台适配问题记录。

## 12. MemRift 压缩权重准备

指标脚本会在缺少 `index.json` 时尝试自动生成压缩权重。也可以手动提前生成。

手动生成 LLaMA 压缩权重：

```bash
source env.sh
cd "$REPO_ROOT"

python3 -m flagscale.compress.memrift.offline_comp.prepare_weight \
  --model "$LLAMA_MODEL_PATH" \
  --outdir "$LLAMA_MEMRIFT_WEIGHT_DIR" \
  --level "$MEMRIFT_PREPARE_LEVEL"
```

手动生成 Aquila 压缩权重：

```bash
source env.sh
cd "$REPO_ROOT"

python3 -m flagscale.compress.memrift.offline_comp.prepare_weight \
  --model "$AQUILA_MODEL_PATH" \
  --outdir "$AQUILA_MEMRIFT_WEIGHT_DIR" \
  --level "$MEMRIFT_PREPARE_LEVEL"
```

检查压缩权重：

```bash
source env.sh

ls -lh "$LLAMA_MEMRIFT_WEIGHT_DIR/index.json"
find "$LLAMA_MEMRIFT_WEIGHT_DIR" -name '*.bin' | head

ls -lh "$AQUILA_MEMRIFT_WEIGHT_DIR/index.json"
find "$AQUILA_MEMRIFT_WEIGHT_DIR" -name '*.bin' | head
```

如需禁止脚本自动生成压缩权重，可设置：

```bash
export PREPARE_MEMRIFT_WEIGHTS=0
```

此时若压缩权重不存在，脚本会直接失败。

## 13. Smoke Test

正式验收前建议先执行短流程，确认配置和路径可用。

LLaMA dryrun：

```bash
source env.sh
cd "$REPO_ROOT"

export MODEL_PATH="$LLAMA_MODEL_PATH"
export MEMRIFT_WEIGHT_DIR="$LLAMA_MEMRIFT_WEIGHT_DIR"
export ALPACA_DATA_PATH="$LLAMA_ALPACA_DATA_PATH"
export OUT_ROOT="$METRICS_OUTPUT_ROOT/llama8b_smoke"

ACTION=dryrun \
TRAIN_ITERS=1 \
BASE_SEQ_LEN=128 \
TARGET_SEQ_LEN=160 \
bash scripts/metrics/llama8b/train_context_gain.sh
```

短训练 smoke test：

```bash
source env.sh
cd "$REPO_ROOT"

export MODEL_PATH="$LLAMA_MODEL_PATH"
export MEMRIFT_WEIGHT_DIR="$LLAMA_MEMRIFT_WEIGHT_DIR"
export ALPACA_DATA_PATH="$LLAMA_ALPACA_DATA_PATH"
export OUT_ROOT="$METRICS_OUTPUT_ROOT/llama8b_smoke"

TRAIN_ITERS=1 \
BASE_SEQ_LEN=1024 \
CONTEXT_SEARCH_START=1024 \
CONTEXT_SEARCH_STEP=256 \
CONTEXT_SEARCH_CAP=1536 \
bash scripts/metrics/llama8b/train_context_gain.sh
```

Smoke test 通过后再执行正式指标。

## 14. LLaMA-3.1-8B 指标测试

运行前设置 LLaMA 相关路径：

```bash
source env.sh
cd "$REPO_ROOT"

export MODEL_PATH="$LLAMA_MODEL_PATH"
export MEMRIFT_WEIGHT_DIR="$LLAMA_MEMRIFT_WEIGHT_DIR"
export ALPACA_DATA_PATH="$LLAMA_ALPACA_DATA_PATH"
export OUT_ROOT="$METRICS_OUTPUT_ROOT/llama8b"
```

逐项运行：

```bash
bash scripts/metrics/llama8b/compression_ratio.sh
bash scripts/metrics/llama8b/load_time_reduction.sh
bash scripts/metrics/llama8b/accuracy_loss.sh
bash scripts/metrics/llama8b/train_context_gain.sh
```

如果仓库中存在汇总脚本，也可运行：

```bash
bash scripts/metrics/llama8b/run_all_metrics.sh
```

输出文件：

```text
$METRICS_OUTPUT_ROOT/llama8b/compression_ratio/result.json
$METRICS_OUTPUT_ROOT/llama8b/load_time_reduction/result.json
$METRICS_OUTPUT_ROOT/llama8b/accuracy_loss/result.json
$METRICS_OUTPUT_ROOT/llama8b/train_context_gain/result.json
```

其中 `load_time_reduction.sh` 只测试推理载入阶段的权重磁盘读取耗时：
baseline 读取 `$MODEL_PATH` 下的原始权重文件，MemRift 读取
`$MEMRIFT_WEIGHT_DIR` 下的 `index.json` 和压缩权重 payload 文件。该脚本不实例化
模型，也不运行首个 forward。读盘时间会受到 OS page cache 影响；如需冷缓存结果，
应按验收机器策略在两个分支测试前清理 page cache 或使用干净环境。

上下文长度搜索默认上限可能不足以测出真实边界。如 `result.json` 中出现
`baseline_hit_search_cap` 或 `memrift_hit_search_cap` 为 `true`，应提高搜索上限：

```bash
CONTEXT_SEARCH_CAP=12288 bash scripts/metrics/llama8b/train_context_gain.sh
```

## 15. Aquila2-7B 指标测试

运行前设置 Aquila 相关路径：

```bash
source env.sh
cd "$REPO_ROOT"

export MODEL_PATH="$AQUILA_MODEL_PATH"
export MEMRIFT_WEIGHT_DIR="$AQUILA_MEMRIFT_WEIGHT_DIR"
export ALPACA_DATA_PATH="$AQUILA_ALPACA_DATA_PATH"
export OUT_ROOT="$METRICS_OUTPUT_ROOT/aquila"
```

逐项运行：

```bash
bash scripts/metrics/aquila/compression_ratio.sh
bash scripts/metrics/aquila/load_time_reduction.sh
bash scripts/metrics/aquila/accuracy_loss.sh
bash scripts/metrics/aquila/train_context_gain.sh
```

输出文件：

```text
$METRICS_OUTPUT_ROOT/aquila/compression_ratio/result.json
$METRICS_OUTPUT_ROOT/aquila/load_time_reduction/result.json
$METRICS_OUTPUT_ROOT/aquila/accuracy_loss/result.json
$METRICS_OUTPUT_ROOT/aquila/train_context_gain/result.json
```

其中 `load_time_reduction.sh` 的测试口径与 LLaMA 一致：比较原始模型权重文件读盘
耗时与 MemRift 压缩权重目录读盘耗时，不实例化模型，也不运行首个 forward。

## 16. 结果汇总与判定

运行以下脚本检查所有指标是否通过：

```bash
source env.sh

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["METRICS_OUTPUT_ROOT"])
paths = [
    root / "llama8b/compression_ratio/result.json",
    root / "llama8b/load_time_reduction/result.json",
    root / "llama8b/accuracy_loss/result.json",
    root / "llama8b/train_context_gain/result.json",
    root / "aquila/compression_ratio/result.json",
    root / "aquila/load_time_reduction/result.json",
    root / "aquila/accuracy_loss/result.json",
    root / "aquila/train_context_gain/result.json",
]

ok = True
for path in paths:
    if not path.is_file():
        print(f"[MISSING] {path}")
        ok = False
        continue

    data = json.loads(path.read_text(encoding="utf-8"))
    passed = bool(data.get("pass"))
    metric = data.get("metric", path.parent.name)
    print(f"[{'PASS' if passed else 'FAIL'}] {metric}: {path}")
    ok = ok and passed

raise SystemExit(0 if ok else 1)
PY
```

建议验收报告至少包含：

- 训练镜像名称和版本
- 芯片型号、数量、驱动版本
- PyTorch 版本和后端版本
- `memrift-flagscale` commit id
- LLaMA 模型 ID、revision、校验和
- Aquila 模型 ID、revision、校验和
- Alpaca 数据源和转换命令
- MemRift 压缩等级
- 八个 `result.json`
- 是否修改过脚本或 YAML 配置

## 17. 常见问题

| 现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| `model path not found` | `MODEL_PATH` 指向错误 | 检查 `LLAMA_MODEL_PATH` / `AQUILA_MODEL_PATH` |
| `No weight files found` | 模型未下载完整 | 重新下载模型，确认 `.safetensors` 或 `.bin` 文件存在 |
| 检测到 `chat_template` | 可能下载了 chat/instruct 模型 | 更换为 base/预训练模型 |
| 找不到 Alpaca `.bin/.idx` | 未执行 Megatron 数据转换 | 按第 10 章重新转换 |
| 数据路径带了 `.bin` 后缀 | `ALPACA_DATA_PATH` 应为 prefix | 设置为不带 `.bin/.idx` 的前缀 |
| `preprocess_data.py` 参数不兼容 | 镜像中的 Megatron 版本不同 | 以 `--help` 为准调整 tokenizer 参数 |
| `index.json` 缺失 | 压缩权重未生成 | 运行第 12 章 `prepare_weight` |
| `cuda:0` 不可用 | 国产芯片设备名不同 | 设置 `LOAD_DEVICE` 为镜像实际支持的设备 |
| 训练脚本仍注入 `CUDA_VISIBLE_DEVICES` | 脚本默认 CUDA 命名 | 在脚本或运行命令中增加厂商设备变量 |
| context 测试达到搜索上限 | `CONTEXT_SEARCH_CAP` 太低 | 提高 `CONTEXT_SEARCH_CAP` |
| OOM | batch/seq/cap 超出设备能力 | 先跑 smoke test，再逐步提高上下文上限 |
| 精度指标波动 | 模型版本、数据转换或随机性差异 | 固定模型 revision、数据转换命令和训练参数 |

## 18. 验收边界说明

以下变更会导致结果不能与默认验收基线直接等价比较：

- 使用 Instruct/Chat/SFT/RLHF 模型替代 base 模型；
- 使用不同的 LLaMA 或 Aquila 权重版本；
- LLaMA 与 Aquila 共用同一份由单一 tokenizer 转换的 indexed dataset；
- 修改 Alpaca 文本模板；
- 替换数据集；
- 修改 LoRA 超参数；
- 修改 MemRift 压缩等级；
- 修改训练脚本中 MemRift 开关；
- 国产芯片后端未提供与 CUDA 路径等价的 MemRift 扩展能力。

如需进行平台适配测试，应在验收报告中单独标注“适配测试”，并记录所有修改项。
