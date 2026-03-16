# MemRift 示例：从 run.py / train.py 跑通训练

本示例用于验证 MemRift 是否已完整集成到 FlagScale 训练流程（`run.py` → Megatron backend → `train_gpt.py` → `train.py` → `inject_memrift_if_configured`）。

## 前置条件

1. **CUDA 扩展**：安装 `flagscale/compress/float_split_stride_pin`（见仓库说明）。
2. **压缩权重**：使用 `scripts/memrift_prepare_weight.sh` 或 `prepare_weight.py` 生成压缩权重目录（含 `index.json` 与 `*.bin`）。
3. **数据与 tokenizer**：准备可用的 `data_path` 与 `tokenizer_path`（例如 TinyLlama 格式）。

## 使用 openassistant-guanaco

**直接使用与 memrift_demo 同源数据**：在配置中关闭 mock、将 `data_path` 设为 `timdettmers/openassistant-guanaco`，训练启动时会自动从 HuggingFace 拉取该数据集并转为 Megatron IndexedDataset（.bin/.idx），无需事先跑 preprocess_data。

```bash
# 仓库根目录执行：使用 guanaco 配置（mock_data: false, data_path: timdettmers/openassistant-guanaco）
python run.py --config-path=examples/memrift/conf --config-name=train_guanaco action=run \
  train.system.memrift_compressed_weight_dir=./memrift_weights/tinyllama_1b_level18
```

数据会缓存在 `./data/guanaco_flagscale/`，与 memrift_demo 相同的预处理（`data_prepare`）和 tokenization（max_length、padding、truncation）。

## 使用与 memrift_demo 相同的数据（手动准备 JSONL / .bin）

若希望**手动**准备数据或使用自定义条数，可按以下步骤准备：

1. **生成与 demo 同源的 JSONL**（在仓库根目录执行）：
   ```bash
   python scripts/data/prepare_guanaco_for_flagscale.py \
     --dataset timdettmers/openassistant-guanaco \
     --outdir ./data/guanaco \
     --batch_size 8 \
     --max_samples 1000
   ```
   得到 `./data/guanaco/guanaco_train.jsonl`，内容与 memrift_demo 的 `data_prepare()` 逻辑一致。

2. **转为 Megatron 格式（.bin/.idx）**：使用 Megatron-LM 的 `preprocess_data.py`，用与 demo 相同的 tokenizer（如 TinyLlama）对该 JSONL 做 tokenize，得到 `*_text_document.bin` 与 `*_text_document.idx`。将训练配置中的 `data_path` 指向该数据前缀（例如 `./data/guanaco/guanaco_text_document`）。

3. **配置示例**：在 `run.py` 中覆盖数据路径：
   ```bash
   train.data.data_path=/path/to/guanaco_text_document
   ```
   数据逻辑与 memrift_demo 的 `--dataset`、`data_prepare`、tokenizer 设置保持一致即可。

## 从 run.py 跑（推荐）

在**仓库根目录**（即 `FlagScale` 目录）下执行：

```bash
# 指定路径与少量 iter 做冒烟测试（约 5 个 iter）
export MEMRIFT_WEIGHT_DIR=/path/to/memrift_weights/tinyllama_1b_level18
export DATA_PATH=/path/to/your/data
export TOKENIZER_PATH=/path/to/TinyLlama-1.1B-Chat-v1.0
./scripts/run_memrift_smoke.sh
```

仅查看将要执行的命令（不真正跑训练）：

```bash
./scripts/run_memrift_smoke.sh dryrun
```

自定义迭代次数（默认 5）：

```bash
TRAIN_ITERS=10 ./scripts/run_memrift_smoke.sh
```

等价的手动命令（不通过脚本）：

```bash
python run.py \
  --config-path=examples/memrift/conf \
  --config-name=train \
  action=run \
  train.trainer.train_iters=5 \
  train.system.memrift_compressed_weight_dir=/path/to/weights \
  train.data.data_path=/path/to/data \
  train.model.tokenizer_path=/path/to/tokenizer
```

## 配置说明

- 主配置：`conf/train.yaml`（指定 `backend: megatron`、`entrypoint: flagscale/train/train_gpt.py`）。
- 默认合并：`conf/train/tinyllama_1b_lora_memrift.yaml`（MemRift 开关、LoRA、数据/模型等）。
- Runner 会把 `train.system` / `train.model` / `train.data` / `train.trainer` 展平为命令行参数传给 `train_gpt.py`（如 `--memrift-enable`、`--memrift-compressed-weight-dir` 等）。

## 验证 MemRift 是否生效

- 日志中应出现 MemRift 相关初始化（如 `inject_memrift_if_configured`、权重释放与 prefetch）。
- 若开启 `memrift_print_debug: true`，会有更详细的调试输出。
- 跑完设定的 `train_iters` 且无报错即可认为本示例集成通过。

## 迭代时长与显存峰值在哪里看

训练由 Runner 调起后，**标准输出**会重定向到日志文件，而不是当前终端。

### 日志文件位置

- 单机：`outputs/memrift_example/logs/host_0_localhost.output`
- 若 `no_shared_fs`：`outputs/memrift_example/logs/host.output`

即：**`<experiment.exp_dir>/logs/`** 下对应 host 的 `.output` 文件。

### 迭代时长（elapsed time per iteration）

- **位置**：上述 `.output` 文件里，每隔 **`log_interval`** 个 iteration 会打一行训练日志（默认 `log_interval: 10`）。
- **字段**：`elapsed time per iteration (ms): 1234.5`（单位：毫秒）。
- 代码位置：`flagscale/train/train.py` 中 `training_log()`，约 1979–1980 行。

示例一行：

```text
iteration       10/      50 | ... | elapsed time per iteration (ms): 1234.5 | ... | loss scale: ...
```

### 显存峰值（max allocated / max reserved）

- **位置**：同一 `.output` 文件中。
- **时机**：在**第一次**达到 `log_interval` 的 iteration 时（optimizer 初始化完成后），会调用一次 `report_memory()`，打印当前 rank 的显存。
- **字段**：
  - `max allocated`：到当前时刻为止的**分配峰值**（MB）
  - `max reserved`：到当前时刻为止的**保留峰值**（MB）

示例：

```text
[Rank 0] (after 10 iterations) memory (MB) | allocated: 1234.5 | max allocated: 5678.9 | reserved: ... | max reserved: ...
```

- **TensorBoard**：若在 system 中开启 `log_memory_to_tensorboard: true`，显存会写入 TensorBoard（如 `mem-max-allocated-bytes`），可在 `outputs/memrift_example/tensorboard` 用 `tensorboard --logdir=...` 查看。

## 内存 Profiling（看训练时显存使用）

需要系统查看 TinyLlama MemRift 训练过程中的显存曲线与峰值时，可用专用脚本开启内存相关参数：

```bash
# 在仓库根目录执行：开启 log_memory_to_tensorboard、record_memory_history，并写内存快照
./scripts/profile_tinyllama_memrift_memory.sh
```

脚本会默认开启：

- **log_memory_to_tensorboard**：每 `tensorboard_log_interval` 将当前/峰值显存写入 TensorBoard。
- **record_memory_history**：每 `log_interval` 将 CUDA 内存快照写入 `memory_snapshot_path`（默认 `memrift_memory_snapshot.pickle`）。

**查看方式：**

1. **日志**：`outputs/memrift_example/logs/host_0_localhost.output` 中 `report_memory()` 的 `max allocated` / `max reserved` (MB)。
2. **TensorBoard 曲线**：`tensorboard --logdir=outputs/memrift_example/tensorboard`，查看 `mem-allocated-bytes`、`mem-max-allocated-bytes` 等。
3. **内存快照**：`memrift_memory_snapshot.pickle` 可用 PyTorch 官方可视化工具查看分配历史。

如需同时抓 PyTorch Profiler trace（含时间线与显存），可设：

```bash
ENABLE_PYTORCH_PROFILER=1 ./scripts/profile_tinyllama_memrift_memory.sh
```

trace 会写入同一 tensorboard 目录，在 TensorBoard 的 PyTorch Profiler 面板查看。

## 不依赖 FlagScale 框架：独立脚本跑 Llama 1.1B + 记录显存峰值

若只想用 **FlagScale 重构的 MemRift 代码**（`flagscale/compress/memrift`）跑 TinyLlama 1.1B 训练并记录显存峰值，而不走 `run.py` / Megatron 流程，可使用独立脚本：

```bash
# 仓库根目录执行（若无压缩权重会先自动执行 prepare_weight）
./scripts/run_memrift_standalone_llama1.1b.sh
```

或手动指定参数：

```bash
# 1）准备压缩权重（若尚未准备）
python -m flagscale.compress.memrift.offline_comp.prepare_weight \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --outdir ./memrift_weights/tinyllama_1b_level18 --level 18

# 2）运行训练并记录峰值
python scripts/train_memrift_standalone_llama1.1b.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --compressed_weights ./memrift_weights/tinyllama_1b_level18 \
  --steps 5 --max_length 512 --profile_memory \
  --output_peak ./scripts/standalone_peak_memory.json
```

可选：`--activation` 开启激活压缩；`--output_peak <path>` 将峰值 (MB) 写入 JSON。脚本使用 HuggingFace Transformers + PEFT LoRA，仅依赖 `flagscale.compress.memrift` 的 `CompressedParam`、`AsyncCompressor`、`DecoderLayerWrapper` 等，不依赖 FlagScale 的 train.py / Megatron。
