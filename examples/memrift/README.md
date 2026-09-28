# MemRift Metric Configs

This directory keeps only the YAML configs needed by the MemRift metric scripts.

The supported metric models are:

- `llama8b`: LLaMA-3.1-8B
- `aquila`: Aquila2-7B

The metric scripts live under `scripts/metrics/` and run four checks:

- GSM8K CoT and HellaSwag relative accuracy loss versus pure LoRA
- compression ratio
- training context gain
- inference weight disk-read time reduction

Training checks are single-GPU and use Alpaca by default:

```bash
/share/project/mengyc/data/alpaca_megatron/alpaca_text_document
```

## YAML Layout

Hydra entry configs:

```text
conf/train_llama31_8b_mock.yaml
conf/train_aquila2_7b_mock.yaml
```

Model configs:

```text
conf/train/llama31_8b_lora_memrift.yaml
conf/train/llama31_8b_lora_memrift_mock.yaml
conf/train/aquila2_7b_lora_memrift.yaml
conf/train/aquila2_7b_lora_memrift_mock.yaml
```

The `*_mock` names are retained for compatibility with the existing
`scripts/metrics/*/model_env.sh` defaults. They are no longer mock-data configs:
the metric scripts override them to use Alpaca and single-GPU MemRift
weight+activation async mode.

## Run

Run all LLaMA-3.1-8B metric scripts:

```bash
cd /share/project/mengyc/code/memrift-flagscale
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/accuracy_loss.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/compression_ratio.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/train_context_gain.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/llama8b/load_time_reduction.sh
```

Run all Aquila2-7B metric scripts:

```bash
cd /share/project/mengyc/code/memrift-flagscale
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/accuracy_loss.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/compression_ratio.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/train_context_gain.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/metrics/aquila/load_time_reduction.sh
```
