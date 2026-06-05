# MemRift Split Path Benchmark

## Environment

- GPU: MetaX C550
- PyTorch: 2.8.0+metax3.3.0.2
- Stream mode: current
- Command: `python scripts/benchmark_memrift_split_paths.py --warmup 0 --iters 1 --out outputs/memrift_split_path_llama31_8b_bf16.json`
- Result JSON: `outputs/memrift_split_path_llama31_8b_bf16.json`

## Isolated Benchmark

| Shape | Numel | Mapped CUDA Event ms | Copy CUDA Event ms | Ratio |
|---|---:|---:|---:|---:|
| `(32, 2048, 2048)` | 134217728 | 23803.055 | 6.014 | 3957.97x |
| `(2048, 28672)` | 58720256 | 10418.868 | 2.932 | 3553.23x |
| `(2048, 1, 14336)` | 29360128 | 5207.108 | 1.667 | 3123.03x |
| `(2048, 1, 4096)` | 8388608 | 1487.483 | 0.572 | 2602.10x |
| `(32, 2048, 128)` | 8388608 | 1486.199 | 0.552 | 2692.70x |

## CPU Compression Cost

| Shape | Mapped zstd ms | Copy zstd ms |
|---|---:|---:|
| `(32, 2048, 2048)` | 104416.941 | 108879.806 |
| `(2048, 28672)` | 47555.271 | 48172.466 |
| `(2048, 1, 14336)` | 23520.326 | 23158.035 |
| `(2048, 1, 4096)` | 5756.927 | 5624.628 |
| `(32, 2048, 128)` | 5556.453 | 5500.173 |

## Decision

Use the copy split path for activation experiments: `MEMRIFT_ACT_SPLIT_PATH=copy`.

The current host-mapped path is the dominant CUDA-side cost. For the largest activation shape, writing exponent bytes directly to mapped pinned host memory took about 23.8 seconds, while GPU staging plus `cudaMemcpyAsync` took about 6 ms. zstd remains expensive and is roughly identical between the two paths because both compress the same exponent bytes; this benchmark isolates the transfer/split path difference.

Training-level validation should run with:

```bash
export MEMRIFT_ACT_SPLIT_PROFILE=1
export MEMRIFT_ACT_SPLIT_PATH=copy
```

and compare `[MemRiftSplitProfile]` lines against the same command with `MEMRIFT_ACT_SPLIT_PATH=mapped`.
