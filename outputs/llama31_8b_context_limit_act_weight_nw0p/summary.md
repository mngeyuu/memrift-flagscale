# Llama-3.1-8B Single-GPU Context Sweep

- Selected physical GPU: `2`
- GPU selection mode: `arg`
- Timeout per run: `1800` seconds

| Variant | Last Success Seq | First Failure Seq | Failure Type | Last Success Max Allocated (MB) | Last Success Elapsed (ms) |
| --- | ---: | ---: | --- | ---: | ---: |
| Pure LoRA | 3328 | 3456 | oom | 60986.38 | 23532.9 |
| Act+Weight MemRift | 3456 | 3584 | oom | 60123.45 | 130362.0 |

- Act+Weight MemRift extends the last-success context by `128` tokens versus pure LoRA.
