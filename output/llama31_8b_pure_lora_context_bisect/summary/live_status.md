# Llama-3.1-8B Pure LoRA Context Bisect Status

- Snapshot time: `2026-06-03T14:13:36+08:00`
- Detached launcher: pid=166071 (exited)
- Current trial seq_length: `9728`
- Latest completed trial: seq=`8448`, status=`PASS`
- Results TSV: `/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/summary/results.tsv`
- Final result file: `/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/summary/final_result.txt`
- Current host log: `/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/trials/seq_9728/logs/host_0_localhost.output`
- Current stdout log: `/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/logs/seq_9728.stdout.log`

## Recent Results

```tsv
4	9728	OOM	0	60	/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/trials/seq_9728/logs/host_0_localhost.output
5	5632	PASS	0	96	/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/trials/seq_5632/logs/host_0_localhost.output
6	7680	PASS	0	98	/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/trials/seq_7680/logs/host_0_localhost.output
7	8704	OOM	0	64	/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/trials/seq_8704/logs/host_0_localhost.output
8	8192	PASS	0	96	/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/trials/seq_8192/logs/host_0_localhost.output
9	8448	PASS	0	95	/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/trials/seq_8448/logs/host_0_localhost.output
```

## Final Summary Tail

```text
Llama-3.1-8B pure LoRA context bisect summary
out_root=/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect
results_tsv=/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/summary/results.tsv
max_runnable_seq_length=8448
best_pass_host_log=/home/secure/myc/memrift-flagscale/output/llama31_8b_pure_lora_context_bisect/trials/seq_8448/logs/host_0_localhost.output
first_failing_seq_length=8704
failing_status=OOM
```

## Error Signals

```text
8:[default0]:/opt/conda/lib/python3.10/site-packages/modelopt/torch/utils/logging.py:115: UserWarning: Failed to import diffusers plugin due to: RuntimeError("Failed to import diffusers.models.modeling_utils because of the following error (look up to see its traceback):\nname 'logger' is not defined"). You may ignore this warning if you do not need this plugin.
18:[default0]:/opt/conda/lib/python3.10/site-packages/modelopt/torch/utils/logging.py:115: UserWarning: Failed to import diffusers plugin due to: RuntimeError("Failed to import diffusers.loaders.single_file_model because of the following error (look up to see its traceback):\nname 'logger' is not defined"). You may ignore this warning if you do not need this plugin.
1077:[default0]:WARNING:megatron.core.utils:CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 63.59 GiB of which 242.31 MiB is free. Of the allocated memory 61.08 GiB is allocated by PyTorch, and 591.85 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
1078:[default0]:[rank0]: Traceback (most recent call last):
1129:[default0]:[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 63.59 GiB of which 242.31 MiB is free. Of the allocated memory 61.08 GiB is allocated by PyTorch, and 591.85 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
1132:[default0]:['Traceback (most recent call last):\n', '  File "/home/secure/myc/memrift-flagscale/flagscale/train/train_gpt.py", line 157, in forward_step\n    output_tensor = model(\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl\n    return forward_call(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/megatron/core/distributed/data_parallel_base.py", line 22, in forward\n    return self.module(*inputs, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl\n    return forward_call(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/megatron/core/transformer/module.py", line 446, in forward\n    outputs = self.module(*inputs, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl\n    return forward_call(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/megatron/core/models/gpt/gpt_model.py", line 470, in forward\n    return self._postprocess(\n', '  File "/opt/conda/lib/python3.10/site-packages/megatron/core/models/gpt/gpt_model.py", line 606, in _postprocess\n    logits, _ = self.output_layer(\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl\n    return self._call_impl(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1784, in _call_impl\n    return forward_call(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/megatron/core/tensor_parallel/layers.py", line 1029, in forward\n    output_parallel = self._forward_impl(\n', '  File "/opt/conda/lib/python3.10/site-packages/megatron/core/tensor_parallel/layers.py", line 956, in _forward_impl\n    return linear_with_frozen_weight(input, weight, *args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/megatron/core/tensor_parallel/layers.py", line 435, in linear_with_frozen_weight\n    return LinearWithFrozenWeight.apply(*args)\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/autograd/function.py", line 576, in apply\n    return super().apply(*args, **kwargs)  # type: ignore[misc]\n', '  File "/opt/conda/lib/python3.10/site-packages/torch/amp/autocast_mode.py", line 517, in decorate_fwd\n    return fwd(*args, **kwargs)\n', '  File "/opt/conda/lib/python3.10/site-packages/megatron/core/tensor_parallel/layers.py", line 336, in forward\n    output = torch.matmul(input, weight.t())\n', 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.32 GiB. GPU 0 has a total capacity of 63.59 GiB of which 242.31 MiB is free. Of the allocated memory 61.08 GiB is allocated by PyTorch, and 591.85 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)\n']
1133:Traceback (most recent call last):
1145:    raise ChildFailedError(
1146:torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
1148:flagscale/train/train_gpt.py FAILED
```

## GPU Snapshot

```text
| 3     MetaX C550 | 3           Off | 0000:66:00.0        | 0%          Disabled |
| 96W / 450W       | 44C          P0 | 859/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 4     MetaX C550 | 4           Off | 0000:a3:00.0        | 0%          Disabled |
| 95W / 450W       | 44C          P0 | 859/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 5     MetaX C550 | 5           Off | 0000:a4:00.0        | 0%          Disabled |
| 96W / 450W       | 39C          P0 | 859/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 6     MetaX C550 | 6           Off | 0000:e3:00.0        | 0%          Disabled |
| 91W / 450W       | 38C          P0 | 859/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+
| 7     MetaX C550 | 7           Off | 0000:e4:00.0        | 0%          Disabled |
| 95W / 450W       | 43C          P0 | 859/65536 MiB       | Available            |
+------------------+-----------------+---------------------+----------------------+

+---------------------------------------------------------------------------------+
| Process:                                                                        |
|  GPU                    PID         Process Name                 GPU Memory     |
|                                                                  Usage(MiB)     |
|=================================================================================|
|  no process found                                                               |
+---------------------------------------------------------------------------------+

End of Log
```
