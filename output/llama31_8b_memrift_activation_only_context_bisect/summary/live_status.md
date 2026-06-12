# Llama-3.1-8B MemRift Activation-Only Context Bisect Status

- Snapshot time: `2026-06-03T14:12:36+08:00`
- Detached launcher: pid=387351 (exited)
- Current trial seq_length: `2048`
- Latest completed trial: seq=`2048`, status=`FAIL`
- Results TSV: `/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/summary/results.tsv`
- Final result file: `/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/summary/final_result.txt`
- Current host log: `/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/trials/seq_2048/logs/host_0_localhost.output`
- Current stdout log: `/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/logs/seq_2048.stdout.log`

## Recent Results

```tsv
3	17920	OOM	0	5328	/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/trials/seq_17920/logs/host_0_localhost.output
4	9728	FAIL	0	2375	/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/trials/seq_9728/logs/host_0_localhost.output
5	5632	FAIL	0	1313	/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/trials/seq_5632/logs/host_0_localhost.output
6	3584	FAIL	0	929	/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/trials/seq_3584/logs/host_0_localhost.output
7	2560	FAIL	0	642	/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/trials/seq_2560/logs/host_0_localhost.output
8	2048	FAIL	0	753	/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/trials/seq_2048/logs/host_0_localhost.output
```

## Final Summary Tail

```text
Llama-3.1-8B MemRift activation-only context bisect summary
out_root=/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect
results_tsv=/home/secure/myc/memrift-flagscale/output/llama31_8b_memrift_activation_only_context_bisect/summary/results.tsv
max_runnable_seq_length=<none>
first_failed_seq_length_near_boundary=2048
first_failed_status_near_boundary=FAIL
```

## Error Signals

```text
8:[default0]:/opt/conda/lib/python3.10/site-packages/modelopt/torch/utils/logging.py:115: UserWarning: Failed to import diffusers plugin due to: RuntimeError("Failed to import diffusers.models.modeling_utils because of the following error (look up to see its traceback):\nname 'logger' is not defined"). You may ignore this warning if you do not need this plugin.
18:[default0]:/opt/conda/lib/python3.10/site-packages/modelopt/torch/utils/logging.py:115: UserWarning: Failed to import diffusers plugin due to: RuntimeError("Failed to import diffusers.loaders.single_file_model because of the following error (look up to see its traceback):\nname 'logger' is not defined"). You may ignore this warning if you do not need this plugin.
1067:[default0]:[rank0]: Traceback (most recent call last):
1076:[default0]:[rank0]: RuntimeError: mat2 must be a matrix
1080:[default0]:[rank0]: Traceback (most recent call last):
1104:[default0]:[rank0]:     raise RuntimeError(
1105:[default0]:[rank0]: RuntimeError: All 3 implementation(s) failed for op='generic_gemm'. Last error: mat2 must be a matrix
1127:Traceback (most recent call last):
1139:    raise ChildFailedError(
1140:torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
1142:flagscale/train/train_gpt.py FAILED
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
| 95W / 450W       | 43C          P9 | 859/65536 MiB       | Available            |
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
