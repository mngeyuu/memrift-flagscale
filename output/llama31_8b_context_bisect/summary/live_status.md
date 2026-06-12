# Llama-3.1-8B MemRift Context Bisect Status

- Snapshot time: `2026-06-03T14:12:36+08:00`
- Detached launcher: pid=77193 (exited)
- Current trial seq_length: `9728`
- Latest completed trial: seq=`10496`, status=`OOM`
- Results TSV: `/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/summary/results.tsv`
- Final result file: `/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/summary/final_result.txt`
- Current host log: `/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/trials/seq_9728/logs/host_0_localhost.output`
- Current stdout log: `/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/logs/seq_9728.stdout.log`

## Recent Results

```tsv
4	9728	PASS	0	128	/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/trials/seq_9728/logs/host_0_localhost.output
5	13824	OOM	0	107	/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/trials/seq_13824/logs/host_0_localhost.output
6	11776	OOM	0	87	/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/trials/seq_11776/logs/host_0_localhost.output
7	10752	OOM	0	86	/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/trials/seq_10752/logs/host_0_localhost.output
8	10240	PASS	0	125	/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/trials/seq_10240/logs/host_0_localhost.output
9	10496	OOM	0	85	/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/trials/seq_10496/logs/host_0_localhost.output
```

## Final Summary Tail

```text
Llama-3.1-8B MemRift context bisect summary
out_root=/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect
results_tsv=/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/summary/results.tsv
max_runnable_seq_length=10240
best_pass_host_log=/home/secure/myc/memrift-flagscale/output/llama31_8b_context_bisect/trials/seq_10240/logs/host_0_localhost.output
first_failed_seq_length_near_boundary=10496
first_failed_status_near_boundary=OOM
```

## Error Signals

```text
8:[default0]:/opt/conda/lib/python3.10/site-packages/modelopt/torch/utils/logging.py:115: UserWarning: Failed to import diffusers plugin due to: RuntimeError("Failed to import diffusers.models.modeling_utils because of the following error (look up to see its traceback):\nname 'logger' is not defined"). You may ignore this warning if you do not need this plugin.
18:[default0]:/opt/conda/lib/python3.10/site-packages/modelopt/torch/utils/logging.py:115: UserWarning: Failed to import diffusers plugin due to: RuntimeError("Failed to import diffusers.loaders.single_file_model because of the following error (look up to see its traceback):\nname 'logger' is not defined"). You may ignore this warning if you do not need this plugin.
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
