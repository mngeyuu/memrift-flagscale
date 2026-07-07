# MemRift Benchmark Accuracy Design

## Goal

Replace the current final-training-loss accuracy proxy with downstream benchmark
accuracy comparisons on GSM8K CoT and HellaSwag. Compare MemRift+LoRA against
pure LoRA for each task and require the relative accuracy drop to be no greater
than 1%.

The comparison is defined as:

```text
relative_accuracy_drop =
    (pure_lora_accuracy - memrift_accuracy) / pure_lora_accuracy
```

A task passes when `relative_accuracy_drop <= 0.01`. For example, if pure LoRA
accuracy is 45%, the maximum permitted absolute decrease is 0.45 percentage
points and MemRift must score at least 44.55%.

## Scope

The change applies to both metric suites:

- `scripts/metrics/llama8b/accuracy_loss.sh`
- `scripts/metrics/aquila/accuracy_loss.sh`

It does not change the compression-ratio, context-length, or load-time metrics.
The existing training comparison remains pure LoRA versus MemRift weight and
activation compression with asynchronous operation enabled.

## Architecture

Add a shared benchmark evaluation module under `scripts/metrics/`. It will reuse
the installed lm-eval task definitions and document processing for GSM8K CoT and
HellaSwag while executing model requests through FlagScale's Megatron model
stack. This preserves standard benchmark prompts and scoring without bypassing
MemRift's dynamic compressed-weight and activation-compression paths.

The evaluator will support the two request types needed here:

- Greedy generation for GSM8K CoT, followed by the task's standard final-answer
  extraction and exact-match scoring.
- Conditional log-likelihood for all four HellaSwag endings, including the
  task's standard length normalization and accuracy calculation.

Both training variants must save their final Megatron LoRA checkpoint. Each
checkpoint is evaluated in a separate process so model state, CUDA allocations,
and MemRift hooks cannot leak between variants.

## Execution Flow

Each model-specific `accuracy_loss.sh` performs these steps:

1. Ensure required MemRift compressed weights exist.
2. Train pure LoRA and save its final checkpoint.
3. Train MemRift+LoRA and save its final checkpoint.
4. Evaluate the pure LoRA checkpoint on GSM8K CoT and HellaSwag.
5. Evaluate the MemRift checkpoint on the same deterministic examples.
6. Aggregate task results, compute absolute and relative drops, and write the
   final acceptance result.

Evaluation uses deterministic ordering and greedy decoding. The same sample
limits and task examples are used for both variants. The scripts default to the
complete GSM8K test split and HellaSwag validation split. Environment variables
`GSM8K_LIMIT` and `HELLASWAG_LIMIT` may restrict sample counts for smoke runs;
unset or empty values mean full evaluation.

The first end-to-end smoke run uses 32 GSM8K examples and 64 HellaSwag examples.

## Outputs

Variant-level benchmark results are written beneath the existing metric output
directory. Each task result records:

- task and split
- requested limit and evaluated sample count
- number correct
- accuracy as a fraction in `[0, 1]`
- scoring mode and benchmark metadata
- per-sample prediction details sufficient to diagnose mismatches

The final `result.json` keeps `metric`, model identity, variant summaries, and a
per-task comparison containing:

- `pure_lora_accuracy`
- `memrift_accuracy`
- `absolute_drop_percentage_points`
- `relative_accuracy_drop`
- `relative_accuracy_drop_percent`
- `threshold_relative_accuracy_drop_percent` set to `1.0`
- `pass`

The top-level `pass` is true only when both GSM8K CoT and HellaSwag pass.
Negative drops are retained when MemRift is more accurate and satisfy the upper
bound.

## Failure Handling

Evaluation fails explicitly when a checkpoint or compressed-weight index is
missing, a task dataset cannot be loaded, no samples are evaluated, model output
cannot be scored, or the evaluator process exits unsuccessfully. A pure LoRA
accuracy of zero makes relative degradation undefined and therefore fails that
task. Missing or invalid data must never produce a passing result.

The shell entrypoint remains fail-fast for training and evaluator failures. When
both valid task result files exist, aggregation always writes the final JSON and
returns nonzero if the acceptance threshold is not met.

## Testing

Unit tests will be written before implementation for:

- relative and absolute accuracy-drop calculations
- the 1% relative threshold boundary
- negative drops when MemRift improves accuracy
- zero baseline accuracy and malformed/missing task results
- the requirement that both tasks pass
- deterministic sample-limit propagation and result schema

Task adapter tests will use small synthetic requests and model outputs rather
than loading a GPU model. Shell syntax and a dry-run invocation will verify the
entrypoint wiring. The final verification sequence is:

1. Run focused unit tests locally.
2. Check shell syntax for both model scripts.
3. Check GPU availability on host `172.24.178.248`.
4. Run the 32/64 sample end-to-end smoke evaluation in conda environment
   `myc-fl312` on an idle GPU.
5. Preserve result JSON and logs under `output/metrics/<model>/accuracy_loss/`.

Full-dataset runs use the same command with the sample limits unset.
