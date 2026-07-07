# MemRift Benchmark Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the final-loss proxy with GSM8K CoT and HellaSwag accuracy comparisons whose relative degradation against pure LoRA is at most 1%.

**Architecture:** A pure-Python result module owns validation, relative-drop arithmetic, and JSON aggregation. A benchmark runner adapts lm-eval task documents into generation and log-likelihood requests, while a Megatron backend loads each saved LoRA checkpoint in either pure or MemRift mode and executes those requests in an isolated process. Model-specific shell scripts train, save, evaluate, and aggregate both variants.

**Tech Stack:** Python 3.10, pytest, lm-eval-harness, Hugging Face datasets, PyTorch, FlagScale/Megatron-Core, Bash, Hydra.

## Global Constraints

- Compute `relative_accuracy_drop = (pure_lora_accuracy - memrift_accuracy) / pure_lora_accuracy`.
- GSM8K CoT and HellaSwag each pass only when `relative_accuracy_drop <= 0.01`.
- A zero pure-LoRA accuracy, missing result, invalid schema, or zero evaluated samples must fail.
- Default to full GSM8K test and HellaSwag validation splits; `GSM8K_LIMIT` and `HELLASWAG_LIMIT` restrict smoke runs.
- Run model evaluation through FlagScale/Megatron so MemRift weight and activation compression remain active.
- Do not modify or revert unrelated dirty-worktree changes.

---

### Task 1: Accuracy Result Arithmetic And Schema

**Files:**
- Create: `scripts/metrics/benchmark_accuracy_result.py`
- Create: `tests/unit_tests/metrics/test_benchmark_accuracy_result.py`

**Interfaces:**
- Produces: `compare_task(task: str, baseline: dict, candidate: dict, threshold: float = 0.01) -> dict`
- Produces: `aggregate_results(model_key: str, model_name: str, baseline: dict, candidate: dict, threshold: float = 0.01) -> dict`
- Produces CLI: `python scripts/metrics/benchmark_accuracy_result.py --baseline FILE --candidate FILE --output FILE --model-key KEY --model-name NAME`

- [ ] **Step 1: Write failing arithmetic and validation tests**

```python
from scripts.metrics.benchmark_accuracy_result import aggregate_results, compare_task


def task_result(task, accuracy, samples=100):
    return {"task": task, "accuracy": accuracy, "num_samples": samples, "num_correct": round(accuracy * samples)}


def test_relative_drop_uses_baseline_accuracy():
    result = compare_task("gsm8k_cot", task_result("gsm8k_cot", 0.45), task_result("gsm8k_cot", 0.4455))
    assert result["absolute_drop_percentage_points"] == pytest.approx(0.45)
    assert result["relative_accuracy_drop_percent"] == pytest.approx(1.0)
    assert result["pass"] is True


def test_both_tasks_must_pass():
    baseline = {"tasks": {"gsm8k_cot": task_result("gsm8k_cot", 0.5), "hellaswag": task_result("hellaswag", 0.5)}}
    candidate = {"tasks": {"gsm8k_cot": task_result("gsm8k_cot", 0.5), "hellaswag": task_result("hellaswag", 0.49)}}
    assert aggregate_results("m", "Model", baseline, candidate)["pass"] is False
```

Add separate tests for a negative drop, a zero baseline, zero samples, mismatched task names, and a missing task.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/unit_tests/metrics/test_benchmark_accuracy_result.py`

Expected: collection fails with `ModuleNotFoundError: scripts.metrics.benchmark_accuracy_result`.

- [ ] **Step 3: Implement validated comparison and CLI**

Implement finite-number validation, exact task-set validation for `gsm8k_cot` and `hellaswag`, the three drop fields, per-task pass values, top-level `pass = all(...)`, and atomic JSON output through a temporary sibling file followed by `Path.replace`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/unit_tests/metrics/test_benchmark_accuracy_result.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/metrics/benchmark_accuracy_result.py tests/unit_tests/metrics/test_benchmark_accuracy_result.py
git commit -m "feat: aggregate benchmark accuracy degradation"
```

### Task 2: Standard Task Document Adapter

**Files:**
- Create: `scripts/metrics/benchmark_tasks.py`
- Create: `tests/unit_tests/metrics/test_benchmark_tasks.py`

**Interfaces:**
- Produces: `load_task_examples(task_name: str, limit: int | None) -> list[BenchmarkExample]`
- Produces: `score_generation(example: BenchmarkExample, generated: str) -> SampleScore`
- Produces: `score_choices(example: BenchmarkExample, loglikelihoods: list[float]) -> SampleScore`
- `BenchmarkExample` includes stable `sample_id`, prompt/context, expected answer, and optional choices.

- [ ] **Step 1: Write failing adapter tests**

Use monkeypatched lm-eval task objects to verify deterministic first-N selection, GSM8K final-answer normalization (commas and trailing text), and HellaSwag selection from four normalized log-likelihoods. Assert invalid limits and wrong choice counts raise `ValueError`.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/unit_tests/metrics/test_benchmark_tasks.py`

Expected: collection fails because `scripts.metrics.benchmark_tasks` does not exist.

- [ ] **Step 3: Implement the adapter against installed lm-eval APIs**

Use `lm_eval.tasks.TaskManager` to resolve `gsm8k_cot` and `hellaswag`, task-owned document-to-text/target methods, and task-owned result processing where available. Preserve the harness task version/config in metadata. Do not duplicate dataset prompts in this module.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/unit_tests/metrics/test_benchmark_tasks.py`

Expected: all tests pass without downloading datasets.

- [ ] **Step 5: Commit**

```bash
git add scripts/metrics/benchmark_tasks.py tests/unit_tests/metrics/test_benchmark_tasks.py
git commit -m "feat: adapt lm-eval benchmark tasks"
```

### Task 3: Megatron Benchmark Backend And Runner

**Files:**
- Create: `scripts/metrics/megatron_benchmark_backend.py`
- Create: `scripts/metrics/benchmark_accuracy.py`
- Create: `tests/unit_tests/metrics/test_benchmark_accuracy.py`

**Interfaces:**
- Produces: `MegatronBenchmarkBackend.generate(prompts: list[str], max_new_tokens: int) -> list[str]`
- Produces: `MegatronBenchmarkBackend.loglikelihood(requests: list[tuple[str, str]]) -> list[float]`
- CLI inputs include config name, checkpoint directory, mode (`lora` or `memrift_async`), compressed weight directory, task limits, and output path.
- CLI output schema: `{"variant": str, "tasks": {task_name: task_result}, "metadata": dict}`.

- [ ] **Step 1: Write failing runner tests with a fake backend**

Verify that GSM8K requests call `generate`, HellaSwag emits four log-likelihood requests per sample, both tasks record `num_samples`, `num_correct`, and `accuracy`, details preserve sample IDs, and a backend output-count mismatch raises `RuntimeError`.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/unit_tests/metrics/test_benchmark_accuracy.py`

Expected: collection fails because `scripts.metrics.benchmark_accuracy` does not exist.

- [ ] **Step 3: Implement the runner with backend injection**

Keep orchestration importable without initializing CUDA. Instantiate the real backend only inside `main()`. Write one result only after both tasks complete; include task limits, checkpoint, mode, model identity, and lm-eval metadata.

- [ ] **Step 4: Implement Megatron model loading and inference**

Reuse the initialization/model-provider path from `flagscale/train/megatron/train_gpt.py` and generation utilities already used by `flagscale/train/generate_gpt.py`. Load the saved LoRA checkpoint with optimizer/RNG disabled. In `memrift_async` mode set the same MemRift flags used by `run_yaml_train`; in `lora` mode disable every MemRift flag. Compute continuation log-likelihood from token-level cross entropy over continuation tokens only, and return summed plus length-normalized values needed by HellaSwag.

- [ ] **Step 5: Run focused tests and import smoke**

Run: `pytest -q tests/unit_tests/metrics/test_benchmark_accuracy.py`

Run: `python -c 'from scripts.metrics.benchmark_accuracy import main; print("ok")'`

Expected: tests pass and import prints `ok` without CUDA initialization.

- [ ] **Step 6: Commit**

```bash
git add scripts/metrics/megatron_benchmark_backend.py scripts/metrics/benchmark_accuracy.py tests/unit_tests/metrics/test_benchmark_accuracy.py
git commit -m "feat: evaluate benchmarks through Megatron"
```

### Task 4: Training Checkpoints And Accuracy Entrypoints

**Files:**
- Modify: `scripts/metrics/common.sh`
- Modify: `scripts/metrics/llama8b/accuracy_loss.sh`
- Modify: `scripts/metrics/aquila/accuracy_loss.sh`
- Create: `tests/unit_tests/metrics/test_accuracy_loss_scripts.py`

**Interfaces:**
- `run_yaml_train` accepts an optional save directory while preserving existing callers.
- `run_benchmark_accuracy variant mode checkpoint output` centralizes evaluator invocation.
- Environment: `GSM8K_LIMIT`, `HELLASWAG_LIMIT`, and `ACCURACY_THRESHOLD` (default `0.01`).

- [ ] **Step 1: Write failing static shell-wiring tests**

Parse both scripts as text and assert they no longer reference `final_lm_loss`, invoke benchmark evaluation for both variants, invoke the shared aggregator, and pass both limit variables. Assert `common.sh` sets checkpoint save path and final-iteration save interval for accuracy runs.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/unit_tests/metrics/test_accuracy_loss_scripts.py`

Expected: assertions fail because current scripts still compare final loss.

- [ ] **Step 3: Extend common helpers and replace both entrypoints**

Add save arguments only when a save directory is provided: checkpoint path, `save_interval=$train_iters`, and final save enabled. Have each accuracy script create isolated checkpoint/evaluation directories, call both training modes, call both evaluator processes with identical limits, aggregate JSON, and exit nonzero when top-level `pass` is false.

- [ ] **Step 4: Verify tests and shell syntax**

Run: `pytest -q tests/unit_tests/metrics/test_accuracy_loss_scripts.py`

Run: `bash -n scripts/metrics/common.sh scripts/metrics/llama8b/accuracy_loss.sh scripts/metrics/aquila/accuracy_loss.sh`

Expected: tests pass and `bash -n` exits zero.

- [ ] **Step 5: Commit**

```bash
git add scripts/metrics/common.sh scripts/metrics/llama8b/accuracy_loss.sh scripts/metrics/aquila/accuracy_loss.sh tests/unit_tests/metrics/test_accuracy_loss_scripts.py
git commit -m "feat: benchmark MemRift accuracy loss"
```

### Task 5: Documentation And End-To-End Verification On 172

**Files:**
- Modify: `scripts/metrics/README.md`
- Modify: `examples/memrift/README.md`
- Modify: `docs/MemRift_第三方验收测试手册.md`

**Interfaces:**
- Documents smoke command and full-run command, result fields, relative threshold, and output paths.

- [ ] **Step 1: Update user-facing metric documentation**

Replace final-loss language with GSM8K CoT/HellaSwag relative accuracy degradation. Document:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> GSM8K_LIMIT=32 HELLASWAG_LIMIT=64 \
  bash scripts/metrics/llama8b/accuracy_loss.sh
```

and state that unsetting both limits runs full splits.

- [ ] **Step 2: Run the complete local verification suite**

Run: `pytest -q tests/unit_tests/metrics`

Run: `bash -n scripts/metrics/common.sh scripts/metrics/llama8b/accuracy_loss.sh scripts/metrics/aquila/accuracy_loss.sh`

Run: `git diff --check`

Expected: all commands exit zero.

- [ ] **Step 3: Check GPU availability on 172**

Run:

```bash
/root/.claude/skills/run-on-172/run172.sh \
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

Expected: identify an idle GPU. If every GPU is busy, do not launch the job.

- [ ] **Step 4: Run 32/64 smoke on an idle GPU**

Run from `/share/project/mengyc/code/memrift-flagscale` on 172:

```bash
CUDA_VISIBLE_DEVICES=<idle-gpu> GSM8K_LIMIT=32 HELLASWAG_LIMIT=64 \
OUT_ROOT=output/metrics/llama8b_benchmark_smoke \
bash scripts/metrics/llama8b/accuracy_loss.sh
```

Expected: both variants train and save, both task result files contain nonzero sample counts, and `accuracy_loss/result.json` contains valid per-task relative drops. A threshold failure is a valid benchmark outcome, but infrastructure/runtime failure is not.

- [ ] **Step 5: Inspect results and logs**

Run:

```bash
python -m json.tool output/metrics/llama8b_benchmark_smoke/accuracy_loss/result.json
rg -n "Traceback|CUDA out of memory|FAILED|error:" output/metrics/llama8b_benchmark_smoke/accuracy_loss
```

Expected: valid JSON and no runtime-failure matches. Report the two baseline accuracies, MemRift accuracies, relative drops, and pass values.

- [ ] **Step 6: Commit documentation**

```bash
git add scripts/metrics/README.md examples/memrift/README.md docs/MemRift_第三方验收测试手册.md
git commit -m "docs: describe benchmark accuracy acceptance"
```
