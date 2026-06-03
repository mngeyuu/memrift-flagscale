# MemRift 权重 / TE backward 时机冲突修复 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 memrift-flagscale 的权重路径在 Megatron+TE 下 backward 不崩，且 iter0 显存峰值显著低于纯 LoRA。

**Architecture:** 把 backward 期间的权重物化时机从"外部 module hook 抢跑"改为"由 `saved_tensors_hooks._unpack` 在 TE 融合 Function 真正消费权重那一刻当场物化"（零竞速、自愈）；再用"释放滞后 K 层"把驻留权重压到 ≈1 层。分三步递进，每步用集成诊断脚本验证。

**Tech Stack:** PyTorch autograd `saved_tensors_hooks`、Megatron-LM、TransformerEngine、自研 CUDA 扩展 `float_split_stride_pin`、zstandard。

**设计文档:** `docs/superpowers/specs/2026-06-02-memrift-weight-te-backward-design.md`

**验证方式说明:** 本修复是训练期 autograd 集成行为，无法用纯单元测试复现 TE 融合 Function 的 backward 交互。故"测试"采用**端到端集成诊断脚本** `scripts/diag_tinyllama_weight_mem.sh`（TinyLlama-1.1B TP=1），断言日志标记。基线已实测：纯 LoRA iter0 峰值 **5674 MB**；当前 memrift backward **崩溃**。

**Git:** 实验依赖仓库绝对路径与共享 FS，**不使用 worktree**；在原仓库内开分支 `fix/memrift-te-backward`。

**运行约定:** 跑实验前先 `ssh 172.24.178.248` 用 `nvidia-smi` 确认 GPU 空闲；远程命令模板：
`ssh 172.24.178.248 "CUDA_VISIBLE_DEVICES=0 MODE=<memrift|lora> MEMRIFT_HOOK_ORDER=<0|1> MASTER_PORT=<port> bash /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh"`
日志在 `output/diag_tinyllama_<mode>/train.log`。

---

## 文件结构

| 文件 | 职责 | 本计划改动 |
|---|---|---|
| `flagscale/compress/memrift/megatron_dynamic_loader.py` | 权重加载/物化/释放、param 绑定、hook 安装 | 新增 backward 期权重生命周期入口 `unpack_weight_for_backward` + 释放滞后逻辑 + 重物化计数；`release_all_layers` 清理追踪态；`install_hooks` 移除 backward 物化/清除职责 |
| `flagscale/compress/memrift/activation_compression.py` | 激活压缩 + WeightPlaceholder 的 `_pack`/`_unpack` | `_unpack` 的 WeightPlaceholder 分支改为调用 `unpack_weight_for_backward` |
| `scripts/diag_tinyllama_weight_mem.sh` | 集成诊断（已存在） | 不改（验证用） |

---

## Task 0: 准备分支

**Files:**
- 无代码改动

- [ ] **Step 1: 开分支**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git checkout -b fix/memrift-te-backward
git status --short
```

Expected: 在新分支上；工作区含已存在的未跟踪文件 `scripts/diag_tinyllama_weight_mem.sh`、`docs/superpowers/specs/...`、`docs/superpowers/plans/...`（保留）。

- [ ] **Step 2: 提交诊断脚本与设计/计划文档作为基线**

```bash
git add scripts/diag_tinyllama_weight_mem.sh docs/superpowers/specs/2026-06-02-memrift-weight-te-backward-design.md docs/superpowers/plans/2026-06-02-memrift-weight-te-backward-fix.md
git commit -m "docs: memrift TE backward 冲突诊断脚本 + 设计 + 实现计划

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: `_unpack` 按需物化（修崩溃）

让 WeightPlaceholder 在 backward 被消费时，若权重已空就**当场物化**，而非抛错。这是纯加性改动，单独即可消除崩溃。

**Files:**
- Modify: `flagscale/compress/memrift/megatron_dynamic_loader.py`（在 `ensure_group_param_materialized` 之后新增函数）
- Modify: `flagscale/compress/memrift/activation_compression.py:38-50`（`_unpack` 的 WeightPlaceholder 分支）
- Test (集成): `scripts/diag_tinyllama_weight_mem.sh`

- [ ] **Step 1: 写"失败测试"——先确认当前 memrift 必崩（红）**

```bash
ssh 172.24.178.248 "CUDA_VISIBLE_DEVICES=0 MODE=memrift PREFETCH=4 MEMRIFT_HOOK_ORDER=0 MASTER_PORT=29541 bash /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh > /tmp/t1_before.out 2>&1; echo EXIT=$?"
ssh 172.24.178.248 "grep -aE 'WeightPlaceholder: weight not materialized|lm loss' /share/project/mengyc/code/memrift-flagscale/output/diag_tinyllama_memrift/train.log | head"
```

Expected: 出现 `RuntimeError: WeightPlaceholder: weight not materialized ... (layer 21)`，无 `lm loss`（确认 bug 复现）。

- [ ] **Step 2: 在 loader 中新增 `unpack_weight_for_backward`（先只物化 + 计数，不释放）**

在 `megatron_dynamic_loader.py` 中 `ensure_group_param_materialized` 函数定义之后（约第 428 行后）新增：

```python
# Counts on-demand re-materializations triggered from _unpack during backward.
# Reset/logged per iteration by MegatronDynamicLoader.release_all_layers().
_BWD_REMATERIALIZE_COUNT = [0]


def unpack_weight_for_backward(group: "MergedWeightGroup") -> torch.Tensor:
    """Materialize a group's weight on demand from saved_tensors_hooks._unpack.

    Runs INSIDE the TE fused autograd Function's backward, at the exact moment
    the weight is consumed (zero race). If a prior release freed the weight,
    this re-materializes it (self-healing).

    Returns the materialized weight tensor (param.data), or an empty tensor if
    materialization failed (caller raises).
    """
    if group.target_module is None:
        return torch.empty(0)
    param = getattr(group.target_module, group.target_attr, None)
    if param is not None and param.data.numel() > 0:
        return param.data
    # Empty → materialize now (perfectly timed for TE backward consumption).
    _BWD_REMATERIALIZE_COUNT[0] += 1
    weight = ensure_group_param_materialized(group)
    return weight if weight is not None else torch.empty(0)
```

- [ ] **Step 3: 改 `_unpack` 调用它**

`activation_compression.py`，把第 38-50 行的 WeightPlaceholder 分支替换为：

```python
    if isinstance(tok, WeightPlaceholder):
        group = tok.group_ref()
        if group is None or group.target_module is None:
            raise RuntimeError("WeightPlaceholder: MergedWeightGroup was GC'd before backward")
        # On-demand materialize at the exact moment TE's fused backward consumes
        # the weight (zero race; self-heals if a prior release freed it).
        from flagscale.compress.memrift.megatron_dynamic_loader import (
            unpack_weight_for_backward,
        )
        weight = unpack_weight_for_backward(group)
        if weight.numel() == 0:
            raise RuntimeError(
                f"WeightPlaceholder: weight could not be materialized at backward "
                f"time for {group.megatron_target} (layer {group.layer_idx})"
            )
        if tuple(weight.shape) != tok.shape or tuple(weight.stride()) != tok.stride:
            weight = weight.as_strided(tok.shape, tok.stride, 0)
        return weight
```

- [ ] **Step 4: 跑诊断验证不再崩（绿）**

```bash
ssh 172.24.178.248 "CUDA_VISIBLE_DEVICES=0 MODE=memrift PREFETCH=4 MEMRIFT_HOOK_ORDER=0 MASTER_PORT=29542 bash /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh > /tmp/t1_after.out 2>&1; echo EXIT=$?"
ssh 172.24.178.248 "grep -aE 'lm loss|MEM_PROBE\] K|WeightPlaceholder|RuntimeError|CUDA error|illegal' /share/project/mengyc/code/memrift-flagscale/output/diag_tinyllama_memrift/train.log | head -20"
```

Expected: 出现 3 行 `lm loss`（跑完 3 iter）；出现 `[MEM_PROBE] K. peak ...`；无 `WeightPlaceholder`/`RuntimeError`/`illegal`（若出现 `CUDA error: illegal address` 说明存在仅经 ctx 属性消费、未走 saved_tensors 的权重 → 记录并转人工：可能需要 Task 3 的 te_ctx_patch 侧补物化，先停下与维护者确认）。

- [ ] **Step 5: 提交**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add flagscale/compress/memrift/megatron_dynamic_loader.py flagscale/compress/memrift/activation_compression.py
git commit -m "fix(memrift): _unpack 按需物化权重，消除 TE backward 时机崩溃

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: 释放滞后 K 层（修峰值）

Task 1 后不崩，但 `_unpack` 物化的权重会累积到 iteration 末才释放。本步在 backward 逆序中，进入层 L 时释放层号 > L 的已完成层，把驻留压到 ≈1 层。

**Files:**
- Modify: `flagscale/compress/memrift/megatron_dynamic_loader.py`（扩展 `unpack_weight_for_backward`，新增追踪态与释放/清理 helper；`release_all_layers` 末尾清理追踪态并打印重物化计数）

- [ ] **Step 1: 记录基线峰值（红：Task 1 后峰值可能仍偏高）**

```bash
ssh 172.24.178.248 "grep -aE 'MEM_PROBE\] K. peak max_memory_allocated' /share/project/mengyc/code/memrift-flagscale/output/diag_tinyllama_memrift/train.log | tail -1"
```

Expected: 记下当前 memrift 峰值数值（与纯 LoRA 5674 MB 比较；若已显著更低，Task 2 仍补齐有界性保证）。

- [ ] **Step 2: 新增追踪态与 helper**

在 `megatron_dynamic_loader.py` 的 `_BWD_REMATERIALIZE_COUNT = [0]` 之后新增：

```python
# layer_idx -> list[MergedWeightGroup] currently materialized during backward.
_BWD_MATERIALIZED: Dict[int, List["MergedWeightGroup"]] = {}


def _bwd_release_lag() -> int:
    try:
        return max(1, int(os.environ.get("MEMRIFT_BWD_RELEASE_LAG", "1")))
    except Exception:
        return 1


def _clear_param_group(group: "MergedWeightGroup") -> None:
    """Free a group's materialized weight (free-function form of loader._clear_param)."""
    if group.target_module is not None:
        param = getattr(group.target_module, group.target_attr, None)
        if param is not None and param.data.numel() > 0:
            old_ptr = int(param.data_ptr())
            with torch.no_grad():
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            _PTR2GROUP.pop(old_ptr, None)
    for cp in group.components.values():
        cp.release()


def _release_layers_above(threshold: int) -> None:
    """Release all tracked groups whose layer_idx > threshold (already backward'd)."""
    for L in [l for l in list(_BWD_MATERIALIZED.keys()) if l > threshold]:
        for g in _BWD_MATERIALIZED.pop(L, []):
            _clear_param_group(g)


def reset_bwd_tracking() -> int:
    """Clear backward tracking state; return and reset the re-materialize count."""
    _BWD_MATERIALIZED.clear()
    n = _BWD_REMATERIALIZE_COUNT[0]
    _BWD_REMATERIALIZE_COUNT[0] = 0
    return n
```

- [ ] **Step 3: 扩展 `unpack_weight_for_backward` 加入追踪与滞后释放**

把 Task 1 写的 `unpack_weight_for_backward` 函数体替换为：

```python
def unpack_weight_for_backward(group: "MergedWeightGroup") -> torch.Tensor:
    """Materialize a group's weight on demand from saved_tensors_hooks._unpack.

    Runs INSIDE the TE fused autograd Function's backward, at the exact moment
    the weight is consumed (zero race; self-heals if a prior release freed it).
    After materializing, releases layers above (L - lag + 1) to bound resident
    weights to ~`lag` layers during backward.
    """
    if group.target_module is None:
        return torch.empty(0)
    param = getattr(group.target_module, group.target_attr, None)
    if param is not None and param.data.numel() > 0:
        weight = param.data
    else:
        _BWD_REMATERIALIZE_COUNT[0] += 1
        weight = ensure_group_param_materialized(group)
        if weight is None:
            return torch.empty(0)

    L = group.layer_idx
    if L >= 0:
        bucket = _BWD_MATERIALIZED.setdefault(L, [])
        if group not in bucket:
            bucket.append(group)
        _release_layers_above(L - (_bwd_release_lag() - 1))
    return weight
```

- [ ] **Step 4: `release_all_layers` 末尾清理追踪态并打印重物化计数**

在 `MegatronDynamicLoader.release_all_layers`（约第 1037 行）方法体末尾追加：

```python
        n_remat = reset_bwd_tracking()
        if self.print_debug:
            print(f"[MemRift] bwd re-materialize count this iter: {n_remat}", flush=True)
```

- [ ] **Step 5: 跑诊断验证峰值下降且重物化可控（绿）**

```bash
ssh 172.24.178.248 "CUDA_VISIBLE_DEVICES=0 MODE=memrift PREFETCH=4 MEMRIFT_HOOK_ORDER=0 MASTER_PORT=29543 bash /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh > /tmp/t2.out 2>&1; echo EXIT=$?"
ssh 172.24.178.248 "grep -aE 'lm loss|MEM_PROBE\] K. peak|re-materialize count|illegal' /share/project/mengyc/code/memrift-flagscale/output/diag_tinyllama_memrift/train.log | head -20"
```

Expected:
- 3 行 `lm loss`，无崩溃/illegal。
- `[MEM_PROBE] K. peak max_memory_allocated` **显著 < 5674 MB**（纯 LoRA 基线）。
- `re-materialize count this iter`（需 `--memrift-print-debug`，已在脚本 memrift 分支开启）数量级合理（≈ 层数×组数或更低；若远超，记为 `K` 调参依据）。

- [ ] **Step 6: 提交**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add flagscale/compress/memrift/megatron_dynamic_loader.py
git commit -m "fix(memrift): backward 释放滞后 K 层，权重驻留有界，峰值低于纯 LoRA

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 移除制造竞速的 backward 物化/清除 hook

`install_hooks` 里 backward 的 module 级"materialize→clear"和 per-linear 兜底是冲突来源；现在权重生命周期已由 `_unpack` 驱动，应移除其物化/清除职责，仅保留无害的异步预取与 empty_cache。

**Files:**
- Modify: `flagscale/compress/memrift/megatron_dynamic_loader.py` `install_hooks`（backward 段，约第 1310-1427 行）

- [ ] **Step 1: `make_bwd_pre` 只保留预取，移除当前层 materialize**

在 `make_bwd_pre._hook` 中，删除"2) Materialize current layer"那段对 `cur_groups` 的 `self._materialize_group(...)` + `self._set_param(...)` 循环（约第 1332-1344 行），**保留**前面对 `prv_groups_list` 的 `materialize_async` 预取提交（约第 1323-1330 行）。改后该 hook 仅提交上一层异步预取。

- [ ] **Step 2: `make_bwd_post` 移除 `_clear_param`，仅保留 empty_cache 计数**

把 `make_bwd_post._hook` 体改为：

```python
                def _hook(mod, grad_in, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        print(f"[HOOK_ORDER] bwd_post layer={cur_name}", flush=True)
                    # 不再在此清除权重：生命周期由 _unpack 的释放滞后逻辑驱动。
                    self._bwd_counter += 1
                    if self._bwd_counter % self._bwd_empty_step == 0:
                        torch.cuda.empty_cache()
```

- [ ] **Step 3: per-linear backward hook 默认关闭**

把第 1422 行的注册条件默认改为关闭（默认 `"1"`=禁用，仅显式设 `MEMRIFT_DISABLE_LINEAR_BWD_PRE=0` 才注册）：

```python
            if os.environ.get("MEMRIFT_DISABLE_LINEAR_BWD_PRE", "1") != "1":
                lpre = make_linear_bwd_pre_all(cur_groups)
                lpost = make_linear_bwd_post_all(cur_groups)
                for tm, _g in mod2group.values():
                    tm.register_full_backward_pre_hook(lpre)
                    tm.register_full_backward_hook(lpost)
```

- [ ] **Step 4: 跑诊断验证仍正确且峰值不退化（绿）**

```bash
ssh 172.24.178.248 "CUDA_VISIBLE_DEVICES=0 MODE=memrift PREFETCH=4 MEMRIFT_HOOK_ORDER=1 MASTER_PORT=29544 bash /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh > /tmp/t3.out 2>&1; echo EXIT=$?"
ssh 172.24.178.248 "L=/share/project/mengyc/code/memrift-flagscale/output/diag_tinyllama_memrift/train.log; grep -aE 'lm loss|MEM_PROBE\] K. peak|re-materialize count|illegal|RuntimeError' \$L | head; echo '--- HOOK_ORDER 尾 ---'; grep -a 'HOOK_ORDER' \$L | tail -8"
```

Expected:
- 3 行 `lm loss`，无崩溃/illegal。
- `K. peak` 与 Task 2 相当或更低。
- `HOOK_ORDER` 不再出现 `lin_bwd_pre`（已默认关闭）；`bwd_post` 不再伴随权重清除导致的 `empty=True` 失败路径。
- 重物化计数与 Task 2 相当（未因移除 hook 而暴增）。

- [ ] **Step 5: 提交**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add flagscale/compress/memrift/megatron_dynamic_loader.py
git commit -m "refactor(memrift): 移除 backward module/per-linear 物化清除，权重生命周期归一到 _unpack

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 端到端对照与收尾

**Files:**
- 无代码改动（汇总验证）

- [ ] **Step 1: 重跑纯 LoRA 基线（确保对照同环境）**

```bash
ssh 172.24.178.248 "CUDA_VISIBLE_DEVICES=0 MODE=lora MEMRIFT_HOOK_ORDER=0 MASTER_PORT=29545 bash /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh > /tmp/t4_lora.out 2>&1; echo EXIT=$?"
ssh 172.24.178.248 "grep -aE 'MEM_PROBE\] K. peak max_memory_allocated' /share/project/mengyc/code/memrift-flagscale/output/diag_tinyllama_lora/train.log | tail -1"
```

Expected: 纯 LoRA 峰值 ≈ 5674 MB（复现基线）。

- [ ] **Step 2: 汇总对照表**

记录三档峰值（纯 LoRA / memrift Task1后 / memrift Task3后）与重物化计数，确认 memrift 峰值 < 纯 LoRA。若 memrift 峰值未低于 LoRA，停下回到设计文档第 5.2.2 评估 `K` 或 `sm` 常驻（非目标项）。

- [ ] **Step 3: （可选，用户确认后）大模型复测**

经用户同意后，用 Mistral-7B / Llama-3.1-8B 复测 backward 不崩与峰值，确认大模型上的累积症状消除。

- [ ] **Step 4: 完成开发分支**

按需走 `superpowers:finishing-a-development-branch`（合并 / PR / 清理），由用户决定。

---

## 自检（spec 覆盖）

- 设计 5.1 按需物化 → Task 1。
- 设计 5.2 + 5.2.1 自愈 + 5.2.2 释放窗口 `K`（`MEMRIFT_BWD_RELEASE_LAG`）+ 重物化计数 → Task 2。
- 设计 5.3 移除竞速 hook（含 per-linear 默认关）→ Task 3。
- 设计 5.4 保留 te_ctx_patch / forward hook / sm 常驻 → 不改，计划未触碰。
- 设计 8 验证（不崩 / 峰值 < LoRA / H_state / HOOK_ORDER / 重物化计数）→ Task 1-4 的验证步。
- 函数命名一致性：`unpack_weight_for_backward`、`_release_layers_above`、`_clear_param_group`、`reset_bwd_tracking`、`_BWD_MATERIALIZED`、`_BWD_REMATERIALIZE_COUNT`、`_bwd_release_lag` 全程一致。
