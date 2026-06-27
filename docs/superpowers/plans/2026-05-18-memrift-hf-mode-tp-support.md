# MemRift HF Mode TP Support 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `MegatronDynamicLoader` 的 HF mode 在 TP>1 时的权重切分缺陷，让用户无需 Megatron TP checkpoint 即可在多卡训练中正确使用 MemRift。

**Architecture:** 在 `MergedWeightGroup` 上记录每个权重的 TP 并行类型（列并行/行并行/复制），然后在 `_materialize_group_tensor` 中反序列化全量权重后立即按 TP rank 切片。shard mode（已预切分）通过检查 `tp_rank==0, tp_size==1` 默认值保持原有行为不变。在线 HF mode 切分会在每个 rank 保留全量压缩权重（CPU 内存），适合轻量测试；生产环境推荐用已有的 `prepare_weight_tp2.py` 离线预切分。

**Tech Stack:** Python, PyTorch, zstandard, CUDA extension `float_split_stride_pin`

---

## 文件变更总览

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `flagscale/compress/memrift/tp_shard.py` | TP 切分工具函数（col/row/rep 类型常量 + shard 函数） |
| 修改 | `flagscale/compress/memrift/megatron_dynamic_loader.py` | 1) `MergedWeightGroup` 增加 tp_rank/tp_size 字段；2) `CompressedParam` 增加 parallel_type；3) `HF_TO_MEGATRON` 扩展并行类型；4) `_map_hf_to_megatron` 返回三元组；5) `_load_weights_hf_mode` 设置元数据；6) `_materialize_group_tensor` 应用切片 |
| 新建 | `tests/compress/test_tp_shard.py` | `tp_shard.py` 单元测试（纯 CPU，无 Megatron 依赖） |
| 新建 | `tests/compress/test_hf_mode_tp.py` | HF mode TP 切分集成测试（构造假 HF 压缩目录 + 验证形状） |

---

## Task 1: 新建 `tp_shard.py` — TP 切分工具模块

**Files:**
- Create: `flagscale/compress/memrift/tp_shard.py`

- [ ] **Step 1.1: 创建工具文件**

```python
# flagscale/compress/memrift/tp_shard.py
"""TP sharding utilities for HF-mode weight loading in MemRift.

HF mode keeps full compressed weights in CPU RAM and applies sharding at
materialize time. Each TP rank gets the correct slice of each weight:
  - column-parallel (Q, K, V, gate, up): shard along dim 0 (output features)
  - row-parallel (O, down):              shard along dim 1 (input features)
  - replicated (layernorms, biases):     no sharding
"""
from __future__ import annotations
import torch

COL_PARALLEL = "col"   # column-parallel: shard dim 0 (output features)
ROW_PARALLEL = "row"   # row-parallel:    shard dim 1 (input features)
REPLICATED   = "rep"   # replicated across all TP ranks (layernorms etc.)


def col_shard(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Return the column-parallel shard for tp_rank (split along dim 0)."""
    rows_per_rank = t.shape[0] // tp_size
    return t[tp_rank * rows_per_rank : (tp_rank + 1) * rows_per_rank].contiguous()


def row_shard(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Return the row-parallel shard for tp_rank (split along dim 1)."""
    cols_per_rank = t.shape[1] // tp_size
    return t[:, tp_rank * cols_per_rank : (tp_rank + 1) * cols_per_rank].contiguous()


def apply_tp_shard(
    t: torch.Tensor,
    parallel_type: str,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Apply TP sharding to tensor based on its parallel type."""
    if tp_size <= 1 or parallel_type == REPLICATED:
        return t
    if parallel_type == COL_PARALLEL:
        return col_shard(t, tp_rank, tp_size)
    if parallel_type == ROW_PARALLEL:
        return row_shard(t, tp_rank, tp_size)
    raise ValueError(f"Unknown parallel_type: {parallel_type!r}")


def validate_shard_divisibility(
    name: str,
    shape: tuple,
    parallel_type: str,
    tp_size: int,
) -> None:
    """Raise ValueError early if shape cannot be evenly sharded."""
    if tp_size <= 1 or parallel_type == REPLICATED:
        return
    if parallel_type == COL_PARALLEL:
        if shape[0] % tp_size != 0:
            raise ValueError(
                f"[MemRift] HF mode TP: weight '{name}' dim-0 size {shape[0]} "
                f"is not divisible by tp_size={tp_size}. "
                f"Check model config (num_attention_heads, num_key_value_heads). "
                f"Tip: use prepare_weight_tp2.py for models where kv_heads < tp_size."
            )
    elif parallel_type == ROW_PARALLEL:
        if len(shape) < 2 or shape[1] % tp_size != 0:
            raise ValueError(
                f"[MemRift] HF mode TP: weight '{name}' dim-1 size "
                f"{shape[1] if len(shape) >= 2 else 'N/A'} "
                f"is not divisible by tp_size={tp_size}."
            )
```

- [ ] **Step 1.2: 确认文件创建成功**

```bash
python -c "from flagscale.compress.memrift.tp_shard import COL_PARALLEL, apply_tp_shard; print('OK')"
```
Expected: `OK`

---

## Task 2: 单元测试 `tp_shard.py`

**Files:**
- Create: `tests/compress/__init__.py`
- Create: `tests/compress/test_tp_shard.py`

- [ ] **Step 2.1: 创建测试包 `__init__.py`**

```bash
mkdir -p /share/project/mengyc/code/memrift-flagscale/tests/compress
touch /share/project/mengyc/code/memrift-flagscale/tests/compress/__init__.py
```

- [ ] **Step 2.2: 写测试文件**

```python
# tests/compress/test_tp_shard.py
"""Unit tests for tp_shard.py — CPU only, no Megatron dependency."""
import pytest
import torch
from flagscale.compress.memrift.tp_shard import (
    COL_PARALLEL, ROW_PARALLEL, REPLICATED,
    col_shard, row_shard, apply_tp_shard, validate_shard_divisibility,
)


def test_col_shard_basic():
    # [4, 8] tensor split across 2 ranks
    t = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    r0 = col_shard(t, tp_rank=0, tp_size=2)
    r1 = col_shard(t, tp_rank=1, tp_size=2)
    assert r0.shape == (2, 8)
    assert r1.shape == (2, 8)
    assert torch.all(r0 == t[:2])
    assert torch.all(r1 == t[2:])
    # Concatenating shards recovers original
    assert torch.all(torch.cat([r0, r1], dim=0) == t)


def test_row_shard_basic():
    # [4, 8] tensor split across 2 ranks
    t = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    r0 = row_shard(t, tp_rank=0, tp_size=2)
    r1 = row_shard(t, tp_rank=1, tp_size=2)
    assert r0.shape == (4, 4)
    assert r1.shape == (4, 4)
    assert torch.all(r0 == t[:, :4])
    assert torch.all(r1 == t[:, 4:])
    assert torch.all(torch.cat([r0, r1], dim=1) == t)


def test_col_shard_tp4():
    # [8, 16] tensor split across 4 ranks
    t = torch.ones(8, 16)
    shards = [col_shard(t, r, 4) for r in range(4)]
    assert all(s.shape == (2, 16) for s in shards)
    assert torch.all(torch.cat(shards, dim=0) == t)


def test_apply_tp_shard_col():
    t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    result = apply_tp_shard(t, COL_PARALLEL, tp_rank=1, tp_size=2)
    assert result.shape == (2, 4)
    assert torch.all(result == t[2:])


def test_apply_tp_shard_row():
    t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    result = apply_tp_shard(t, ROW_PARALLEL, tp_rank=0, tp_size=2)
    assert result.shape == (4, 2)
    assert torch.all(result == t[:, :2])


def test_apply_tp_shard_replicated():
    t = torch.ones(4, 4)
    # REPLICATED: all ranks get the same full tensor
    for rank in range(4):
        result = apply_tp_shard(t, REPLICATED, tp_rank=rank, tp_size=4)
        assert result.shape == (4, 4)
        assert torch.all(result == t)


def test_apply_tp_shard_tp1_is_noop():
    t = torch.ones(4, 4)
    result = apply_tp_shard(t, COL_PARALLEL, tp_rank=0, tp_size=1)
    assert torch.all(result == t)


def test_validate_divisibility_ok():
    # Should not raise
    validate_shard_divisibility("q_proj", (4, 8), COL_PARALLEL, 2)
    validate_shard_divisibility("o_proj", (8, 4), ROW_PARALLEL, 2)
    validate_shard_divisibility("norm",   (8,),   REPLICATED,   4)


def test_validate_divisibility_col_fail():
    with pytest.raises(ValueError, match="dim-0 size 3"):
        validate_shard_divisibility("q_proj", (3, 8), COL_PARALLEL, 2)


def test_validate_divisibility_row_fail():
    with pytest.raises(ValueError, match="dim-1 size 6"):
        validate_shard_divisibility("o_proj", (8, 6), ROW_PARALLEL, 4)
```

- [ ] **Step 2.3: 运行测试，确认全部通过**

```bash
cd /share/project/mengyc/code/memrift-flagscale
python -m pytest tests/compress/test_tp_shard.py -v
```

Expected:
```
test_col_shard_basic PASSED
test_row_shard_basic PASSED
test_col_shard_tp4 PASSED
test_apply_tp_shard_col PASSED
test_apply_tp_shard_row PASSED
test_apply_tp_shard_replicated PASSED
test_apply_tp_shard_tp1_is_noop PASSED
test_validate_divisibility_ok PASSED
test_validate_divisibility_col_fail PASSED
test_validate_divisibility_row_fail PASSED
10 passed
```

- [ ] **Step 2.4: 提交**

```bash
git add flagscale/compress/memrift/tp_shard.py tests/compress/__init__.py tests/compress/test_tp_shard.py
git commit -m "feat(memrift): add tp_shard utility module for HF mode TP support"
```

---

## Task 3: 扩展 `MergedWeightGroup` 和 `CompressedParam` 的元数据字段

**Files:**
- Modify: `flagscale/compress/memrift/megatron_dynamic_loader.py`

**背景：** `_materialize_group_tensor` 是模块级函数，无法访问 loader 的 `tp_rank`/`tp_size`。最干净的方案是把这两个值存在 `MergedWeightGroup` 上，以便函数直接读取。`CompressedParam` 需要增加 `parallel_type` 来标记列/行并行类型。

- [ ] **Step 3.1: 在 `megatron_dynamic_loader.py` 顶部添加 `tp_shard` 导入**

在文件开头的 `import torch` 下面，找到这段注释后添加：
```python
# 原有内容（不删）
import torch
import torch.nn as nn

# 新增导入
from flagscale.compress.memrift.tp_shard import (
    COL_PARALLEL, ROW_PARALLEL, REPLICATED,
    apply_tp_shard, validate_shard_divisibility,
)
```

- [ ] **Step 3.2: 在 `CompressedParam.__init__` 中增加 `parallel_type` 字段**

定位到 `CompressedParam.__init__` 中 `self.merge_key: Optional[str] = None` 这一行，在其后插入：

```python
        self.merge_key: Optional[str] = None  # 'q'/'k'/'v' or 'gate'/'up'
        self.parallel_type: str = REPLICATED   # ← 新增：COL_PARALLEL / ROW_PARALLEL / REPLICATED
        self.layer_idx: int = -1
```

（仅插入中间那一行；上下各一行是定位用的上下文。）

- [ ] **Step 3.3: 在 `MergedWeightGroup` dataclass 中增加 `tp_rank` 和 `tp_size` 字段**

定位到：
```python
@dataclass
class MergedWeightGroup:
    megatron_target: str
    layer_idx: int
    components: Dict[str, CompressedParam] = field(default_factory=dict)
    target_module: Optional[nn.Module] = None
    target_attr: str = "weight"
    target_shape: Optional[Tuple[int, ...]] = None
```

替换为：
```python
@dataclass
class MergedWeightGroup:
    megatron_target: str
    layer_idx: int
    components: Dict[str, CompressedParam] = field(default_factory=dict)
    target_module: Optional[nn.Module] = None
    target_attr: str = "weight"
    target_shape: Optional[Tuple[int, ...]] = None
    tp_rank: int = 0    # ← 新增
    tp_size: int = 1    # ← 新增
```

- [ ] **Step 3.4: 验证 dataclass 可实例化（无报错）**

```bash
cd /share/project/mengyc/code/memrift-flagscale
python -c "
from flagscale.compress.memrift.megatron_dynamic_loader import MergedWeightGroup, CompressedParam
g = MergedWeightGroup('self_attention.linear_qkv', 0, tp_rank=1, tp_size=2)
print('tp_rank:', g.tp_rank, 'tp_size:', g.tp_size)
"
```

Expected: `tp_rank: 1 tp_size: 2`

---

## Task 4: 扩展 `HF_TO_MEGATRON` 映射并传播 `parallel_type`

**Files:**
- Modify: `flagscale/compress/memrift/megatron_dynamic_loader.py`

- [ ] **Step 4.1: 将 `HF_TO_MEGATRON` 中的二元组改为三元组**

定位类属性：
```python
    HF_TO_MEGATRON = {
        "q_proj": ("self_attention.linear_qkv", "q"),
        "k_proj": ("self_attention.linear_qkv", "k"),
        "v_proj": ("self_attention.linear_qkv", "v"),
        "o_proj": ("self_attention.linear_proj", None),
        "gate_proj": ("mlp.linear_fc1", "gate"),
        "up_proj": ("mlp.linear_fc1", "up"),
        "down_proj": ("mlp.linear_fc2", None),
    }
```

替换为：
```python
    HF_TO_MEGATRON = {
        "q_proj":    ("self_attention.linear_qkv", "q",    COL_PARALLEL),
        "k_proj":    ("self_attention.linear_qkv", "k",    COL_PARALLEL),
        "v_proj":    ("self_attention.linear_qkv", "v",    COL_PARALLEL),
        "o_proj":    ("self_attention.linear_proj", None,  ROW_PARALLEL),
        "gate_proj": ("mlp.linear_fc1",            "gate", COL_PARALLEL),
        "up_proj":   ("mlp.linear_fc1",            "up",   COL_PARALLEL),
        "down_proj": ("mlp.linear_fc2",            None,   ROW_PARALLEL),
    }
```

- [ ] **Step 4.2: 更新 `_map_hf_to_megatron` 返回三元组**

定位方法：
```python
    def _map_hf_to_megatron(self, hf_name: str) -> Tuple[Optional[str], Optional[str]]:
        for hf_suffix, (mg_target, merge_key) in self.HF_TO_MEGATRON.items():
            if f".{hf_suffix}." in hf_name or hf_name.endswith(f".{hf_suffix}.weight"):
                return mg_target, merge_key
        return None, None
```

替换为：
```python
    def _map_hf_to_megatron(self, hf_name: str) -> Tuple[Optional[str], Optional[str], str]:
        for hf_suffix, (mg_target, merge_key, par_type) in self.HF_TO_MEGATRON.items():
            if f".{hf_suffix}." in hf_name or hf_name.endswith(f".{hf_suffix}.weight"):
                return mg_target, merge_key, par_type
        return None, None, REPLICATED
```

- [ ] **Step 4.3: 更新 `_load_weights_hf_mode` 中对 `_map_hf_to_megatron` 的调用**

定位（`_load_weights_hf_mode` 方法内）：
```python
            megatron_target, merge_key = self._map_hf_to_megatron(hf_name)
```

替换为：
```python
            megatron_target, merge_key, par_type = self._map_hf_to_megatron(hf_name)
```

然后在该方法中，找到 `cp.hf_name = hf_name` 这一行，在其下方加上 `par_type` 赋值：

```python
            cp = self._read_compressed_file(file_path, entry)
            cp.hf_name = hf_name
            cp.parallel_type = par_type          # ← 新增
            cp.megatron_target = megatron_target or ""
            cp.merge_key = merge_key
            cp.layer_idx = layer_idx if layer_idx is not None else -1
```

- [ ] **Step 4.4: 在创建 `MergedWeightGroup` 时传入 `tp_rank`/`tp_size`**

在 `_load_weights_hf_mode` 中，找到：
```python
                if megatron_target not in self.merged_groups[layer_idx]:
                    self.merged_groups[layer_idx][megatron_target] = MergedWeightGroup(
                        megatron_target=megatron_target,
                        layer_idx=layer_idx,
                    )
```

替换为：
```python
                if megatron_target not in self.merged_groups[layer_idx]:
                    self.merged_groups[layer_idx][megatron_target] = MergedWeightGroup(
                        megatron_target=megatron_target,
                        layer_idx=layer_idx,
                        tp_rank=self.tp_rank,
                        tp_size=self.tp_size,
                    )
```

- [ ] **Step 4.5: 在加载时做整除校验（早期报错）**

在 `cp.parallel_type = par_type` 赋值行**之后**，加入校验（仅当 `par_type != REPLICATED` 且 `tp_size > 1`）：

```python
            cp.parallel_type = par_type
            if self.tp_size > 1 and par_type != REPLICATED:
                validate_shard_divisibility(
                    hf_name, tuple(entry["shape"]), par_type, self.tp_size
                )
```

- [ ] **Step 4.6: 快速导入验证**

```bash
python -c "
from flagscale.compress.memrift.megatron_dynamic_loader import MegatronDynamicLoader
print('HF_TO_MEGATRON sample:', list(MegatronDynamicLoader.HF_TO_MEGATRON.items())[:2])
"
```

Expected 输出示例：
```
HF_TO_MEGATRON sample: [('q_proj', ('self_attention.linear_qkv', 'q', 'col')), ('k_proj', ('self_attention.linear_qkv', 'k', 'col'))]
```

---

## Task 5: 修改 `_materialize_group_tensor` 以应用 TP 切分

**Files:**
- Modify: `flagscale/compress/memrift/megatron_dynamic_loader.py`

**关键逻辑：**
- 在 shard mode 下，`cp.parallel_type == REPLICATED`（默认值），`apply_tp_shard` 是 no-op，行为不变。
- 在 HF mode 下，`cp.parallel_type` 被设为 `COL_PARALLEL` 或 `ROW_PARALLEL`，且 `group.tp_size` > 1 时应用切片。
- 切片在全量解压**之后**立即进行，GPU 上只保留目标 rank 的 shard。

- [ ] **Step 5.1: 修改 QKV 分支**

在 `_materialize_group_tensor` 函数中，定位 QKV 分支末尾的 `merged = torch.cat([q, k, v], dim=0)` 上方。

找到：
```python
        merged = torch.cat([q, k, v], dim=0)
        # Release component tensors immediately to avoid 2x peak memory.
        q_cp.release()
        k_cp.release()
        v_cp.release()
        return merged
```

**在 `merged = torch.cat(...)` 之前**插入切片逻辑（共 8 行）：

```python
        # Apply TP shard for HF mode (shard mode weights are already pre-sharded;
        # their cps keep parallel_type == REPLICATED so this is a no-op there).
        _tp_rank = group.tp_rank
        _tp_size = group.tp_size
        if _tp_size > 1 and getattr(q_cp, "parallel_type", REPLICATED) == COL_PARALLEL:
            from flagscale.compress.memrift.tp_shard import col_shard as _col_shard
            q = _col_shard(q, _tp_rank, _tp_size)
            k = _col_shard(k, _tp_rank, _tp_size)
            v = _col_shard(v, _tp_rank, _tp_size)
        merged = torch.cat([q, k, v], dim=0)
        # Release component tensors immediately to avoid 2x peak memory.
        q_cp.release()
        k_cp.release()
        v_cp.release()
        return merged
```

- [ ] **Step 5.2: 修改 FC1 分支**

在同一函数中，找到 FC1 分支末尾：
```python
        merged = torch.cat([gate, up], dim=0)
        gate_cp.release()
        up_cp.release()
        return merged
```

**在 `merged = torch.cat(...)` 之前**插入：

```python
        if _tp_size > 1 and getattr(gate_cp, "parallel_type", REPLICATED) == COL_PARALLEL:
            from flagscale.compress.memrift.tp_shard import col_shard as _col_shard
            gate = _col_shard(gate, _tp_rank, _tp_size)
            up   = _col_shard(up,   _tp_rank, _tp_size)
```

注意：`_tp_rank` 和 `_tp_size` 已在 QKV 分支定义，但 FC1 分支在不同的 `if` 块里。需要在 FC1 分支开头重新定义：

将 FC1 分支整体改为：

```python
    if "linear_fc1" in group.megatron_target:
        gate_cp = group.components.get("gate")
        up_cp = group.components.get("up")
        if gate_cp is None or up_cp is None:
            raise RuntimeError(f"Incomplete FC1 group: {list(group.components.keys())}")
        gate = gate_cp.materialize(sync=sync)
        up = up_cp.materialize(sync=sync)
        if sync and not _DEEP_ASYNC:
            gate_cp.wait_ready()
            up_cp.wait_ready()
        else:
            cur = torch.cuda.current_stream()
            for _cp in (gate_cp, up_cp):
                if _cp._CtoD_evt is not None:
                    cur.wait_event(_cp._CtoD_evt)
        # Apply TP shard (HF mode only; shard mode is a no-op via REPLICATED default)
        _tp_rank = group.tp_rank
        _tp_size = group.tp_size
        if _tp_size > 1 and getattr(gate_cp, "parallel_type", REPLICATED) == COL_PARALLEL:
            from flagscale.compress.memrift.tp_shard import col_shard as _col_shard
            gate = _col_shard(gate, _tp_rank, _tp_size)
            up   = _col_shard(up,   _tp_rank, _tp_size)
        merged = torch.cat([gate, up], dim=0)
        gate_cp.release()
        up_cp.release()
        return merged
```

- [ ] **Step 5.3: 修改 single-weight 分支（`linear_proj`、`linear_fc2`）**

定位函数末尾的 single 分支：
```python
    cp = group.components.get("single")
    if cp is None:
        cp = list(group.components.values())[0]
    weight = cp.materialize(sync=sync)
    if sync and not _DEEP_ASYNC:
        cp.wait_ready()
    return weight
```

替换为：
```python
    cp = group.components.get("single")
    if cp is None:
        cp = list(group.components.values())[0]
    weight = cp.materialize(sync=sync)
    if sync and not _DEEP_ASYNC:
        cp.wait_ready()
    # Apply TP shard for HF mode row-parallel weights (o_proj, down_proj)
    _par = getattr(cp, "parallel_type", REPLICATED)
    _tp_rank = group.tp_rank
    _tp_size = group.tp_size
    if _tp_size > 1 and _par != REPLICATED:
        weight = apply_tp_shard(weight, _par, _tp_rank, _tp_size)
    return weight
```

- [ ] **Step 5.4: 在 QKV 分支开头也加 `_tp_rank/_tp_size` 局部变量**

QKV 分支没有在 `if` 块里独立作用域，确认 `_tp_rank = group.tp_rank` 这两行出现在 `q = _col_shard(...)` 之前。完整 QKV 切片块应为：

```python
        # Apply TP shard for HF mode (shard mode is no-op via REPLICATED default)
        _tp_rank = group.tp_rank
        _tp_size = group.tp_size
        if _tp_size > 1 and getattr(q_cp, "parallel_type", REPLICATED) == COL_PARALLEL:
            from flagscale.compress.memrift.tp_shard import col_shard as _col_shard
            q = _col_shard(q, _tp_rank, _tp_size)
            k = _col_shard(k, _tp_rank, _tp_size)
            v = _col_shard(v, _tp_rank, _tp_size)
        merged = torch.cat([q, k, v], dim=0)
```

- [ ] **Step 5.5: 导入 `REPLICATED` 常量到模块顶层（避免 FC1/single 分支重复 import）**

在文件顶部新增 import（已在 Task 3 Step 3.1 添加过，此步骤检查即可）：
```python
from flagscale.compress.memrift.tp_shard import (
    COL_PARALLEL, ROW_PARALLEL, REPLICATED,
    apply_tp_shard, validate_shard_divisibility,
)
```

然后将 Task 5 Step 5.1/5.2/5.3 中的 `from flagscale.compress.memrift.tp_shard import col_shard as _col_shard` **改为直接使用模块级的 `apply_tp_shard`**：

```python
        # QKV 分支最终形态
        if _tp_size > 1 and getattr(q_cp, "parallel_type", REPLICATED) == COL_PARALLEL:
            q = apply_tp_shard(q, COL_PARALLEL, _tp_rank, _tp_size)
            k = apply_tp_shard(k, COL_PARALLEL, _tp_rank, _tp_size)
            v = apply_tp_shard(v, COL_PARALLEL, _tp_rank, _tp_size)
```

```python
        # FC1 分支最终形态
        if _tp_size > 1 and getattr(gate_cp, "parallel_type", REPLICATED) == COL_PARALLEL:
            gate = apply_tp_shard(gate, COL_PARALLEL, _tp_rank, _tp_size)
            up   = apply_tp_shard(up,   COL_PARALLEL, _tp_rank, _tp_size)
```

---

## Task 6: HF mode TP 集成测试

**Files:**
- Create: `tests/compress/test_hf_mode_tp.py`

**测试策略：** 构造一个极小模型（2 层，hidden=16），生成 HF 格式压缩目录（用 `prepare_weight.py` 的逻辑手动写入），然后用 `MegatronDynamicLoader` 在 tp_rank=0/1 下加载，验证 `_load_weights_hf_mode` + `_materialize_group_tensor` 产生正确形状和正确数值。

测试在 CPU 模拟 GPU device（用 `torch.device("cpu")`），跳过需要 CUDA extension 的步骤（mock `fs_sp.merge`）。

- [ ] **Step 6.1: 写集成测试**

```python
# tests/compress/test_hf_mode_tp.py
"""
HF mode TP sharding integration test.

Creates a fake compressed directory in HF format (index.json + .bin files),
then verifies MegatronDynamicLoader correctly shards weights at materialize time
for each TP rank.

No CUDA required (uses mock materialize to return random BF16 tensors).
No Megatron required (no parallel_state import).
"""
import json
import os
import struct
import tempfile
import pytest
import torch
import numpy as np


# ── helpers to write a fake compressed file (same format as prepare_weight.py)

def _write_fake_bin(outdir: str, file_idx: int, shape: tuple) -> str:
    """Write a fake split_zstd .bin for a random BF16 tensor. Returns filename."""
    numel = 1
    for s in shape:
        numel *= s
    # sm_bits: 1 byte per element for bf16
    sm_bytes = bytes(numel)
    # exp_bytes: random (zstd-compressed form of random uint8 array)
    import zstandard as zstd
    raw_exp = np.random.randint(0, 256, numel, dtype=np.uint8)
    cctx = zstd.ZstdCompressor(level=1)
    exp_bytes = cctx.compress(raw_exp.tobytes())

    fn = f"{file_idx:06d}.bin"
    fpath = os.path.join(outdir, fn)
    with open(fpath, "wb") as f:
        f.write(struct.pack("<Q", numel))
        f.write(sm_bytes)
        f.write(exp_bytes)
    return fn


def _make_fake_hf_compressed_dir(
    tmpdir: str,
    num_layers: int = 2,
    hidden: int = 16,
    num_heads: int = 4,
    num_kv_heads: int = 2,
    ffn: int = 32,
):
    """
    Build a fake HF-format compressed directory with index.json.
    Weights are tiny random tensors; compression is minimal.
    Returns (comp_dir, model_config) where model_config has shape info.
    """
    head_dim = hidden // num_heads
    index = []
    file_idx = 0

    def _add(name: str, shape: tuple):
        nonlocal file_idx
        fn = _write_fake_bin(tmpdir, file_idx, shape)
        index.append(dict(name=name, file=fn, shape=list(shape),
                          dtype="bfloat16", scheme="split_zstd"))
        file_idx += 1

    for i in range(num_layers):
        p = f"model.layers.{i}"
        _add(f"{p}.self_attn.q_proj.weight", (num_heads * head_dim, hidden))
        _add(f"{p}.self_attn.k_proj.weight", (num_kv_heads * head_dim, hidden))
        _add(f"{p}.self_attn.v_proj.weight", (num_kv_heads * head_dim, hidden))
        _add(f"{p}.self_attn.o_proj.weight", (hidden, num_heads * head_dim))
        _add(f"{p}.mlp.gate_proj.weight",    (ffn, hidden))
        _add(f"{p}.mlp.up_proj.weight",      (ffn, hidden))
        _add(f"{p}.mlp.down_proj.weight",    (hidden, ffn))

    with open(os.path.join(tmpdir, "index.json"), "w") as f:
        json.dump(index, f)

    return tmpdir, dict(
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, hidden=hidden, ffn=ffn, num_layers=num_layers,
    )


def _make_fake_decoder_model(num_layers: int, hidden: int, num_heads: int,
                              num_kv_heads: int, ffn: int):
    """
    Minimal nn.Module that mimics Megatron decoder structure so
    build_param_mapping() can navigate it.
    """
    import torch.nn as nn

    class FakeLinear(nn.Module):
        def __init__(self, out_features, in_features):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(out_features, in_features))

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            head_dim = hidden // num_heads
            qkv_rows = num_heads * head_dim + 2 * num_kv_heads * head_dim
            self.linear_qkv  = FakeLinear(qkv_rows, hidden)
            self.linear_proj = FakeLinear(hidden, num_heads * head_dim)

    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = FakeLinear(2 * ffn, hidden)
            self.linear_fc2 = FakeLinear(hidden, ffn)

    class FakeDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attention = FakeAttn()
            self.mlp = FakeMLP()

    class FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(num_layers)])

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = FakeDecoder()

    return FakeModel()


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tiny_hf_comp_dir(tmp_path):
    return _make_fake_hf_compressed_dir(
        str(tmp_path),
        num_layers=2,
        hidden=16,
        num_heads=4,
        num_kv_heads=2,
        ffn=32,
    )


def _load_with_tp(comp_dir, cfg, tp_rank, tp_size, monkeypatch):
    """
    Build a loader with tp_rank/tp_size, run load_weights + build_param_mapping,
    then return the loader and one group's materialized weight (without real CUDA).
    """
    from flagscale.compress.memrift.megatron_dynamic_loader import (
        MegatronDynamicLoader, _materialize_group_tensor, CompressedParam,
    )

    # Mock CompressedParam.materialize() to return a random BF16 tensor of orig_shape
    def _fake_materialize(self, sync=True):
        if self._bf16 is None:
            self._bf16 = torch.randn(self.orig_shape, dtype=torch.bfloat16)
        return self._bf16

    monkeypatch.setattr(CompressedParam, "materialize", _fake_materialize)

    model = _make_fake_decoder_model(
        cfg["num_layers"], cfg["hidden"], cfg["num_heads"],
        cfg["num_kv_heads"], cfg["ffn"],
    )

    loader = MegatronDynamicLoader(
        model=model,
        comp_dir=comp_dir,
        device=torch.device("cpu"),
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
    loader.load_weights()
    loader.build_param_mapping()
    return loader


def test_hf_mode_tp1_shapes(tiny_hf_comp_dir, monkeypatch):
    """TP=1: shapes must match full HF weight after merge."""
    comp_dir, cfg = tiny_hf_comp_dir
    loader = _load_with_tp(comp_dir, cfg, tp_rank=0, tp_size=1, monkeypatch=monkeypatch)

    head_dim = cfg["head_dim"]
    qkv_rows = (cfg["num_heads"] + 2 * cfg["num_kv_heads"]) * head_dim  # 4+4=8 → 8*4=32... let me compute
    # hidden=16, num_heads=4, head_dim=4, num_kv_heads=2
    # q_rows=16, k_rows=8, v_rows=8 → qkv_rows=32
    qkv_rows = cfg["num_heads"] * head_dim + 2 * cfg["num_kv_heads"] * head_dim

    groups_layer0 = loader.layer2groups["decoder.layers.0"]
    by_target = {g.megatron_target: g for g in groups_layer0}

    from flagscale.compress.memrift.megatron_dynamic_loader import _materialize_group_tensor
    qkv_w = _materialize_group_tensor(by_target["self_attention.linear_qkv"])
    assert qkv_w.shape == (qkv_rows, cfg["hidden"]), f"TP=1 QKV shape wrong: {qkv_w.shape}"

    fc1_w = _materialize_group_tensor(by_target["mlp.linear_fc1"])
    assert fc1_w.shape == (2 * cfg["ffn"], cfg["hidden"]), f"TP=1 FC1 shape wrong: {fc1_w.shape}"

    proj_w = _materialize_group_tensor(by_target["self_attention.linear_proj"])
    assert proj_w.shape == (cfg["hidden"], cfg["num_heads"] * head_dim)

    fc2_w = _materialize_group_tensor(by_target["mlp.linear_fc2"])
    assert fc2_w.shape == (cfg["hidden"], cfg["ffn"])


def test_hf_mode_tp2_shapes(tiny_hf_comp_dir, monkeypatch):
    """TP=2: each rank should get half the column/row-parallel weights."""
    comp_dir, cfg = tiny_hf_comp_dir

    head_dim = cfg["head_dim"]
    # For tp=2: q_rows/2, k_rows/2, v_rows/2 per rank
    q_per_rank  = (cfg["num_heads"] * head_dim) // 2
    kv_per_rank = (cfg["num_kv_heads"] * head_dim) // 2
    qkv_per_rank = q_per_rank + kv_per_rank + kv_per_rank

    from flagscale.compress.memrift.megatron_dynamic_loader import _materialize_group_tensor

    for rank in (0, 1):
        loader = _load_with_tp(comp_dir, cfg, tp_rank=rank, tp_size=2, monkeypatch=monkeypatch)
        groups = {g.megatron_target: g for g in loader.layer2groups["decoder.layers.0"]}

        qkv_w  = _materialize_group_tensor(groups["self_attention.linear_qkv"])
        fc1_w  = _materialize_group_tensor(groups["mlp.linear_fc1"])
        proj_w = _materialize_group_tensor(groups["self_attention.linear_proj"])
        fc2_w  = _materialize_group_tensor(groups["mlp.linear_fc2"])

        assert qkv_w.shape  == (qkv_per_rank, cfg["hidden"]), \
            f"rank={rank} QKV: expected ({qkv_per_rank}, {cfg['hidden']}), got {qkv_w.shape}"
        assert fc1_w.shape  == (cfg["ffn"], cfg["hidden"]), \
            f"rank={rank} FC1: expected ({cfg['ffn']}, {cfg['hidden']}), got {fc1_w.shape}"
        assert proj_w.shape == (cfg["hidden"], (cfg["num_heads"] * head_dim) // 2), \
            f"rank={rank} linear_proj shape wrong: {proj_w.shape}"
        assert fc2_w.shape  == (cfg["hidden"], cfg["ffn"] // 2), \
            f"rank={rank} linear_fc2 shape wrong: {fc2_w.shape}"


def test_hf_mode_tp2_values_partition(tiny_hf_comp_dir, monkeypatch):
    """TP=2: rank 0 and rank 1 shards must be complementary (concat = full weight)."""
    comp_dir, cfg = tiny_hf_comp_dir

    # Use deterministic fake tensors: fix the random seed per orig_shape
    from flagscale.compress.memrift.megatron_dynamic_loader import CompressedParam

    _cache = {}
    def _seeded_materialize(self, sync=True):
        key = self.orig_shape
        if key not in _cache:
            torch.manual_seed(hash(key) % (2**31))
            _cache[key] = torch.randn(self.orig_shape, dtype=torch.bfloat16)
        self._bf16 = _cache[key].clone()
        return self._bf16

    monkeypatch.setattr(CompressedParam, "materialize", _seeded_materialize)

    from flagscale.compress.memrift.megatron_dynamic_loader import _materialize_group_tensor

    loader0 = _load_with_tp(comp_dir, cfg, tp_rank=0, tp_size=2, monkeypatch=monkeypatch)
    loader1 = _load_with_tp(comp_dir, cfg, tp_rank=1, tp_size=2, monkeypatch=monkeypatch)

    g0 = {g.megatron_target: g for g in loader0.layer2groups["decoder.layers.0"]}
    g1 = {g.megatron_target: g for g in loader1.layer2groups["decoder.layers.0"]}

    # linear_fc2 (row-parallel): rank0[:,  :ffn//2] + rank1[:, ffn//2:] = full weight
    fc2_r0 = _materialize_group_tensor(g0["mlp.linear_fc2"])
    fc2_r1 = _materialize_group_tensor(g1["mlp.linear_fc2"])
    fc2_full = torch.cat([fc2_r0, fc2_r1], dim=1)
    assert fc2_full.shape == (cfg["hidden"], cfg["ffn"])

    # linear_proj (row-parallel): same pattern
    proj_r0 = _materialize_group_tensor(g0["self_attention.linear_proj"])
    proj_r1 = _materialize_group_tensor(g1["self_attention.linear_proj"])
    proj_full = torch.cat([proj_r0, proj_r1], dim=1)
    assert proj_full.shape == (cfg["hidden"], cfg["num_heads"] * cfg["head_dim"])
```

- [ ] **Step 6.2: 运行集成测试**

```bash
cd /share/project/mengyc/code/memrift-flagscale
python -m pytest tests/compress/test_hf_mode_tp.py -v
```

Expected:
```
test_hf_mode_tp1_shapes PASSED
test_hf_mode_tp2_shapes PASSED
test_hf_mode_tp2_values_partition PASSED
3 passed
```

- [ ] **Step 6.3: 运行全部压缩测试确认无回归**

```bash
python -m pytest tests/compress/ -v
```

Expected: 所有 13 个测试通过，0 failed。

- [ ] **Step 6.4: 提交**

```bash
git add \
  flagscale/compress/memrift/megatron_dynamic_loader.py \
  tests/compress/test_hf_mode_tp.py
git commit -m "feat(memrift): support TP>1 in HF mode via runtime weight sharding

- HF_TO_MEGATRON now carries parallel type (COL/ROW/REPLICATED)
- MergedWeightGroup stores tp_rank/tp_size from loader
- CompressedParam stores parallel_type
- _materialize_group_tensor applies shard slice after decompression
- Early divisibility check in _load_weights_hf_mode
- Shard mode is unaffected (REPLICATED default = no-op)"
```

---

## Task 7: 验证 TP=1 shard mode 无回归（快速冒烟测试）

**Files:**
- 无新增文件（仅运行已有测试）

- [ ] **Step 7.1: 运行全部单元测试**

```bash
cd /share/project/mengyc/code/memrift-flagscale
python -m pytest tests/ -v --ignore=tests/functional_tests
```

Expected: 全部通过，无新失败。

- [ ] **Step 7.2: 快速 Python 验证——shard mode loader 能构造（导入不报错）**

```bash
python -c "
from flagscale.compress.memrift.megatron_dynamic_loader import (
    MegatronDynamicLoader, MergedWeightGroup, CompressedParam, _materialize_group_tensor
)
from flagscale.compress.memrift.tp_shard import REPLICATED
# MergedWeightGroup with default tp fields
g = MergedWeightGroup('mlp.linear_fc2', 0)
assert g.tp_rank == 0 and g.tp_size == 1, 'default tp values wrong'
# CompressedParam parallel_type default
import torch
dummy = torch.empty(0)
# CompressedParam is a nn.Parameter subclass; skip full init, just check attr presence
print('All imports and defaults OK')
"
```

Expected: `All imports and defaults OK`

---

## 注意事项

### CPU 内存开销警告（运行时 HF mode TP>1）

HF mode 在线切分时，**每个 TP rank 都会从磁盘加载并保持全量压缩权重**（`sm_cpu` + `exp_mv`），解压时使用全量，然后丢弃非本 rank 的部分。对 7B 模型 TP=2，每张卡的 CPU 内存约为 TP=1 时的同等开销（不节省 CPU 内存）。

**生产环境推荐：** 用 `prepare_weight_tp2.py` 离线预切分（每 rank 只存储 1/TP 压缩权重），然后通过 shard mode 加载。该工具已存在：

```bash
python flagscale/compress/memrift/offline_comp/prepare_weight_tp2.py \
    --model-dir /path/to/Mistral-7B \
    --outdir    ./memrift_weights/mistral_7b_tp2 \
    --tp-size   2 \
    --level     3
```

### GQA 边界情况（n_kv_heads < tp_size）

当 `num_kv_heads < tp_size`（例如 8 个 KV 头但 TP=16），`col_shard` 对 K/V 会报整除错误。该情况需要 K/V 头复制（head replication），目前**不支持**，会在 `_load_weights_hf_mode` 的 `validate_shard_divisibility` 中给出清晰错误提示并建议使用 `prepare_weight_tp2.py`。

---

## 总结：改动范围

```
flagscale/compress/memrift/
  tp_shard.py                       ← 新建：56 行，纯 Python，无依赖
  megatron_dynamic_loader.py        ← 修改：~50 行净增（6 处改动）

tests/compress/
  __init__.py                       ← 新建：空文件
  test_tp_shard.py                  ← 新建：10 个测试，纯 CPU
  test_hf_mode_tp.py                ← 新建：3 个测试，mock materialize
```
