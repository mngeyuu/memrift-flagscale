# MemRift 激活压缩 zstd level 解耦 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给激活压缩一个独立的低默认 zstd level（`--memrift-act-zstd-level`，默认 3），与权重 `--memrift-zstd-level` 解耦，消除 ~32× 速度回归。

**Architecture:** 训练期那个共享 `AsyncCompressor` 的 `zstd_level` 实际只作用于激活压缩（权重预取仅解压、与 level 无关）。新增独立参数喂给该 compressor 的 level（及激活 fallback compressor），权重 level 不再影响激活。

**Tech Stack:** Megatron 参数注册（arguments_fs.py）、flagscale memrift train_hooks、zstandard、集成诊断脚本。

**设计文档:** `docs/superpowers/specs/2026-06-03-memrift-activation-zstd-level-design.md`
**分支:** `fix/memrift-te-backward`（承接权重修复；设计文档已在此分支提交）

**验证方式:** 无 pytest 适用；用集成诊断脚本 `scripts/diag_tinyllama_weight_mem.sh`（TinyLlama TP=1）。
已知基线：full memrift @ 激活 level18 = **117.8s/iter**；@ level1 = **~3.7s/iter**；weight-only = **1.8s/iter**；峰值 3739 MB。
运行约定：跑实验前 `ssh 172.24.178.248` 先 `nvidia-smi` 确认 GPU 空闲；脚本内部已激活 conda `myc`，远程共享 /share。

---

## 文件结构

| 文件 | 职责 | 改动 |
|---|---|---|
| `flagscale/train/megatron/training/arguments_fs.py` | memrift CLI 参数注册 | 新增 `--memrift-act-zstd-level`（默认 3） |
| `flagscale/compress/memrift/train_hooks.py` | memrift 注入编排 | 读新参数；共享 AsyncCompressor 用激活 level；激活 fallback 用激活 level；status 增列 |
| `scripts/diag_tinyllama_weight_mem.sh` | 集成诊断（验证用） | memrift 分支传 `--memrift-act-zstd-level ${ACT_ZLEVEL:-3}` |

---

## Task 1: 注册 `--memrift-act-zstd-level` 参数

**Files:**
- Modify: `flagscale/train/megatron/training/arguments_fs.py`（在 `--memrift-zstd-level` 块 :954-959 之后）

- [ ] **Step 1: 在 `--memrift-zstd-level` 参数块后新增参数**

当前 :954-965 是：
```python
    group.add_argument(
        '--memrift-zstd-level',
        type=int,
        default=6,
        help='Zstd compression level for MemRift (1-22, default 6).',
    )
    group.add_argument(
        '--memrift-prefetch-layers',
```
在 `--memrift-zstd-level` 块与 `--memrift-prefetch-layers` 块之间插入：
```python
    group.add_argument(
        '--memrift-act-zstd-level',
        type=int,
        default=3,
        help='Zstd level for online ACTIVATION compression only (1-22, default 3). '
             'Independent of --memrift-zstd-level (which is the offline weight-prep '
             'level and has no runtime effect on activations). High levels (e.g. 18) '
             'make online activation compression extremely slow.',
    )
```

- [ ] **Step 2: 校验解析**

Run:
```bash
cd /share/project/mengyc/code/memrift-flagscale
python -c "import ast; ast.parse(open('flagscale/train/megatron/training/arguments_fs.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add flagscale/train/megatron/training/arguments_fs.py
git commit -m "feat(memrift): 新增 --memrift-act-zstd-level（激活压缩独立 level，默认3）

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: train_hooks 用激活 level 驱动激活压缩器

**Files:**
- Modify: `flagscale/compress/memrift/train_hooks.py`（4 处）

- [ ] **Step 1: 读取新参数**

在 `inject_memrift_if_configured` 中，找到（约 :119）：
```python
    zstd_level = getattr(args, "memrift_zstd_level", 6)
```
在其后新增一行：
```python
    act_zstd_level = getattr(args, "memrift_act_zstd_level", 3)
```

- [ ] **Step 2: 共享 AsyncCompressor 改用激活 level**

找到共享 compressor 创建处（约 :180-187）：
```python
            async_compressor = AsyncCompressor(
                compress_workers=compress_workers,
                decode_workers=decode_workers,
                concurrency_limit=4,
                zstd_level=zstd_level,
                enable_async=True,
            )
```
把 `zstd_level=zstd_level` 改为 `zstd_level=act_zstd_level`，并在该行上方加注释：
```python
            async_compressor = AsyncCompressor(
                compress_workers=compress_workers,
                decode_workers=decode_workers,
                concurrency_limit=4,
                # This compressor's level governs ONLY activation compression
                # (weight prefetch is decompress-only). Use the activation level.
                zstd_level=act_zstd_level,
                enable_async=True,
            )
```

- [ ] **Step 3: 把激活 level 传入 `_inject_activation_compression` 并用于 fallback**

(a) 在调用处（约 :216-224）：
```python
        _inject_activation_compression(
            model_chunks=model_chunks,
            async_compressor=async_compressor,
            act_async=act_async,
            print_debug=print_debug,
            rank=rank,
            compress_activations=activation_enable,
        )
```
新增一个实参 `act_zstd_level=act_zstd_level,`，变为：
```python
        _inject_activation_compression(
            model_chunks=model_chunks,
            async_compressor=async_compressor,
            act_async=act_async,
            print_debug=print_debug,
            rank=rank,
            compress_activations=activation_enable,
            act_zstd_level=act_zstd_level,
        )
```

(b) 函数定义（约 :370-377）：
```python
def _inject_activation_compression(
    model_chunks: List[nn.Module],
    async_compressor: Optional[Any],
    act_async: bool,
    print_debug: bool,
    rank: int = 0,
    compress_activations: bool = True,
) -> None:
```
新增形参 `act_zstd_level: int = 3,`，变为：
```python
def _inject_activation_compression(
    model_chunks: List[nn.Module],
    async_compressor: Optional[Any],
    act_async: bool,
    print_debug: bool,
    rank: int = 0,
    compress_activations: bool = True,
    act_zstd_level: int = 3,
) -> None:
```

(c) fallback compressor 的 level（约 :396）。当前：
```python
    zstd_level = 18
    compressor = async_compressor
```
改为：
```python
    zstd_level = act_zstd_level
    compressor = async_compressor
```

- [ ] **Step 4: get_memrift_status 增列**

找到 `get_memrift_status` 的返回 dict（约 :582-591），在 `"memrift_zstd_level": ...` 行后新增：
```python
        "memrift_act_zstd_level": getattr(args, "memrift_act_zstd_level", 3),
```

- [ ] **Step 5: 校验解析**

Run:
```bash
cd /share/project/mengyc/code/memrift-flagscale
python -c "import ast; ast.parse(open('flagscale/compress/memrift/train_hooks.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 6: 提交**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add flagscale/compress/memrift/train_hooks.py
git commit -m "fix(memrift): 激活压缩用 act_zstd_level（默认3），与权重 level 解耦

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: 诊断脚本支持激活 level 开关

**Files:**
- Modify: `scripts/diag_tinyllama_weight_mem.sh`（memrift 分支 MEMRIFT_ARGS）

- [ ] **Step 1: 给 memrift 分支传新参数**

找到（memrift 分支内）：
```bash
    --memrift-zstd-level ${ZLEVEL:-18} --memrift-prefetch-layers ${PREFETCH} --memrift-weight-async \
```
改为（追加激活 level，默认 3）：
```bash
    --memrift-zstd-level ${ZLEVEL:-18} --memrift-act-zstd-level ${ACT_ZLEVEL:-3} --memrift-prefetch-layers ${PREFETCH} --memrift-weight-async \
```

- [ ] **Step 2: 语法校验**

Run:
```bash
bash -n /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh && echo OK
```
Expected: `OK`

- [ ] **Step 3: 提交**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add scripts/diag_tinyllama_weight_mem.sh
git commit -m "test(memrift): diag 脚本支持 ACT_ZLEVEL 激活压缩 level 开关

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 端到端验证（TinyLlama）

**Files:** 无代码改动（验证）

- [ ] **Step 1: 默认（激活 level 3）应快**

先查 GPU 空闲：
```bash
ssh 172.24.178.248 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
```
选空闲 GPU（memory.used 接近 0，缺省 0）。运行：
```bash
ssh 172.24.178.248 "CUDA_VISIBLE_DEVICES=0 MODE=memrift PREFETCH=4 TRAIN_ITERS=3 MEMRIFT_HOOK_ORDER=0 MASTER_PORT=29581 bash /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh > /tmp/v_def.out 2>&1; echo EXIT=$?"
ssh 172.24.178.248 "grep -aE 'lm loss|MEM_PROBE\] K. peak max_memory_allocated|illegal|WeightPlaceholder' /share/project/mengyc/code/memrift-flagscale/output/diag_tinyllama_memrift/train.log | grep -v apex | tail -8"
```
Expected（PASS）：3 行 `lm loss`，无崩溃；稳态每 iter **~3-5s**（远低于 117.8s）；峰值 `K` ~3739 MB。记录 iter2/iter3 的 `elapsed time per iteration`。

- [ ] **Step 2: 开关有效性——显式 act level 18 复现旧慢速**

```bash
ssh 172.24.178.248 "CUDA_VISIBLE_DEVICES=0 MODE=memrift ACT_ZLEVEL=18 PREFETCH=4 TRAIN_ITERS=2 MEMRIFT_HOOK_ORDER=0 MASTER_PORT=29582 bash /share/project/mengyc/code/memrift-flagscale/scripts/diag_tinyllama_weight_mem.sh > /tmp/v_18.out 2>&1; echo EXIT=$?"
ssh 172.24.178.248 "grep -aE 'iteration .*elapsed time per iteration' /share/project/mengyc/code/memrift-flagscale/output/diag_tinyllama_memrift/train.log | tail -2"
```
Expected：iter 耗时回到 ~100s+ 量级（证明 `--memrift-act-zstd-level` 旋钮确实作用于激活压缩、默认 3 是提速来源）。

- [ ] **Step 3: 汇总**

记录三档：默认(act3) 每 iter 秒数 / 显式 act18 每 iter 秒数 / （已知）weight-only 1.8s。确认默认配置下 memrift 提速 ~25-30×、峰值与正确性不变。

- [ ] **Step 4: （留待之后，用户指定）Mistral-7B 复测**

经用户确认后，用 `run_compare_mistral7b_v02_fix.sh`（去掉脚本里的 `--memrift-zstd-level 18`，改为依赖新默认 act level 3，或加 `--memrift-act-zstd-level 3`）复测 memrift vs LoRA 的峰值与速度。本计划不执行此步。

---

## 自检（spec 覆盖）

- 设计 §3 新增 `--memrift-act-zstd-level`（默认3）→ Task 1。
- 设计 §4 改动清单：arguments_fs（Task 1）、train_hooks 三处+status（Task 2）、diag 脚本（Task 3）。
- 设计 §5 验证（TinyLlama 默认快 / 旋钮有效 / weight-only 不受影响）→ Task 4 Step 1-3。
- 设计 §5 Mistral 留待之后 → Task 4 Step 4（不执行）。
- 命名一致性：参数 `--memrift-act-zstd-level` / 属性 `memrift_act_zstd_level` / 变量 `act_zstd_level` / 形参 `act_zstd_level` 全程一致；诊断 env `ACT_ZLEVEL`。
- 无占位符；所有改动给出精确 old→new 代码。
