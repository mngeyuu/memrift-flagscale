# D2H/H2D 零拷贝改 cudaMemcpyAsync 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `float_split_stride_pin.cu` 的 `split`(D2H)和 `merge`(H2D)零拷贝传输路径原地改为 GPU 暂存 buffer + `cudaMemcpyAsync`,签名/返回值不变,memrift 全部调用点零改动。

**Architecture:** 当前 kernel 通过 `cudaHostGetDevicePointer` 直接读/写 pinned host 内存(零拷贝,细粒度 PCIe)。改为:D2H 时 kernel 写本地显存暂存再 `cudaMemcpyAsync` 整块回拷 host;H2D 时先 `cudaMemcpyAsync` 把 host exp 拷进显存暂存再让 kernel 读。memcpy 与 kernel enqueue 在同一条外部 stream 保证有序;暂存 buffer 用自定义 `Pool` + `record_stream` 管生命周期。搬运字节与零拷贝完全相同,故数值逐位不变 —— 这是验证用 loss 对比的理论依据。

**Tech Stack:** CUDA / C++ / PyTorch C++ 扩展(`torch/extension.h`、`c10::cuda`)、自定义 `Pool` 内存池、Megatron-LM(memrift 训练)、zstd。

**关键约束 & 环境:**
- 目标文件唯一:`flagscale/compress/float_split_stride_pin/float_split_stride_pin.cu`(UTF-8 编码)。
- **不要**碰仓库根目录的 `backup.cu`(ISO-8859 乱码副本,非目标)。
- 不新增 pybind 函数,不改任何 `.py`。
- 编译 + 跑训练必须在 GPU 机:`ssh 172.24.178.248`,脚本内已 `conda activate myc`。`/share/project/...` 为共享路径,本地编辑远端可见。
- 设计文档:`docs/superpowers/specs/2026-06-10-h2d-d2h-cudamemcpyasync-design.md`。

---

### Task 0: 捕获零拷贝 baseline(改任何代码之前)

**为什么先做:** 验证标准是"改后 loss 与零拷贝 baseline 对比"。一旦改了 `.cu`,就再也拿不到干净 baseline。必须先跑。

**Files:**
- 只读运行,不改源码。产出日志:
  - `outputs/baseline_zerocopy_roundtrip.log`
  - `outputs/baseline_zerocopy.log`

- [ ] **Step 1: 确认当前扩展已编译(零拷贝版)**

在 GPU 机执行:
```bash
ssh 172.24.178.248
cd /share/project/mengyc/code/memrift-flagscale/flagscale/compress/float_split_stride_pin
source /root/miniconda3/etc/profile.d/conda.sh && conda activate myc
python -c "from float_split_stride_pin import float_split_stride_pin as f; print('split' in dir(f._ext), 'merge' in dir(f._ext))"
```
Expected: `True True`。若 import 失败,先 `pip install -e .` 编译当前(未改动)版本。

- [ ] **Step 2: 跑 round-trip 单测,确认零拷贝版数值正确(green 基线)**

```bash
cd /share/project/mengyc/code/memrift-flagscale
source /root/miniconda3/etc/profile.d/conda.sh && conda activate myc
CUDA_VISIBLE_DEVICES=2 python test_roundtrip_minimal.py 2>&1 | tee outputs/baseline_zerocopy_roundtrip.log
```
Expected: 每个 case 打印 `torch.equal: True`、`max_abs_diff: 0`、`n_diff: 0/...`。这证明零拷贝版无损;改后必须仍然全 True。

- [ ] **Step 3: 跑 baseline 训练,记录 iter-1 loss**

```bash
cd /share/project/mengyc/code/memrift-flagscale
bash memrift_full.sh 2>&1 | tee outputs/baseline_zerocopy.log
```
Expected: 训练正常结束,日志含一行形如 `iteration        1/...` 带 `lm loss: <值>`。记下该 loss 数值(后续对比)。无 NaN、无 traceback。

- [ ] **Step 4: 提交 baseline 日志(便于回溯)**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add outputs/baseline_zerocopy_roundtrip.log outputs/baseline_zerocopy.log
git commit -m "test(memrift): 零拷贝 baseline (roundtrip + memrift_full iter-1 loss)"
```

> 若 `outputs/` 被 `.gitignore` 忽略导致 add 失败:改用 `git add -f`,或把两份日志另存到 `docs/superpowers/plans/baseline/` 再提交。loss 数值务必同时手抄进 commit message。

---

### Task 1: 改动 A —— `pack_tensor`(`split`,D2H)原地改为 cudaMemcpyAsync

**Files:**
- Modify: `flagscale/compress/float_split_stride_pin/float_split_stride_pin.cu`(`pack_tensor` 函数体,约行 328-362)

**改动要点:** 删除 host 映射三行,新增 GPU 暂存 `exp_gpu`;四个 kernel 分支的 exp 实参从 `dev_ptr` 改为 `exp_gpu.data_ptr<uint8_t>()`;kernel 后加 D2H `cudaMemcpyAsync` + `record_stream` + `Pool::put`。返回值不变。

- [ ] **Step 1: 删除 host 映射、改为申请 GPU 暂存**

把这段(当前)代码:
```cpp
    uint8_t* host_ptr = exp_host.data_ptr<uint8_t>();
    uint8_t* dev_ptr;
    cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    auto ix  = make_indexer<4>(t);
```
替换为:
```cpp
    auto exp_gpu = Pool::inst().get(N, at::kByte, dev_idx, /*pinned=*/false, raw);
    auto ix  = make_indexer<4>(t);
```

- [ ] **Step 2: 把四个 kernel 分支的 exp 实参 `dev_ptr` 换成 `exp_gpu.data_ptr<uint8_t>()`**

`pack_tensor` 内 4 处 `dev_ptr,`(分别在 `pack_fp32_kernel_vec2` / `pack_fp32_kernel` / `pack_bf16_kernel_vec2` / `pack_bf16_kernel` 调用中)全部替换为:
```cpp
                exp_gpu.data_ptr<uint8_t>(),
```
改完后该 `if/else` 块形如:
```cpp
    if (t.scalar_type() == at::kFloat) {
        if (ix.is_contig)
            pack_fp32_kernel_vec2<4><<<grid,256,0,raw>>>(
                t.data_ptr<float>(), ix,
                exp_gpu.data_ptr<uint8_t>(),
                sm.data_ptr<uint8_t>(), N);
        else
            pack_fp32_kernel<4><<<grid,256,0,raw>>>(
                t.data_ptr<float>(), ix,
                exp_gpu.data_ptr<uint8_t>(),
                sm.data_ptr<uint8_t>(), N);
    } else {    // bf16
        if (ix.is_contig)
            pack_bf16_kernel_vec2<4><<<grid,256,0,raw>>>(
                reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
                ix,
                exp_gpu.data_ptr<uint8_t>(),
                sm.data_ptr<uint8_t>(), N);
        else
            pack_bf16_kernel<4><<<grid,256,0,raw>>>(
                reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
                ix,
                exp_gpu.data_ptr<uint8_t>(),
                sm.data_ptr<uint8_t>(), N);
    }
```

- [ ] **Step 3: kernel 后、`return` 前插入 D2H 异步回拷 + 生命周期管理**

把结尾这行:
```cpp
    return {exp_host, sm};
```
替换为:
```cpp
    cudaError_t cpy_err = cudaMemcpyAsync(exp_host.data_ptr<uint8_t>(),
                                          exp_gpu.data_ptr<uint8_t>(),
                                          static_cast<size_t>(N),
                                          cudaMemcpyDeviceToHost, raw);
    TORCH_CHECK(cpy_err == cudaSuccess,
                "cudaMemcpyAsync D2H failed: ", cudaGetErrorString(cpy_err));
    exp_gpu.record_stream(s);
    Pool::inst().put(std::move(exp_gpu), /*pinned=*/false, raw);
    return {exp_host, sm};
```

- [ ] **Step 4: 静态自检(不编译,肉眼核对)**

确认:① `pack_tensor` 内已无 `cudaHostGetDevicePointer` / `host_ptr` / `dev_ptr`;② `exp_gpu` 声明一次、kernel 实参用 4 次、memcpy 用 1 次、`put` 用 1 次;③ 返回仍是 `{exp_host, sm}`(2 元)。
Run:
```bash
cd /share/project/mengyc/code/memrift-flagscale/flagscale/compress/float_split_stride_pin
grep -n "cudaHostGetDevicePointer\|dev_ptr\|exp_gpu" float_split_stride_pin.cu
```
Expected: `cudaHostGetDevicePointer` 只剩 `unpack_tensor` 里那 1 处(行 ~406);`pack_tensor` 区间(约 314-365)只见 `exp_gpu`,无 `dev_ptr`。

- [ ] **Step 5: 提交**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add flagscale/compress/float_split_stride_pin/float_split_stride_pin.cu
git commit -m "feat(fsp): split(D2H) 零拷贝改 GPU暂存+cudaMemcpyAsync (签名不变)"
```

---

### Task 2: 改动 B —— `unpack_tensor`(`merge`,H2D)原地改为 cudaMemcpyAsync

**Files:**
- Modify: `flagscale/compress/float_split_stride_pin/float_split_stride_pin.cu`(`unpack_tensor` 的 pinned-host 分支,约行 396-415)

**改动要点:** 只改 `else`(CPU pinned)分支;`exp.is_cuda()` 分支不变。删 `cudaHostGetDevicePointer` 及其 `.to()` fallback,改为 GPU 暂存 + 异步 H2D,复用已存在的 `tmp_gpu_exp` 持活。

- [ ] **Step 1: 替换 pinned-host 分支**

把这段(当前)代码:
```cpp
    if (exp.is_cuda()) {
        exp_dev_ptr = exp.data_ptr<uint8_t>();
    } else {
        TORCH_CHECK(exp.is_pinned(),
                    "exp on CPU must be pinned for zero-copy");

        cudaError_t err = cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&exp_dev_ptr),
                exp.data_ptr(), 0);

        if (err != cudaSuccess) {
            tmp_gpu_exp.emplace(
                exp.to(sm.device(), /*non_blocking=*/true));
            exp_dev_ptr = tmp_gpu_exp->data_ptr<uint8_t>();
        }
    }
```
替换为:
```cpp
    if (exp.is_cuda()) {
        exp_dev_ptr = exp.data_ptr<uint8_t>();
    } else {
        TORCH_CHECK(exp.is_pinned(),
                    "exp on CPU must be pinned for async H2D copy");

        const int64_t E = exp.numel();
        auto exp_gpu = Pool::inst().get(E, at::kByte, dev_idx, /*pinned=*/false, raw);
        cudaError_t err = cudaMemcpyAsync(exp_gpu.data_ptr<uint8_t>(),
                                          exp.data_ptr<uint8_t>(),
                                          static_cast<size_t>(E),
                                          cudaMemcpyHostToDevice, raw);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaMemcpyAsync H2D failed: ", cudaGetErrorString(err));
        exp_gpu.record_stream(s);
        exp_dev_ptr = exp_gpu.data_ptr<uint8_t>();
        tmp_gpu_exp.emplace(std::move(exp_gpu));
    }
```

- [ ] **Step 2: 静态自检**

Run:
```bash
cd /share/project/mengyc/code/memrift-flagscale/flagscale/compress/float_split_stride_pin
grep -n "cudaHostGetDevicePointer\|cudaMemcpyAsync" float_split_stride_pin.cu
```
Expected: `cudaHostGetDevicePointer` 已 **0 处**(全文消失);`cudaMemcpyAsync` 共 2 处(`pack_tensor` 的 D2H + `unpack_tensor` 的 H2D)。

- [ ] **Step 3: 提交**

```bash
cd /share/project/mengyc/code/memrift-flagscale
git add flagscale/compress/float_split_stride_pin/float_split_stride_pin.cu
git commit -m "feat(fsp): merge(H2D) 零拷贝改 GPU暂存+cudaMemcpyAsync (签名不变)"
```

---

### Task 3: 重新编译扩展

**Files:** 无源码改动(仅 build)。

- [ ] **Step 1: 在 GPU 机重编译**

```bash
ssh 172.24.178.248
cd /share/project/mengyc/code/memrift-flagscale/flagscale/compress/float_split_stride_pin
source /root/miniconda3/etc/profile.d/conda.sh && conda activate myc
pip install -e . 2>&1 | tail -30
```
Expected: 编译无 error,结尾 `Successfully installed float_split_stride_pin` 或 `Finished processing dependencies`。若报 nvcc/类型错误,回到对应 Task 修代码,勿继续。

- [ ] **Step 2: 确认符号仍在、签名未变**

```bash
python -c "from float_split_stride_pin import float_split_stride_pin as f; print('split' in dir(f._ext), 'merge' in dir(f._ext))"
```
Expected: `True True`。

---

### Task 4: 单元 round-trip(数值正确性闸门)

**Files:** 运行 `test_roundtrip_minimal.py`;产出 `outputs/after_memcpy_roundtrip.log`。

> 说明:spec §6.2 的"小流程冒烟"由 Task 5 的 `memrift_full.sh` 全路径运行承担(它本身就是端到端集成冒烟),此处不再单列轻量 smoke,避免重复。

- [ ] **Step 1: 跑 round-trip,必须与零拷贝 baseline 同样全 True**

```bash
cd /share/project/mengyc/code/memrift-flagscale
source /root/miniconda3/etc/profile.d/conda.sh && conda activate myc
CUDA_VISIBLE_DEVICES=2 python test_roundtrip_minimal.py 2>&1 | tee outputs/after_memcpy_roundtrip.log
```
Expected: 所有 case `torch.equal: True`、`max_abs_diff: 0`、`n_diff: 0/...`。
若任一 case `False`:说明 D2H 或 H2D 传输/生命周期错误 —— 进入 superpowers:systematic-debugging,优先怀疑 `record_stream`/`Pool::put` 时序或 memcpy 大小(D2H 用 `N`、H2D 用 `exp.numel()`)。

- [ ] **Step 2: 对比两份 round-trip 日志一致**

```bash
diff outputs/baseline_zerocopy_roundtrip.log outputs/after_memcpy_roundtrip.log && echo IDENTICAL
```
Expected: `IDENTICAL`(数值部分完全一致)。

- [ ] **Step 3: 提交单测日志**

```bash
git add outputs/after_memcpy_roundtrip.log
git commit -m "test(fsp): memcpy 版 round-trip 与零拷贝 baseline 逐位一致"
```

---

### Task 5: 端到端训练验证(memrift_full.sh,与 baseline 对比 loss)

**Files:** 运行 `memrift_full.sh`;产出 `outputs/after_memcpy.log`。

- [ ] **Step 1: 同脚本跑改后训练**

```bash
cd /share/project/mengyc/code/memrift-flagscale
bash memrift_full.sh 2>&1 | tee outputs/after_memcpy.log
```
Expected: 正常结束,无 NaN、无 traceback,日志含 `iteration 1/...` 的 `lm loss`。

- [ ] **Step 2: 对比 iter-1 loss 与 baseline**

```bash
grep -i "lm loss" outputs/baseline_zerocopy.log | head -1
grep -i "lm loss" outputs/after_memcpy.log    | head -1
```
Expected: 两行 `lm loss` 数值近似逐位一致(fp 噪声内极小偏差,通常完全相同)。
若明显偏差:进入 superpowers:systematic-debugging —— 优先怀疑 async race / 暂存 buffer 被提前复用(检查 `record_stream(s)` 是否对两个 `exp_gpu` 都加了、H2D 的 `tmp_gpu_exp` 是否持活到 kernel 后)。

- [ ] **Step 3: (可选)曲线增强验证**

若想更稳地排查偶发 race:临时编辑 `memrift_full.sh` 把 `--train-iters 1` 改成 `--train-iters 5`,baseline 与改后各跑一次,逐 iter 对比 loss。验证完把脚本改回 `1`(勿提交临时改动)。

- [ ] **Step 4: 提交训练日志,收尾**

```bash
git add outputs/after_memcpy.log
git commit -m "test(memrift): memrift_full 端到端 loss 与零拷贝 baseline 一致,验证通过"
```

---

## 完成判据(全部满足才算端到端正常)

1. 扩展编译通过,`split`/`merge` 符号与签名不变。
2. `test_roundtrip_minimal.py` 改后仍全 `torch.equal: True`,且与零拷贝 baseline 日志一致。
3. `cudaHostGetDevicePointer` 全文 0 处;`cudaMemcpyAsync` 恰 2 处(D2H + H2D)。
4. `memrift_full.sh` 改后正常跑完,iter-1 `lm loss` 与零拷贝 baseline 近似逐位一致。
5. memrift 任何 `.py` 与 pybind 均未改动。
