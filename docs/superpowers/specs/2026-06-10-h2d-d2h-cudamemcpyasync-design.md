# 设计:零拷贝传输路径改为 cudaMemcpyAsync(D2H + H2D)

日期:2026-06-10
目标文件:`flagscale/compress/float_split_stride_pin/float_split_stride_pin.cu`
关联模块:memrift(`flagscale/compress/memrift/*`)

## 1. 背景与目标

`float_split_stride_pin` CUDA 扩展为 memrift 提供 fp32/bf16 的指数/尾数拆分(`split`)与重建(`merge`),用于压缩权重与激活的存储。当前两个方向的 host↔device 传输都走**零拷贝**:kernel 通过 `cudaHostGetDevicePointer` 拿到 pinned host 内存的设备映射地址,直接对 host 内存做细粒度 PCIe 读/写,没有显式 memcpy。

零拷贝在大块传输上吞吐低。本设计把两个方向都改为 **GPU 暂存 buffer + `cudaMemcpyAsync`(整块 DMA)**:

- **D2H(`split` / `pack_tensor`)**:kernel 写本地显存暂存 → `cudaMemcpyAsync` 回拷 pinned host。
- **H2D(`merge` / `unpack_tensor`)**:`cudaMemcpyAsync` 把 pinned host 的 exp 拷到显存暂存 → kernel 从显存读。

**核心约束:对 memrift 透明。** 两个函数的 pybind 名、签名、返回值全部保持不变,memrift 侧所有 Python 调用点零改动。

## 2. 现状(production 文件)

- `split` = `pack_tensor`(行 314-363):零拷贝 D2H,关键在行 332-334
  `cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0)`,kernel 写 `dev_ptr`(host 映射)。返回 `{exp_host, sm}`。
- `merge` = `unpack_tensor`(行 379-459):零拷贝 H2D,关键在行 ~400-415,pinned-host 分支
  `cudaHostGetDevicePointer(&exp_dev_ptr, exp.data_ptr(), 0)`(带一条 `.to()` fallback)。返回单 tensor。
- pybind(行 465-483):仅 `split` / `merge` / `acquire_pin` / `release_pin` / `release_cuda`。无 `split_copy`。
- 文件为 UTF-8 编码(注:仓库根目录 `backup.cu` 是 ISO-8859 乱码副本,**不是**本次目标)。

### memrift 调用约定(必须保持兼容)

- `split`:8 处调用,全部 `cpu_exp, sm_bits = fs_sp.split(t, stream)`(严格 2 元解包)。
  位置:`async_compressor.py`(2)、`vllm_loader.py`、`megatron_tp_hooks.py`、`offline_comp/{prepare_weight,prepare_weight_tp2,compress_megatron_tp×2}`。
- `merge`:7 处调用,`rst = fs_sp.merge(exp, sm, shape, stride, offset, dtype, stream)`(返回单 tensor)。
  位置:`async_compressor.py`(3)、`vllm_loader.py`(2)、`megatron_dynamic_loader.py`。
- GPU 侧 buffer 通过 `release_cuda` 显式归还自定义 `Pool`。

## 3. 方案决策(已与用户确认)

| 决策点 | 选择 |
|--------|------|
| 方向 | 零拷贝 → `cudaMemcpyAsync`,D2H 与 H2D 都改 |
| D2H 风格 | **原地改 `split`,保持 2 元返回**(不新增 `split_copy`) |
| H2D 风格 | **原地改 `merge` 内部**,签名/返回不变 |
| H2D 机制 | 显式 `cudaMemcpyAsync` + 自定义 `Pool` 暂存 buffer(与 D2H 对称) |
| 调用点 | memrift 15 处全部零改动,透明生效 |

明确**不采用**的替代:`split_copy` 3 元返回 + 改 8 个调用点(更侵入);`exp.to(non_blocking=True)`(更短但不复用自定义 Pool、对 stream/生命周期控制不显式)。

## 4. 详细改动

### 改动 A — `pack_tensor`(`split`,D2H,行 314-363)

删除 host 映射相关三行(`host_ptr` / `dev_ptr` / `cudaHostGetDevicePointer`),改为申请 GPU 暂存 `exp_gpu`;kernel 的 exp 输出实参由 `dev_ptr` 改为 `exp_gpu.data_ptr<uint8_t>()`;kernel 后整块异步回拷到 pinned host。返回值不变。

```cpp
auto exp_host = Pool::inst().get(N, at::kByte, -1, true, raw);          // 不变:pinned host
int64_t sm_elems = (t.scalar_type()==at::kFloat) ? N*3 : N;
auto sm  = Pool::inst().get(sm_elems, at::kByte, dev_idx, false, raw);  // 不变
auto exp_gpu = Pool::inst().get(N, at::kByte, dev_idx, false, raw);     // 新增:GPU 暂存
auto ix  = make_indexer<4>(t);

// 四个 kernel 分支的 exp 实参:dev_ptr → exp_gpu.data_ptr<uint8_t>()
... pack_*_kernel<<<grid,256,0,raw>>>( ..., exp_gpu.data_ptr<uint8_t>(),
                                       sm.data_ptr<uint8_t>(), N);

cudaError_t err = cudaMemcpyAsync(exp_host.data_ptr<uint8_t>(),         // dst: pinned host
                                  exp_gpu.data_ptr<uint8_t>(),          // src: GPU
                                  static_cast<size_t>(N),
                                  cudaMemcpyDeviceToHost, raw);          // 同一外部 stream
TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync D2H failed: ", cudaGetErrorString(err));
exp_gpu.record_stream(s);
Pool::inst().put(std::move(exp_gpu), /*pinned=*/false, raw);            // 暂存归还自定义池
return {exp_host, sm};
```

### 改动 B — `unpack_tensor`(`merge`,H2D,行 379-459)

只改 pinned-host 分支(`exp.is_cuda()` 路径不变)。删除 `cudaHostGetDevicePointer` 及其 `.to()` fallback,改为 GPU 暂存 + 异步 H2D。复用已有的 `std::optional<at::Tensor> tmp_gpu_exp` 持活暂存 buffer 到 kernel 结束。

```cpp
if (exp.is_cuda()) {
    exp_dev_ptr = exp.data_ptr<uint8_t>();                 // 不变:exp 已在 GPU,无传输
} else {
    TORCH_CHECK(exp.is_pinned(), "exp on CPU must be pinned for async H2D copy");
    const int64_t E = exp.numel();
    auto exp_gpu = Pool::inst().get(E, at::kByte, dev_idx, /*pinned=*/false, raw);
    cudaError_t err = cudaMemcpyAsync(exp_gpu.data_ptr<uint8_t>(),   // dst: GPU
                                      exp.data_ptr<uint8_t>(),       // src: pinned host
                                      static_cast<size_t>(E),
                                      cudaMemcpyHostToDevice, raw);   // 同一外部 stream
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync H2D failed: ", cudaGetErrorString(err));
    exp_gpu.record_stream(s);
    exp_dev_ptr = exp_gpu.data_ptr<uint8_t>();
    tmp_gpu_exp.emplace(std::move(exp_gpu));               // 持活到 kernel 结束
}
```

### 不改动

pybind(不新增任何函数)、`Pool`、所有 CUDA kernel、`__init__.py`、所有 Python 调用方。

## 5. 正确性要点

- **stream 顺序**:memcpy 与 kernel 都 enqueue 在同一条外部 `raw` stream → 严格有序,无需额外 event。
  - D2H:kernel(写 exp_gpu)→ memcpy(读 exp_gpu)→ 完成。
  - H2D:memcpy(写 exp_gpu)→ kernel(读 exp_gpu)。
- **生命周期**:
  - D2H 暂存 `exp_gpu`:`record_stream(s)` + `Pool::put(..., raw)`(put 内部记录 event 守护)→ 回拷完成前不被复用。
  - H2D 暂存 `exp_gpu`:`record_stream(s)` + `tmp_gpu_exp` 函数作用域持活 → kernel 读完前不析构、不复用。
- **数值不变性**:本改动只换传输机制,搬运的 exp/sm 字节与零拷贝完全相同,kernel 输入逐位一致 → 输出逐位一致。这是第 6 节 loss 对比验证的理论依据。
- **`is_pinned()` 校验保留**:异步 H2D/D2H 要求 host 端为 pinned。
- **兼容性**:`split` 返回 2 元、`merge` 返回单 tensor,memrift 15 处调用点全部透明生效。

## 6. 验证计划

前置:真机执行需 SSH 到 `172.24.178.248`,`conda activate myc`;改后重编译扩展
`cd flagscale/compress/float_split_stride_pin && pip install -e .`。

### 6.1 单元 round-trip(数值正确性)
- `test_roundtrip_minimal.py`、`test_decompress_accuracy.py`:split→merge 数值一致。

### 6.2 小流程冒烟
- 跑一个 memrift 小流程(如 `run_memrift_weight_only_test.sh` 或 `run_memrift_dbg.sh`)确认压缩/解压链路不崩。

### 6.3 端到端训练验证(单卡全路径 + 零拷贝 baseline 对比)
**配置**:单卡 LLaMA-3.1-8B 全路径(`memrift_full.sh` 或 `run_ablation_llama31_8b.sh` 的"权重+激活"档),5 iters,mock data,需现有 `memrift_weights/llama31_8b_level18` 压缩权重。该配置同时压到:
- 新 `split`(激活压缩 D2H)
- 新 `merge`(权重解压 + 激活解压 H2D)

**判定标准——与零拷贝 baseline 逐 iter 对比 loss**:
1. **改动前**:在当前零拷贝版本上跑一次,保存逐 iter loss(baseline 日志)。
2. **改动后**:重编译扩展,同配置、同随机种子再跑一次。
3. **对比**:逐 iter loss 应与 baseline 近似逐位一致(允许 fp 累加噪声内的极小偏差);无 NaN、无崩溃、loss 正常下降。

> 注:由于第 5 节"数值不变性",理论上 loss 应当几乎逐位相同。若出现明显偏差,说明改动引入了 race / 生命周期 / 传输错误,需回到 systematic-debugging。

## 7. 范围外(本设计不含)

- `backup.cu`(根目录 ISO-8859 副本)不在本次范围。
- 不重新压缩离线权重:旧压缩权重的字节格式与新 `split` 输出完全一致,`merge` 可直接读。
- 不改 memrift 任何 Python 文件。
- TP=2 分片路径验证为可选扩展,非本次必需。

## 8. 实现步骤次序(供 writing-plans 参考)

1. **先**跑零拷贝 baseline 训练,存 loss 日志(否则改完无从对比)。
2. 改 `pack_tensor`(改动 A)。
3. 改 `unpack_tensor`(改动 B)。
4. 重编译扩展。
5. 单元 round-trip + 小流程冒烟。
6. 端到端训练,与 baseline 对比 loss。
