# GPU 显存高但利用率为 0 的原因与排查

## 现象

训练时 GPU 显存占用高（如 7GB+），但 **GPU 利用率为 0**，说明显存已被占用（模型/权重/激活）但 **GPU 几乎没有在执行 kernel**，即主线程长时间卡在 **CPU 侧**，没有持续向 GPU 提交新任务。

---

## 原因 1：每层权重的同步 materialize 阻塞主线程（最主要）

**位置**: `flagscale/compress/memrift/megatron_dynamic_loader.py` 的 forward_pre / backward_pre hook。

**流程**（每层都会发生）：

1. **forward_pre** 里对当前层调用 `_materialize_group(group, sync=True)`。
2. `sync=True` 的 materialize 在 **主线程** 上依次做：
   - **zstd 解压 exp**（纯 CPU，主线程阻塞）；
   - 再在 GPU 上做 merge；
   - 最后 `ev.synchronize()`，主线程等 GPU merge 完成。
3. 之后才执行当前层的 forward。

因此：在 **zstd 解压** 这段时间里，主线程在忙 CPU，**没有新的 GPU 任务被提交**，GPU 就空转，表现为 util=0。  
解压越慢、层数越多，GPU 空转比例越高。

**Prefetch 的作用与局限**：

- 在上一层的 forward_pre 里会 **异步** 提交下一层的 `materialize_async()`（在 decode 线程池里解压 + merge）。
- 到当前层时，若 prefetch 的 future 已 done，直接取结果，几乎不阻塞。
- 若 **解压耗时 > 上一层 forward 耗时**，下一层进来时 future 常未完成，会走 **fallback 同步 materialize**（主线程自己解压），GPU 再次长时间空闲。

所以：**只要经常命中“prefetch not ready”的同步路径，就会出现显存高、util 为 0。**

**已按 memrift_demo 方式改进**：在 `CompressedParam.materialize()` 中，当本层有 prefetch 但未完成时，**不再回退到主线程同步解压**，改为 **一直等待** `future.result()`。解压与 merge 保持在线程池 / h2d_stream，主线程等待时 GPU 仍可执行 merge，解压与训练在时间上重叠，减轻 util=0。

---

## 原因 2：DataLoader 取 batch 阻塞主线程

**位置**: `flagscale/train/megatron/training/utils.py` 的 `get_batch_on_this_tp_rank()` → `data = next(data_iterator)`。

- 若 **num_workers=0**（或很小），`next(data_iterator)` 在 **主进程** 里做 `dataset.__getitem__()`（读 IndexedDataset 等），主线程会在这里卡住。
- 这段时间内不会进入 forward，GPU 没有新 work，util 也会是 0。
- Megatron 默认 `num_workers=2`；若数据在慢盘或预处理重，仍可能成为瓶颈。

---

## 原因 3：每 step 开始时先等 batch 再算

**位置**: `flagscale/train/train_gpt.py` 的 `forward_step()`。

```text
timers('batch-generator').start()
tokens, labels, ... = get_batch(data_iterator, ...)  # 这里可能阻塞
timers('batch-generator').stop()
output_tensor = model(...)
```

若 `get_batch()` 耗时很长（数据加载慢或 num_workers 不足），则每个 step 开头 GPU 都在等 batch，util 会偏低甚至为 0。

---

## 排查步骤

1. **看 prefetch 是否经常未就绪**  
   运行前设置：
   ```bash
   export MEMRIFT_TRACE=1
   ```
   若日志里大量出现 `materialize: prefetch not ready (layer=...), fallback to sync`，说明 **解压跟不上**，主线程频繁做同步 materialize，GPU 空转。
2. **看 batch-generator 是否占大头**  
   若开了 Megatron timers，看 `batch-generator` 在每 step 里占比。若接近或超过 50%，说明 **数据加载** 在拖累 GPU。
3. **Nsight Systems**  
   - CPU 时间线：主线程是否长时间停在 zstd/decompress 或 DataLoader 相关调用；  
   - GPU 时间线：是否大段空白、只有零星 kernel（对应“显存高、util 0”）。

---

## 建议修改（配置与参数）

| 项目 | 作用 | 建议 |
|------|------|------|
| **data.num_workers** | 多进程预取 batch，减少主线程在 get_batch 上的阻塞 | 在 MemRift 的 data 配置里设为 **4 或 8**（例如 `data.num_workers: 4`） |
| **memrift_prefetch_layers** | 多预取几层，提高“下一层解压完成”的概率，减少 fallback 同步 | 从 1 提到 **2**（或 3），在 system 下配置 |
| **memrift_decode_pool_workers** | 解压线程池大小，并行解压更多层 | 可设为 **4 或 8**（已有则适当调大） |

**YAML 示例**（在 `examples/memrift/conf/train/` 下对应 yaml 的 `data` / `system` 中增加或覆盖）：

```yaml
# data 段
data:
  num_workers: 4   # 与 DataLoader 的 num_workers 对应，减少 get_batch 阻塞

# system 段（MemRift 相关）
system:
  memrift_prefetch_layers: 2      # 预取 2 层，降低“prefetch not ready”概率
  memrift_decode_pool_workers: 8 # 解压线程数，可按 CPU 核数调整
```

---

## 小结

- **显存高、util=0** 的本质是：**主线程大量时间花在 CPU 上（解压或取数据），没有持续给 GPU 派活**。
- 最主要的是 **MemRift 每层 sync materialize 里的 zstd 解压**；prefetch 跟不上时会退化成主线程同步解压，GPU 空转。
- 次要的是 **DataLoader num_workers 不足** 导致 `get_batch()` 阻塞。
- 通过 **增大 num_workers、prefetch_layers、decode_pool_workers** 并配合 **MEMRIFT_TRACE=1** 和 Nsight 排查，可以定位并缓解问题。
