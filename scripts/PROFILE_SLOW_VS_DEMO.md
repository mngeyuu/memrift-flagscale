# 为什么 FlagScale MemRift profile 比 memrift_demo 慢很多

## 1. 每「iter」的实际步数不同

| 项目 | memrift_demo | FlagScale (guanaco 默认) |
|------|--------------|---------------------------|
| 1 个 step | 1 次 forward + 1 次 backward | **8 次** get_batch + forward + backward（microbatch） |
| 2 iters | 2 次 forward/backward | 2 iters × 8 = **16 次** forward/backward |

FlagScale 里 `global_batch_size=8`、`micro_batch_size=1` → **num_microbatches=8**。  
所以 `train_iters=2` 时，实际是 **16 个 microbatch 步**，而不是 2 步。memrift_demo 的 2 步就是 2 次 forward/backward。

---

## 2. 启动阶段开销大

FlagScale 在第一次 forward 前会做：

- 分布式 / Megatron / 模型 + 优化器构建（约 4s+）
- **构建 train/valid/test 数据集**：train 最少 16、valid 800、test 800 条，建索引、shuffle 等（约 1.8s+）
- 构建 DataLoader、迭代器

memrift_demo 一般是单进程、小数据或已 tokenize 的 inputs，没有这套 dataset/dataloader 构建。

---

## 3. 数据路径不同

- **memrift_demo**：多为 `model(**inputs).loss`，inputs 已准备好（或很小 dataloader）。
- **FlagScale**：每个 microbatch 都走 `get_batch(data_iterator)` → Megatron DataLoader → IndexedDataset（guanaco 转的 .bin/.idx），首 batch 还可能触发 worker 启动。

---

## 4. 其他

- **rerun_state_machine**、Result validation 等每步有少量逻辑，但通常不是数量级差异。
- **nsys** 本身有 overhead，步数多、时间长时 profile 总时间会被拉长。

---

## 让 profile 更快、更接近 demo「2 步」的用法

在 profile 时把 **一个 iter 缩成 1 个 microbatch**，这样 2 iters ≈ 2 步，和 demo 可比：

- `data.global_batch_size=1`
- `data.micro_batch_size=1`

这样 `num_microbatches=1`，`train_iters=2` 就是 **2 次** get_batch + forward + backward，总耗时和步数都会明显下降。

可选：用 **train_mock** 做 profile（mock 数据、不建 guanaco IndexedDataset），进一步减少启动和数据准备时间。
