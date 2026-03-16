# FlagScale 脚本目录

本目录集中存放 FlagScale 仓库下的各类脚本，由原 `tools/` 及仓库根目录部分脚本整理而来。

## 目录结构

- **根目录**：MemRift/训练/对比等 `.sh`、`.py` 入口脚本（如 `run_nsight_profile.sh`、`profile_memrift_nsight.py`、`run_tinyllama_memrift_train.sh` 等）
- **checkpoint/**：权重转换等脚本
- **codestyle/**：代码风格与 pre-commit
- **data/**：数据准备（如 guanaco）
- **datasets/**：各数据集构建脚本（llava_onevision、qwenvl、vla 等）
- **visualize/**：可视化脚本

## 使用说明

脚本均需在 **仓库根目录**（`FlagScale/`）下执行，例如：

```bash
cd /path/to/FlagScale
./scripts/run_nsight_profile.sh
./scripts/run_tinyllama_memrift_train.sh
```

文档中原来的 `./tools/xxx` 已统一改为 `./scripts/xxx`。
