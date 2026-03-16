# FlagScale 升级到新版本指南

当前仓库：`origin` → https://github.com/FlagOpen/FlagScale.git，本地有 MemRift 等大量修改与未跟踪文件。按下面步骤可安全升级到上游新版本并保留你的改动。

---

## 一、升级前备份与分支

```bash
cd /share/project/mengyc/flagScale/FlagScale

# 1. 当前改动先存到分支（方便回滚）
git checkout -b mengyc-memrift-before-upgrade
git add -A
git status   # 确认要保留的都在
git commit -m "backup: current memrift + custom changes before upstream upgrade"

# 2. 回到 main，准备拉上游
git checkout main
```

---

## 二、拉取上游新版本

```bash
# 获取上游最新（或指定 tag）
git fetch origin

# 方式 A：合并上游 main 到当前 main（保留你的 commit 历史）
git merge origin/main -m "merge upstream FlagScale"

# 方式 B：想跟到某个发布版本，先看有哪些 tag
git tag -l
# 例如合并 v0.9.0：
# git merge v0.9.0 -m "merge upstream v0.9.0"
```

合并后若有冲突，git 会列出文件，按下面「三」处理。

---

## 三、解决冲突（若有）

冲突多出现在你改过的文件，例如：

- `flagscale/train/train_gpt.py`
- `flagscale/train/megatron/training/arguments_fs.py`
- `flagscale/runner/backend/backend_megatron.py`
- `flagscale/train/train.py`

处理方式：

1. 打开冲突文件，保留「你的逻辑」（尤其是 MemRift、入口、参数），把上游的新功能或修复合进去。
2. 或对单文件用“保留我方版本”再手动补上游改动：
   ```bash
   git checkout --ours -- path/to/file.py
   # 再手工把上游新增部分贴进去
   ```
3. 解决完每个冲突文件后：
   ```bash
   git add path/to/file.py
   git status   # 直到没有 "Unmerged paths"
   git commit -m "resolve merge conflicts with upstream"
   ```

---

## 四、确认未跟踪的自有代码仍在

这些是你们的新增/自有内容，**不会**被 merge 覆盖，升级后要确认仍在、能跑：

- `examples/memrift/` — MemRift 示例配置
- `flagscale/compress/memrift/` — MemRift 核心实现
- `flagscale/compress/float_split_stride_pin/` — CUDA 扩展
- `scripts/` — 自有脚本（如 `train_memrift_standalone_llama1.1b.py`、对比脚本等）

若上游新版本里**同名路径**有新增或重构，需要你手动把自有逻辑对接到新结构（例如新入口、新参数）。

---

## 五、改版本号（可选）

升级并验证通过后，可把本地版本号改成新一版，便于区分：

```bash
# 编辑 version.py，例如改为 1.1.0 或与上游对齐
# FLAGSCALE_VERSION = "1.1.0"
```

---

## 六、快速命令汇总（复制执行）

```bash
cd /share/project/mengyc/flagScale/FlagScale

# 备份当前状态
git checkout -b mengyc-memrift-before-upgrade
git add -A && git commit -m "backup before upstream upgrade" || true
git checkout main

# 拉上游并合并
git fetch origin
git merge origin/main -m "merge upstream FlagScale"

# 若有冲突：按上面「三」处理后再
# git add . && git commit -m "resolve conflicts"
```

---

## 七、若不想合并、只想对比差异

不执行 merge，只查看当前 main 与上游差多少：

```bash
git fetch origin
git log main..origin/main --oneline   # 上游多了哪些 commit
git diff main origin/main --stat      # 文件级差异
```

再决定是部分 cherry-pick 还是按上面步骤整库合并。
