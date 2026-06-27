# MemRift 原生融入 Megatron 集成设计

- 日期：2026-06-25
- 目标仓库：`/share/project/mengyc/code/memrift-flagscale`（FlagScale fork）
- 关联源码：`/share/project/mengyc/code/megatron-LM-FL`（Megatron-LM fork，可改源码）
- 路线：**B —— 自定义子类 + ModuleSpec（经 Megatron 官方扩展点装配）**

---

## 1. 背景与目标

### 1.1 现状
MemRift 是一套面向 LLM 微调的"权重 + 激活"压缩方案（float-split：指数流 zstd 无损压缩 + 符号位/尾数原样保留），核心实现约 5400 行，位于独立包 `flagscale/compress/memrift/`，通过**运行时注入**接入训练：

- `train.py` 在 `get_model()` 内调用 `inject_memrift_if_configured(model, args)`，事后遍历 `named_modules()` 用鸭子类型发现 `decoder.layers[i].self_attention.linear_qkv` 等模块，注册 `register_forward/backward_pre_hook` 做权重换入换出。
- 激活压缩用 `DecoderLayerWrapper` 把每个 decoder layer 包一层，在 `forward` 内用 `torch.autograd.graph.saved_tensors_hooks(_pack, _unpack)` 拦截/压缩激活。
- `te_ctx_patch.py` monkey-patch TransformerEngine 的 `_LayerNormMLP/_LayerNormLinear/_Linear` 的 `backward`，释放其存在 autograd ctx 上、绕过 saved_tensors_hooks 的权重引用（否则约 15GB decoder 权重在整个 backward 期间被 pin 住）。

### 1.2 目标
把 MemRift 从"独立包 + 运行时注入/打补丁"改造为**原生融入 Megatron 的模型与训练代码**：模型实例里的每一层本身就是 MemRift-aware 的层，权重流式与激活压缩是它的原生行为；去掉运行时 `register_*_hook` 外挂与（尽量）monkey-patch；`flagscale/compress/memrift` 不再作为一个"外挂包"独立出现。同时**不牺牲 FlagScale 从上游 Megatron-LM 合并升级的能力**。

---

## 2. 关键结构性约束

FlagScale 用到的 Megatron 代码分两半，可改性与升级影响完全不同：

| 部分 | 位置 | 可改性 | 升级影响 |
|---|---|---|---|
| 训练驱动（`flagscale/train/train.py`、`train_gpt.py`、`model_provider.py`、`gpt_builders.py`、vendored `flagscale/train/megatron/training`、`legacy`） | 本仓库内 | 直接编辑 | 这些本就是 FlagScale 自有/分叉文件 |
| 模型本体（`megatron.core` 的 `GPTModel`/`TransformerLayer`/TE Functions） | `megatron-LM-FL/`（Megatron-LM fork，源码可改） | 可改源码 | **改了上游也常改的文件 → 合并冲突** |

因此"原生融入"对两半含义不同：训练驱动可做到真原生；模型本体若直接改源码（路线 A）会在升级时产生合并冲突。本设计选择路线 B，用 Megatron 自带的扩展机制达到同样的原生效果而避免冲突。

---

## 3. 选定方案：B —— 子类 + ModuleSpec

### 3.1 Megatron 官方扩展点（ModuleSpec / TransformerLayerSubmodules）
Megatron 设计了"不改源码即可替换组件"的机制：

- `megatron/core/transformer/transformer_layer.py:280` `TransformerLayer.__init__(self, submodules: TransformerLayerSubmodules, ...)`：层的 `self_attention`、`mlp`、`linear_qkv` 等子模块都由一份"配方"(spec) 经 `build_module()` 实例化，并非写死。
- `megatron/core/models/gpt/gpt_layer_specs.py:343` `get_gpt_layer_with_transformer_engine_spec()` 返回 `ModuleSpec(module=TransformerLayer, submodules=TransformerLayerSubmodules(...))`。
- `flagscale/train/gpt_builders.py:48-68` 选定 `transformer_layer_spec`（已支持官方 `args.spec → import_module(args.spec)` 的字符串注入机制），再传给 `GPTModel(..., transformer_layer_spec=spec)`。

**结论**：只要提供一份不同的 spec（替换 `module=` 为自定义子类），Megatron 就用我们的层来搭模型，完全不碰 `megatron.core` 源码文件；spec 是本仓库的独立文件，升级时不参与冲突。

### 3.2 三条注入链的原生化对照

| 能力 | 现状（注入式） | 改后（原生 B） |
|---|---|---|
| 权重流式换入换出 | `register_forward/backward_pre_hook` 外挂 | `MemRiftTransformerLayer.forward` + 内部注册的 backward hook |
| 激活压缩 | `DecoderLayerWrapper` 外壳 + saved_tensors_hooks | 子类 `forward` 内 `saved_tensors_hooks(super().forward())` |
| 模型装配 | `inject_memrift_if_configured(model)` 事后改 ModuleList | `gpt_builder` 按 `--memrift-enable` 选 memrift spec |
| TE ctx 释放 | monkey-patch `_LayerNormMLP.backward` | **保留**（TE 在包内，子类够不到），收敛为版本守卫 adapter |

---

## 4. 组件设计

### 4.1 模型侧：子类 + spec（核心，新文件，放本仓库）
- `flagscale/train/memrift/layer.py`
  - `class MemRiftTransformerLayer(TransformerLayer)`：
    - `__init__(self, config, submodules, layer_number, ..., memrift_state: MemRiftLayerState)`：除原 `submodules` 外接收本层的 `MemRiftLayerState`（持有本层压缩权重组、解压器引用、跨层预取的相邻层引用）。
    - `forward(self, *args, **kwargs)`：进层 → materialize 本层权重组并写回对应 linear `.weight` → 用 `saved_tensors_hooks(_pack, _unpack)` 包 `super().forward(*args, **kwargs)`（复用官方计算）→ 出层 release 本层权重。激活压缩与权重占位（`WeightPlaceholder`）都在这层 saved_tensors_hooks 内完成。
    - backward：在 `__init__` 内 `self.register_full_backward_pre_hook(...)`（子类自身的原生成员，非外部注入）按需重换入本层权重 / 触发激活异步解压。
- `flagscale/train/memrift/spec.py`
  - `get_memrift_gpt_layer_spec(use_te, config) -> ModuleSpec`：复制官方 `get_gpt_layer_with_transformer_engine_spec` 的 `submodules`，仅把 `module=TransformerLayer` 换为 `module=MemRiftTransformerLayer`，并通过 spec 的 `params` 透传 MemRift 运行期所需信息（见 4.4）。
- 插入点：`flagscale/train/gpt_builders.py:48-68`
  ```python
  if getattr(args, "memrift_enable", False):
      transformer_layer_spec = get_memrift_gpt_layer_spec(use_te, config)
  ```
  （或直接用 `args.spec` 指向 `flagscale.train.memrift.spec:get_memrift_gpt_layer_spec`。）

### 4.2 权重加载器：复用值逻辑，只换调用方
`flagscale/compress/memrift/megatron_dynamic_loader.py` 的**值逻辑**保留，因为与"hook 还是子类"无关：
- split_zstd 解析、`merge qkv` / `merge gate-up`、`CompressedParam`、`MergedWeightGroup`、`materialize/_set_param/_clear_param`、async prefetch、CUDA 扩展 `float_split_stride_pin` 调用。

改动：
- 调用方从 hook 闭包改为 `MemRiftTransformerLayer.forward/backward`。
- `build_param_mapping`：不再全模型 `named_modules()` 扫描，改为**每层在 `__init__` 时把本层 group 绑定到本层 linear 子模块**（局部、稳，天然适配 PP/虚拟流水，无需手工算 layer offset）。
- 重构为可被子类持有的 `MemRiftLayerState`（每层一个）+ 一个进程级 `MemRiftRuntime`（持有解压线程池 `AsyncCompressor`、跨层预取注册表、CUDA ext 句柄）。

### 4.3 激活压缩：去外壳，逻辑内移
`flagscale/compress/memrift/activation_compression.py` 的 `_pack/_unpack`、`PlaceHolderToken`、`WeightPlaceholder`、`_should_compress_activation` 全保留。删除 `DecoderLayerWrapper` 外壳——其 `saved_tensors_hooks` 包裹与 token 生命周期（`_bwd_pre_hook` 异步解压、`_bwd_hook` 清理）移进 `MemRiftTransformerLayer`。开关由 `--memrift-activation-enable` 控制；LoRA 适配器 storage 的 `skip_storage_ptrs` 在建模阶段从 `requires_grad` 参数收集。

### 4.4 PEFT/LoRA 信息透传
当前依赖 `args.peft_type`、`args.lora_target_modules` 计算 `allowed_targets`（只压被 LoRA 包的 base 权重）与激活跳过集合。改后由 `get_memrift_gpt_layer_spec` 在装配时从 `args` 读取并写入 `MemRiftLayerState`，供子类使用。

### 4.5 TE ctx 释放（必要的 monkey-patch，诚实记录）
`flagscale/compress/memrift/te_ctx_patch.py` **保留**：TE 的 fused backward 在 `transformer_engine` 包内部、把权重存在 autograd ctx 上、绕过 saved_tensors_hooks，子类无法原生接管。改进：
- 从"无条件 patch"收敛为**带 TE 版本检测的 adapter**；patch 失败时给出清晰报错（指明 TE 版本与受影响的 Function），而非静默泄漏 ~15GB。
- 在 `MemRiftRuntime` 初始化时调用一次，幂等。
- 这是路线 A 与 B 都绕不开的一处必要补丁，单列于风险。

### 4.6 训练驱动侧（in-repo，真原生）
- `flagscale/train/train.py`：
  - 删除 `get_model()` 内 `inject_memrift_if_configured` 的 try/except 外挂块（约 1287-1296 行）。
  - `_get_memrift_activation_context(model)`（约 1731 行）退化为不再需要（压缩已在子类内）；保留 `nullcontext` 兼容或一并移除调用。
  - profiler 的 `on_iter_start/end`、收尾 `release_all_layers()` 保留，但改为通过 `MemRiftRuntime` 单例访问，替代当前的 `_LOADERS` / `_PTR2GROUP` 全局散落状态。
  - 移除 `MEM_PROBE` 诊断块对 `train_hooks._LOADERS`、`megatron_dynamic_loader._PTR2GROUP` 的内部依赖（或迁移到 `MemRiftRuntime` 的查询接口）。
- `flagscale/train/megatron/training/arguments_fs.py`：`--memrift-*` 已是原生注册；新增 `--memrift-on-error {fallback,raise}`（默认 `fallback`，见 4.7）；顺带修复 `--memrift-prefetch-layers` 实际未生效（代码内硬编码 `span=1`）的已知 bug——要么让参数真正驱动预取跨度，要么删参数并在文档注明固定为 1。

### 4.7 启用失败的 Fallback 策略
**需求**：MemRift 启用失败时不应让整个训练崩溃，应可回退到普通 Megatron 训练（无 memrift）。

**边界（必须先讲清）**：
- Fallback 只能发生在**建模/初始化阶段（建模前）**。一旦 GPTModel 用 `MemRiftTransformerLayer` + `CompressedParam` 占位建好、进了计算图，训练中途无法再回退（权重已是压缩态）。
- 因此回退决策必须在 **spec 选择之前**完成，且是**全模型一致的 all-or-nothing**——不允许出现"部分层 memrift、部分层普通"的半成品模型。

**设计：建模前 preflight + 统一回退**
- 在 `gpt_builders.py` 选 spec 之前调用 `memrift_preflight(args) -> MemRiftRuntime | None`，集中执行所有可能失败的检查并构建运行期对象：
  - CUDA 扩展 `float_split_stride_pin` 可用性；`zstandard` 可用性；
  - 压缩目录存在与 index 校验（TP/PP 分片目录解析）；
  - TE ctx patch 应用（见 4.5）；
  - `MemRiftRuntime` / 各层 `MemRiftLayerState` 构建。
- preflight **成功** → 选 `get_memrift_gpt_layer_spec(...)`，把 runtime 传入 spec。
- preflight **抛异常** → 按 `--memrift-on-error` 处理：
  - `fallback`（默认）：捕获异常，**改用官方 `get_gpt_layer_with_transformer_engine_spec(...)` 跑普通训练**，并在 rank0 打**响亮的多行告警 banner**（含失败原因、堆栈摘要、"本次运行已禁用 MemRift，按普通 Megatron 训练"），同时在训练起始的配置摘要里标注 `memrift: DISABLED (fallback)`，确保绝不静默。
  - `raise`：直接抛出，fail-fast（用于必须确保压缩生效的场景）。

**不可回退的子情况（必须 fail-fast，即使 `fallback`）**：
- 若压缩目录是**唯一权重来源**（即没有通过普通 `--load` 提供 base 权重，模型权重只能从 compressed dir 恢复），退普通 Megatron 将**无权重可加载**。此时 preflight 必须识别并 fail-fast，报错指明"compressed dir 为唯一权重源，无法回退到普通训练；请提供 --load 或修复 memrift 启用"。

**已知代价（写入告警文案）**：
- 若模型本来只靠压缩才装得下，回退普通后很可能立即 OOM。告警需明确提示这一点，避免用户误以为"回退=安全"。

**运行时（建模后）失败**：不在 fallback 范围，只能 fail-fast 并输出诊断（哪一层、哪个 group、解压/CUDA ext 错误），不静默吞掉。

### 4.8 不受影响 / 二阶段
- 离线压缩工具链（`offline_comp/prepare_weight*.py`、`compress_megatron_tp.py`）产出 compressed dir，与运行时解耦，**不改**。
- 推理路径 `inject_memrift_for_inference`（`generate_gpt.py`）可同法用一份"推理 spec"改造（forward-only、无激活压缩），列为**第二阶段**，本设计先聚焦训练。

---

## 5. 文件改动清单

新增（本仓库）：
- `flagscale/train/memrift/layer.py` — `MemRiftTransformerLayer`
- `flagscale/train/memrift/spec.py` — `get_memrift_gpt_layer_spec`
- `flagscale/train/memrift/runtime.py` — `MemRiftRuntime` / `MemRiftLayerState`（封装 loader 值逻辑 + 解压池 + 跨层预取注册表）+ `memrift_preflight(args)`（集中检查并构建 runtime，供 fallback 决策）

改写（复用值逻辑）：
- `flagscale/compress/memrift/megatron_dynamic_loader.py` — 拆出可被子类持有的 state，去掉 hook 安装代码
- `flagscale/compress/memrift/activation_compression.py` — 删 `DecoderLayerWrapper`，保留 pack/unpack 原语

改动（训练驱动）：
- `flagscale/train/gpt_builders.py` — 加 memrift spec 选择分支
- `flagscale/train/train.py` — 删注入块、改用 `MemRiftRuntime`
- `flagscale/train/megatron/training/arguments_fs.py` — 修 prefetch 参数 bug

不变：
- `flagscale/compress/float_split_stride_pin/`（CUDA 扩展）
- `flagscale/compress/memrift/offline_comp/`（离线压缩）
- `flagscale/compress/memrift/te_ctx_patch.py`（仅收敛为版本守卫）

`megatron-LM-FL/` 源码：**零改动**（路线 B 的核心收益）。

---

## 6. 落地顺序（迁移路径）

1. 抽 `MemRiftRuntime` / `MemRiftLayerState`：把 `megatron_dynamic_loader` 的值逻辑搬进 runtime，保持现有 hook 路径仍能跑（绿）。
2. 写 `MemRiftTransformerLayer` + `spec`，先只接权重流式（不含激活），用 `args.spec` 手工指向，单卡 1 层 1 iter 验证 loss 与 baseline 一致。
3. 把激活压缩从 `DecoderLayerWrapper` 移进子类，验证显存曲线与现状一致。
4. 在 `gpt_builders` 接 `--memrift-enable` 自动选 spec；加 `memrift_preflight` + fallback 分支（4.7）；删 `train.py` 注入块。
5. 修 prefetch 参数 bug、收敛 TE patch、清理 `_LOADERS/_PTR2GROUP` 全局态。
6. 全路径回归（TP>1、PP>1、LoRA）+ fallback 路径回归（故意触发启用失败，验证退普通训练且告警可见；验证"压缩目录为唯一权重源"时 fail-fast）。

每步保持可回退：旧注入路径在第 4 步前不删。

---

## 7. 风险与待决

1. **TE fused backward**：必留 monkey-patch（`te_ctx_patch`），需 TE 版本守卫与失败即报错。子类无法完全原生接管 TE 包内 backward。
2. **构建顺序 / 显存峰值**：spec 装配发生在 GPTModel 建模时，而压缩权重的"加载 → 释放原权重"原本在 PEFT 之后。需重排为**建模即用 `CompressedParam` 占位**，避免先分配满精度权重再释放造成峰值。这是从"事后注入"转为"建模即原生"最需要验证的一点。
3. **跨层预取**：现有 hook 闭包能引用"下一层"对象做预取；子类内只看得到自己。需 `MemRiftRuntime` 维护 layer i → i±1 的 `MemRiftLayerState` 注册表，供子类在 forward/backward 触发相邻层异步解压。
4. **PEFT/LoRA 假设**：`allowed_targets` 映射、`skip_storage_ptrs` 依赖 `args.peft_type/lora_target_modules`，需在 spec 装配时一次性读入 state。非 LoRA 全量微调场景的行为需明确（当前主要为 LoRA 冻结 base）。
5. **PP / 虚拟流水**：每层局部绑定后，layer 全局/局部索引由 GPTModel 自然处理，预期比现状手工算 offset 更稳，但需在 PP>1、VPP 下实测。
6. **prefetch 参数**：`--memrift-prefetch-layers` 当前不生效；本次明确其语义（驱动 span 或废弃固定为 1）。
7. **Fallback 一致性**：回退决策必须在建模前完成且全模型一致；preflight 要尽量把"会失败的事"都前移到建模前，避免失败漏到建模中途产生半成品模型。"压缩目录为唯一权重源"的子情况需可靠识别并 fail-fast。

---

## 8. 验收标准

- 单卡、`train-iters=1`、`memrift_full.sh` 场景：开 memrift 子类路径与 baseline（无 memrift）首步 loss 在数值容差内一致。
- 显存：开 memrift 后 decoder 权重常驻显存显著下降，backward 期间无 ~15GB 权重泄漏（沿用现有 `MEM_PROBE` 探针口径）。
- `megatron-LM-FL/` 工作树 `git diff` 为空（验证零源码改动、升级无冲突）。
- TP=2、PP=2、LoRA 三组配置回归通过。
- Fallback：人为制造启用失败（如压缩目录缺失 / CUDA ext 不可用）时，`--memrift-on-error=fallback` 能继续以普通 Megatron 训练并打印可见告警 banner；`--memrift-on-error=raise` 能 fail-fast；"压缩目录为唯一权重源"场景即使 `fallback` 也能 fail-fast 且报错清晰。

---

## 9. 不在本设计范围（Out of Scope）

- 推理路径（`inject_memrift_for_inference`）的子类化改造——二阶段。
- 离线压缩工具链与 split_zstd 格式本身的任何改动。
- 压缩算法/压缩率优化。
- 路线 A（直改 `megatron-LM-FL` 源码）的具体实现——已评估并因升级冲突放弃，仅作对照保留于第 2 节。
