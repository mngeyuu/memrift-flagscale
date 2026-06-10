# 零拷贝 baseline 记录(D2H/H2D cudaMemcpyAsync 改造前)

捕获时间:2026-06-10
扩展状态:`float_split_stride_pin` 零拷贝版(`cudaHostGetDevicePointer`),env `myc`,GPU device 2(172.24.178.248)。
用途:改成 `cudaMemcpyAsync` 后逐项对比,验证数值/行为不变。

## 环境说明(与原计划差异)

原计划用 `memrift_full.sh`(Llama-3.1-8B),但本环境缺 `memrift_weights/llama31_8b_level18` 与 `models/Meta-Llama-3-8B-Instruct`,无法运行。改用本环境可用资源,新增两个单卡全路径脚本:
- `memrift_full_aquila.sh` —— **主验证目标**(Aquila2-7B,用户指定)
- `memrift_full_mistral.sh` —— 参照

均为:单卡 device 2、TE + LoRA、`--memrift-weight-enable`(权重解压走 `merge`/H2D)+ `--memrift-activation-enable --memrift-act-async`(激活压缩走 `split`/D2H、激活解压走 `merge`/H2D)、mock-data、`--train-iters 1`、`--global-batch-size 1`。

## Round-trip 单测(synthetic,model-无关)

`test_roundtrip_minimal.py` cases 1-2(case 3 因缺 Llama-8B 模型报错,与本改造无关):

| case | torch.equal | max_abs_diff | n_diff |
|------|-------------|--------------|--------|
| tiny contig bf16 (8×16) | True | 0 | 0/128 |
| 4096×4096 contig bf16 | True | 0 | 0/16777216 |

→ 零拷贝版 split→merge **bit-exact**。改后必须仍然全 True。

## 端到端训练 iter-1 loss(baseline)

| 目标 | 脚本 | lm loss | grad norm | nan iters | elapsed/iter |
|------|------|---------|-----------|-----------|--------------|
| **Aquila2-7B(主)** | `memrift_full_aquila.sh` | **1.273392E+01** | 0.924 | 0 | 32807 ms |
| Mistral-7B(参照) | `memrift_full_mistral.sh` | 1.124897E+01 | 4.510 | 0 | 36503 ms |

判定标准:改成 cudaMemcpyAsync 并重编译后,同脚本再跑,iter-1 `lm loss` 应与上表近似逐位一致(理论上完全相同,因搬运字节不变),且无 NaN。

完整日志(未入库,`outputs/` 被 gitignore):
`outputs/baseline_zerocopy_aquila.log`、`outputs/baseline_zerocopy_mistral.log`、`outputs/baseline_zerocopy_roundtrip.log`。
