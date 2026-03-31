#!/usr/bin/env python3
"""
使用 Nsight Systems profile MemRift + FlagScale 训练（Hydra 配置可选 TinyLlama、Llama-3.1-8B 等）。

在 FlagScale 仓库根目录执行:
  python scripts/profile_memrift_nsight.py [--iters 5] [--output memrift_profile]
  python scripts/profile_memrift_nsight.py --config-name train_llama31_8b_seq4096_bs1_memrift_weight_only \\
      --compressed-weight-dir ./memrift_weights/llama31_8b_nvcomp_ans --iters 3 --fast-profile --python-backtrace-cuda

可选: --dry-run 只打印命令不执行。

默认采集: GPU (--trace=cuda,nvtx,osrt)、CPU (--sample=cpu，可用 --no-cpu-sample 关闭)、
显存曲线 (--cuda-memory-usage=true，可用环境变量 CUDA_MEMORY_USAGE=0 关闭)。

依赖: 已安装 nsys (Nsight Systems), 且已准备压缩权重目录。
"""
import os
import sys
import argparse
import subprocess

# FlagScale 仓库根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="Profile MemRift + FlagScale training with Nsight Systems")
    parser.add_argument("--iters", type=int, default=1, help="训练步数 (profile 时 1~2 步即可，省时间)")
    parser.add_argument("--output", type=str, default="memrift_llama11b_nsight", help="nsys 输出文件名 (不含扩展名)")
    parser.add_argument("--compressed-weight-dir", type=str, default=None,
                        help="压缩权重目录，默认用 config 里的 memrift_compressed_weight_dir")
    parser.add_argument("--config-name", type=str, default="train_mock",
                        help="Hydra 配置名：train_mock（mock 数据）或 train_guanaco（真实数据 guanaco）")
    parser.add_argument("--fast-profile", action="store_true",
                        help="profile 加速：global_batch_size=1, micro_batch_size=1，使 2 iters = 2 步（与 demo 可比），显著缩短耗时")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    parser.add_argument(
        "--no-cpu-sample",
        action="store_true",
        help="关闭 nsys 周期性 CPU 采样（默认开启，用于 CPU 侧热点）",
    )
    parser.add_argument(
        "--python-backtrace-cuda",
        action="store_true",
        help="开启 --python-backtrace=cuda，在 CUDA API 上关联 Python 栈（开销更大）",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        dest="hydra_overrides",
        metavar="KEY=VALUE",
        help="追加 Hydra override，可重复多次（如 train.model.tokenizer_path=/path）",
    )
    args = parser.parse_args()

    # 用 Hydra 加载 MemRift 配置（train_mock 或 train_guanaco）
    try:
        from hydra import compose, initialize_config_dir
        _hydra_init = lambda: initialize_config_dir(config_dir=config_path, version_base=None)
    except ImportError:
        from hydra import compose, initialize_config_path
        _hydra_init = lambda: initialize_config_path(config_path=config_path, version_base=None)
    from flagscale.runner.runner_train import _get_args_megatron

    config_path = os.path.join(ROOT, "examples", "memrift", "conf")
    if not os.path.isdir(config_path):
        print(f"错误: 配置目录不存在 {config_path}")
        sys.exit(1)

    overrides = [
        f"train.trainer.train_iters={args.iters}",
        "train.trainer.lr_warmup_iters=0",
    ]
    if args.compressed_weight_dir:
        overrides.append(f"train.system.memrift_compressed_weight_dir={args.compressed_weight_dir}")
    # 加速 profile：1 iter = 1 个 microbatch，与 memrift_demo 的「2 步」一致，避免 2 iters × 8 = 16 步
    if getattr(args, "fast_profile", False):
        overrides.extend([
            "train.data.global_batch_size=1",
            "train.data.micro_batch_size=1",
        ])
        print("[profile] --fast-profile: global_batch_size=1, micro_batch_size=1 (2 iters = 2 steps)")

    if args.hydra_overrides:
        overrides.extend(args.hydra_overrides)

    with _hydra_init():
        config = compose(config_name=args.config_name, overrides=overrides)

    user_args = _get_args_megatron(config)
    entrypoint = config.experiment.task.entrypoint
    nproc = config.experiment.runner.get("nproc_per_node", 1)

    # Nsight Systems 命令：直接包住 torchrun，这样 profile 的是实际训练进程
    # 显存时间线：--cuda-memory-usage=true（有开销，可通过 CUDA_MEMORY_USAGE=0 关闭）
    cuda_mem = os.environ.get("CUDA_MEMORY_USAGE", "1") == "1"
    master_port = os.environ.get("MASTER_PORT", os.environ.get("NSIGHT_MASTER_PORT", "29500"))
    nsys_base = [
        "nsys",
        "profile",
        "--stats=true",
        "--trace=cuda,nvtx,osrt",
        "--show-output=true",
        f"--output={args.output}",
        "--force-overwrite=true",
    ]
    if not args.no_cpu_sample:
        nsys_base.append("--sample=cpu")
    if args.python_backtrace_cuda:
        nsys_base.append("--python-backtrace=cuda")
    if cuda_mem:
        nsys_base.append("--cuda-memory-usage=true")
    nsys_cmd = nsys_base + [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_addr=localhost",
        f"--master_port={master_port}",
        entrypoint,
    ] + user_args

    print("Working directory:", ROOT)
    print("Command:", " ".join(nsys_cmd[:8]), "...", f"({len(user_args)} train args)")
    if args.dry_run:
        print("\n[DRY-RUN] 完整命令:")
        print(" ".join(nsys_cmd))
        return

    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    env["PYTHONPATH"] = (ROOT + os.pathsep + env.get("PYTHONPATH", "")).rstrip(os.pathsep)
    ret = subprocess.run(nsys_cmd, cwd=ROOT, env=env)
    if ret.returncode == 0:
        print(f"\nProfile 已保存: {args.output}.nsys-rep")
        print("用 Nsight Systems 打开该文件查看时间线。")
        if cuda_mem:
            print("显存: 在 Timeline 中展开 CUDA Memory 轨道查看分配曲线与峰值；或运行:")
            print(f"  nsys stats --report cuda_gpu_mem_size_sum {args.output}.nsys-rep")
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
