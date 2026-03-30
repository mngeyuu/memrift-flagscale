#!/usr/bin/env python3
"""
在 MemRift TinyLlama 训练上启用 torch.profiler，并导出 Chrome Trace JSON。

用法（仓库根目录）:
  python scripts/run_memrift_chrome_trace.py \\
    --chrome-dir ./profiles/chrome \\
    --profile-step-start 0 --profile-step-end 3

查看:
  Chrome 打开 chrome://tracing 或 https://ui.perfetto.dev/ → Load 选择 chrome_trace_*.json

说明:
  - 与 Nsight 的 .nsys-rep 不同；本脚本生成的是 PyTorch Kineto Chrome JSON，可在同一视图里对比
    CPU 与多路 CUDA stream（含默认训练流与 MemRift 的 decode_stream 等）。
  - 代码里对权重 decode、激活 decode 打了 NVTX 名（memrift_weight_decode_stream 等），在部分
    工具链中会与 CUDA 区间一起显示。

可选环境变量:
  MEMRIFT_WEIGHT_DIR - 压缩权重目录
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="MemRift training → PyTorch Profiler Chrome Trace JSON")
    parser.add_argument(
        "--chrome-dir",
        type=str,
        default=os.path.join(ROOT, "profiles", "chrome_memrift"),
        help="输出目录（将写入 chrome_trace_0.json 等）",
    )
    parser.add_argument("--iters", type=int, default=8, help="train_iters，须 > profile_step_end")
    parser.add_argument(
        "--profile-step-start", type=int, default=0, help="与 Megatron --profile-step-start 一致"
    )
    parser.add_argument(
        "--profile-step-end", type=int, default=3, help="与 Megatron --profile-step-end 一致（在此 iteration 结束时 stop）"
    )
    parser.add_argument("--compressed-weight-dir", type=str, default=None)
    parser.add_argument("--config-name", type=str, default="train_mock")
    parser.add_argument(
        "--fast-profile",
        action="store_true",
        help="global_batch_size=1, micro_batch_size=1，缩短单步时间",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.iters <= args.profile_step_end:
        print(f"错误: train_iters ({args.iters}) 应大于 profile_step_end ({args.profile_step_end})")
        sys.exit(1)

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

    cw = args.compressed_weight_dir or os.environ.get(
        "MEMRIFT_WEIGHT_DIR", os.path.join(ROOT, "memrift_weights", "tinyllama_1b_level18")
    )

    overrides = [
        f"train.trainer.train_iters={args.iters}",
        "train.trainer.lr_warmup_iters=0",
    ]
    if cw and os.path.isdir(cw):
        overrides.append(f"train.system.memrift_compressed_weight_dir={cw}")
    else:
        print(f"警告: 压缩权重目录不存在或未定: {cw}（请设 --compressed-weight-dir 或 MEMRIFT_WEIGHT_DIR）")

    if args.fast_profile:
        overrides.extend(
            [
                "train.data.global_batch_size=1",
                "train.data.micro_batch_size=1",
            ]
        )

    with _hydra_init():
        config = compose(config_name=args.config_name, overrides=overrides)

    user_args = _get_args_megatron(config)
    entrypoint = config.experiment.task.entrypoint
    nproc = config.experiment.runner.get("nproc_per_node", 1)

    chrome_dir = os.path.abspath(args.chrome_dir)
    os.makedirs(chrome_dir, exist_ok=True)

    extra = [
        "--profile",
        "--use-pytorch-profiler",
        "--profile-step-start",
        str(args.profile_step_start),
        "--profile-step-end",
        str(args.profile_step_end),
        "--profile-chrome-trace-dir",
        chrome_dir,
    ]

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_addr=localhost",
        "--master_port=29501",
        entrypoint,
    ] + user_args + extra

    print("Chrome trace 目录:", chrome_dir)
    print("命令:", " ".join(cmd[:6]), "...")
    if args.dry_run:
        print(" ".join(cmd))
        return

    env = os.environ.copy()
    train_dir = os.path.join(ROOT, "flagscale", "train")
    _pp = [ROOT, train_dir]
    if env.get("PYTHONPATH"):
        _pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(_pp)
    # 部分环境装了损坏/不兼容的 torch 设备后端（如 flagcx），会导致 import torch 失败
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    ret = subprocess.run(cmd, cwd=ROOT, env=env)
    if ret.returncode == 0:
        print(f"\n完成。用 Chrome Tracing 或 Perfetto 打开: {chrome_dir}/chrome_trace_*.json")
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
