#!/usr/bin/env python
"""
全量基准测试：seq × mode matrix
配置:
  mode=lora:      仅 LoRA（memrift_enable=false）
  mode=weight:    权重压缩（memrift_weight_enable=true, act=false）
  mode=weight_act: 权重+激活压缩（weight=true, act=true）
seq: 512, 2048, 4096

使用方式:
  python scripts/bench_full_matrix.py          # 全部跑
  python scripts/bench_full_matrix.py --seq 512 2048
  python scripts/bench_full_matrix.py --modes lora weight
"""
import os, sys, json, time, subprocess, argparse, re
from pathlib import Path

FLAGSCALE_DIR = Path(__file__).parent.parent.resolve()
WEIGHT_DIR = "/share/project/mengyc/code/memrift_v1/weight_comp/test_nvcomp_ans"
TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOG_FILE = FLAGSCALE_DIR / "outputs/memrift_example/logs/host_0_localhost.output"

MODES = {
    "lora":       dict(memrift_enable=False, weight_enable=False, act_enable=False),
    "weight":     dict(memrift_enable=True,  weight_enable=True,  act_enable=False),
    "weight_act": dict(memrift_enable=True,  weight_enable=True,  act_enable=True),
}


def run_one(seq: int, mode: str, train_iters: int = 8, warmup_iters: int = 2) -> dict:
    cfg = MODES[mode]
    print(f"\n{'='*60}")
    print(f"  seq={seq}  mode={mode}")
    print(f"{'='*60}")

    # clear old log and checkpoints
    ckpt_dir = FLAGSCALE_DIR / "outputs/memrift_example/checkpoints"
    pid_file = FLAGSCALE_DIR / "outputs/memrift_example/logs/pids/host_0_localhost.pid"
    # Remove entire checkpoints dir to avoid stale latest_checkpointed_iteration.txt
    subprocess.run(f"rm -rf {ckpt_dir} && mkdir -p {ckpt_dir}", shell=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("")
    if pid_file.exists():
        pid_file.unlink()

    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    overrides = [
        f"train.system.memrift_enable={'true' if cfg['memrift_enable'] else 'false'}",
        f"train.system.memrift_weight_enable={'true' if cfg['weight_enable'] else 'false'}",
        f"train.system.memrift_activation_enable={'true' if cfg['act_enable'] else 'false'}",
        f"train.system.memrift_compressed_weight_dir={WEIGHT_DIR}",
        f"train.model.tokenizer_path={TOKENIZER}",
        f"train.model.tokenizer_model={TOKENIZER}",
        f"train.trainer.train_iters={train_iters}",
        f"train.trainer.lr_warmup_iters={warmup_iters}",
        "++train.trainer.eval_interval=100000",
        "++train.trainer.eval_iters=0",
        "train.data.micro_batch_size=1",
        "train.data.global_batch_size=1",
        f"train.model.seq_length={seq}",
        f"train.model.max_position_embeddings={max(seq, 2048)}",
        "train.system.logging.log_interval=1",
        # Disable checkpoint saving during benchmark to prevent stale ckpt files
        "++train.system.checkpoint.save_interval=99999",
        "action=run",
    ]

    cmd = [
        sys.executable, "run.py",
        "--config-path=examples/memrift/conf",
        "--config-name=train_mock",
    ] + overrides

    t0 = time.time()
    # run.py launches training via `nohup ... &`, so it returns immediately.
    # We launch run.py, wait for it to write the PID file, then wait for the
    # training process to finish by polling the log file.
    launcher = subprocess.Popen(
        cmd, env=env, cwd=str(FLAGSCALE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    launcher.wait()

    # Now wait for training to actually complete (it runs in background)
    # Poll the log file until we see train_iters elapsed-time lines or timeout
    timeout = 600  # 10 minutes max per run
    poll_interval = 3
    deadline = t0 + timeout
    train_pid = None

    # Try to read PID from pid_file
    for _ in range(30):  # wait up to 30s for pid file
        if pid_file.exists():
            try:
                train_pid = int(pid_file.read_text().strip())
                print(f"  Training PID={train_pid}")
                break
            except Exception:
                pass
        time.sleep(1)

    # Poll log file until train completes (process exits) or timeout
    # We wait for the training process to exit NATURALLY so it can finish
    # writing checkpoints before we clean up for the next run.
    print(f"  Waiting for training to complete (pid={train_pid})...", flush=True)
    seen_iters = 0
    while time.time() < deadline:
        log_text = LOG_FILE.read_text() if LOG_FILE.exists() else ""
        n_iter = len(re.findall(r'elapsed time per iteration', log_text))
        if n_iter != seen_iters:
            seen_iters = n_iter
            elapsed = int(time.time() - t0)
            print(f"  ...{elapsed}s: {n_iter}/{train_iters} iterations logged", flush=True)
        # Check if training process has exited
        if train_pid:
            r = subprocess.run(["kill", "-0", str(train_pid)], capture_output=True)
            if r.returncode != 0:
                # Process exited; wait a moment for final writes to flush
                time.sleep(3)
                break
        else:
            # No PID – just wait until we see all iterations
            if n_iter >= train_iters:
                time.sleep(5)  # give extra time for checkpoint to be written
                break
        time.sleep(poll_interval)

    elapsed_wall = time.time() - t0
    log_text = LOG_FILE.read_text() if LOG_FILE.exists() else ""

    step_times = re.findall(r'elapsed time per iteration \(ms\):\s*([\d.]+)', log_text)
    step_times = [float(x) for x in step_times]
    # skip warmup
    step_times = step_times[warmup_iters:] if len(step_times) > warmup_iters else step_times

    # Last memory report (appears after iteration N)
    mem_match = re.search(r'max allocated:\s*([\d.]+)', log_text)
    max_alloc_mb = float(mem_match.group(1)) if mem_match else None

    mem_reserv_match = re.search(r'max reserved:\s*([\d.]+)', log_text)
    max_reserv_mb = float(mem_reserv_match.group(1)) if mem_reserv_match else None

    if not step_times:
        print("  [WARN] no timing found, check log")
        print(log_text[-2000:])

    return {
        "seq": seq,
        "mode": mode,
        "step_times_ms": step_times,
        "avg_ms": sum(step_times) / len(step_times) if step_times else None,
        "min_ms": min(step_times) if step_times else None,
        "max_ms": max(step_times) if step_times else None,
        "max_alloc_mb": max_alloc_mb,
        "max_reserv_mb": max_reserv_mb,
        "wall_s": elapsed_wall,
        "exit_code": launcher.returncode,
    }


def print_table(results):
    seqs = sorted(set(r["seq"] for r in results))
    modes = list(MODES.keys())

    print("\n\n" + "="*90)
    print("  全量基准测试结果  |  TinyLlama-1.1B  |  bs=1")
    print("="*90)
    header = f"{'模式':<16}" + "".join(f"  seq={s:<6}" for s in seqs)
    print(f"\n【迭代时间 avg (ms)】")
    print(header)
    print("-"*80)
    by = {(r["seq"], r["mode"]): r for r in results}
    for mode in modes:
        row = f"{mode:<16}"
        for s in seqs:
            r = by.get((s, mode))
            if r and r["avg_ms"]:
                row += f"  {r['avg_ms']:>8.1f}  "
            else:
                row += f"  {'N/A':>8}  "
        print(row)

    print(f"\n【显存峰值 max_alloc (GB)】")
    print(header)
    print("-"*80)
    for mode in modes:
        row = f"{mode:<16}"
        for s in seqs:
            r = by.get((s, mode))
            if r and r["max_alloc_mb"]:
                row += f"  {r['max_alloc_mb']/1024:>7.2f}G  "
            else:
                row += f"  {'N/A':>8}  "
        print(row)

    print(f"\n【各步时间明细 (ms)】")
    for r in results:
        ts = [f"{t:.0f}" for t in r["step_times_ms"]]
        print(f"  {r['mode']:<16} seq={r['seq']:<5}  {' '.join(ts)}")
    print("="*90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", nargs="+", type=int, default=[512, 2048, 4096])
    parser.add_argument("--modes", nargs="+", default=list(MODES.keys()))
    parser.add_argument("--train-iters", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = []
    for mode in args.modes:
        for seq in args.seq:
            r = run_one(seq, mode, args.train_iters, args.warmup_iters)
            results.append(r)
            avg_str = f"{r['avg_ms']:.1f}ms" if r['avg_ms'] is not None else "N/A"
            mem_str = f"{r['max_alloc_mb']:.1f}MB" if r['max_alloc_mb'] is not None else "N/A"
            print(f"  → avg={avg_str}  mem={mem_str}  exit={r['exit_code']}")

    print_table(results)

    out_path = args.out or f"bench_matrix_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path = str(FLAGSCALE_DIR / out_path) if not os.path.isabs(out_path) else out_path
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
