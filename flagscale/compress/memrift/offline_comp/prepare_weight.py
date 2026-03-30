# === prepare_compressed_weights.py =================================================
import torch, zstandard as zstd, numpy as np, struct, json, os, argparse
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import time

from flagscale.compress.float_split_stride_pin import float_split_stride_pin as fs_sp

# Magic headers prepended to nvCOMP compressed exponent bytes.
# Runtime uses them to pick LZ4 vs ANS decode (see megatron_dynamic_loader / async_compressor).
NVCOMP_LZ4_MAGIC = b"NVL4"
NVCOMP_ANS_MAGIC = b"NVAN"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--outdir", default="./zstd_comped_weights")
parser.add_argument("--level", default=1)
parser.add_argument(
    "--compression",
    default="zstd",
    choices=["zstd", "nvcomp_lz4", "nvcomp_ans"],
    help=(
        "Compression backend for exponent bytes:\n"
        "  'zstd'        - CPU zstd (original, ~1.28x ratio, ~6ms CPU decomp at runtime)\n"
        "  'nvcomp_lz4'  - GPU nvCOMP LZ4 (~1.0x on exponents); startup may re-encode to ANS\n"
        "  'nvcomp_ans'  - GPU nvCOMP ANS (better ratio on real exponents); skips LZ4→ANS recompress"
    ),
)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

print(f"{args.model=}, {args.outdir=}, compression={args.compression}")
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0})

# ── Compression backend initialization ──────────────────────────────────────
_nvcomp_algo = None
if args.compression == "nvcomp_lz4":
    _nvcomp_algo = "lz4"
elif args.compression == "nvcomp_ans":
    _nvcomp_algo = "ans"

nvcomp_codec = None
nvcomp_stream = None
nvcomp_lib = None

if _nvcomp_algo is not None:
    try:
        import nvidia.nvcomp as nvcomp_lib
        nvcomp_stream = torch.cuda.Stream()
        nvcomp_codec = nvcomp_lib.Codec(
            algorithm=_nvcomp_algo, cuda_stream=nvcomp_stream.cuda_stream)
        print(f"Using GPU nvCOMP {_nvcomp_algo.upper()} compression.")
    except ImportError:
        print("WARNING: nvidia.nvcomp not available, falling back to CPU zstd.")
        _nvcomp_algo = None
        nvcomp_codec = None

cpr = zstd.ZstdCompressor(level=int(args.level))

comp_time = 0
uncomp_bytes = 0
index = []                                  # <layer name, binary file, shape, dtype>
MiB = 1024 * 1024

for name, p in model.named_parameters():
    fn = f"{len(index):06}.bin"
    out = os.path.join(args.outdir, fn)

    if p.dtype == torch.float32:
        print(f"{name=}, {p.shape=}")
    else:
        if p.dtype != torch.bfloat16:
            print(f"{name=}, {p.shape=}, {p.dtype=}")

    if p.dtype in (torch.bfloat16, torch.float32):
        # Split into exp (CPU pinned) + sm (GPU) components
        stm = torch.cuda.current_stream()
        with torch.cuda.stream(stm):
            cpu_exp, sm_bits = fs_sp.split(p, stm.cuda_stream)
        stm.synchronize()
        sm_cpu = sm_bits.cpu().contiguous()
        del sm_bits

        with open(out, "wb") as f:
            f.write(struct.pack("<Q", p.numel()))
            f.write(sm_cpu.numpy().tobytes())

            t0 = time.time()

            if _nvcomp_algo is not None and nvcomp_codec is not None:
                # ── GPU nvCOMP LZ4 or ANS compression ───────────────────
                exp_gpu = cpu_exp.to(p.device, non_blocking=False)
                nvcomp_stream.synchronize()
                arr = nvcomp_lib.as_array(exp_gpu)
                comp = nvcomp_codec.encode(arr)
                nvcomp_stream.synchronize()
                if _nvcomp_algo == "ans":
                    ans_valid = int(comp.buffer_size)
                    comp_t = torch.from_dlpack(comp).view(torch.uint8)
                    nvcomp_stream.synchronize()
                    comp_bytes_raw = comp_t[:ans_valid].cpu().numpy().tobytes()
                    comped_bytes = NVCOMP_ANS_MAGIC + comp_bytes_raw
                    scheme = "split_nvcomp_ans"
                else:
                    comp_t = torch.from_dlpack(comp.to_dlpack())
                    nvcomp_stream.synchronize()
                    comp_bytes_raw = comp_t.cpu().numpy().tobytes()
                    comped_bytes = NVCOMP_LZ4_MAGIC + comp_bytes_raw
                    scheme = "split_nvcomp_lz4"
            else:
                # ── CPU zstd compression (original behavior) ─────────────
                raw = cpu_exp.numpy().tobytes()
                comped_bytes = cpr.compress(raw)
                scheme = "split_zstd"

            dt = time.time() - t0
            comp_time += dt
            uncomp_bytes += p.numel()
            ratio = p.numel() / len(comped_bytes) if len(comped_bytes) else 0
            print(f"  {name}: {p.numel()} → {len(comped_bytes)//1024}KB "
                  f"(ratio={ratio:.2f}x), {dt*1000:.1f}ms")

            f.write(comped_bytes)
    else:
        torch.save(p.detach().cpu(), out)
        scheme = "raw_torch"

    index.append(dict(name=name,
                      file=fn,
                      shape=list(p.shape),
                      dtype=str(p.dtype).replace("torch.", ""),
                      scheme=scheme))

json.dump(index, open(os.path.join(args.outdir, "index.json"), "w"))
total_mib = uncomp_bytes / MiB
tput = total_mib / max(comp_time, 1e-9)
print(f"Wrote {len(index)} frozen tensors → '{args.outdir}', "
      f"compression time: {comp_time*1000:.2f} ms, "
      f"throughput: {tput:.1f} MiB/s")
