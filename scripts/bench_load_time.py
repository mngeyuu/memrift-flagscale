#!/usr/bin/env python3
"""
Benchmark: model load time + inference time
  - standard (pure base model, no MemRift)
  - memrift   (compressed weights, streaming decompress)
"""
import time, sys, os
sys.path.insert(0, '/share/project/mengyc/code/memrift-flagscale')
sys.path.insert(0, '/share/project/mengyc/code/memrift-flagscale/flagscale/train')
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ['WANDB_MODE'] = 'offline'
os.environ['HF_HUB_DISABLE_XET'] = '1'

import argparse
from functools import partial
import torch

MODEL_PATH    = '/share/project/mengyc/models/Meta-Llama-3-8B-Instruct'
MEMRIFT_DIR   = '/share/project/mengyc/code/memrift-flagscale/memrift_weights/llama31_8b_level18'
PROMPT        = 'Once upon a time'
MAX_NEW_TOKENS = 20

# ── minimal fake argv for megatron ─────────────────────────────────────────
def build_argv(use_memrift: bool):
    args = [
        '--num-layers', '32',
        '--hidden-size', '4096',
        '--ffn-hidden-size', '14336',
        '--num-attention-heads', '32',
        '--num-query-groups', '8',
        '--seq-length', '2048',
        '--max-position-embeddings', '131072',
        '--norm-init-weight', '1.0',
        '--use-rotary-position-embeddings',
        '--rotary-base', '500000',
        '--no-position-embedding',
        '--normalization', 'RMSNorm',
        '--norm-epsilon', '1e-05',
        '--swiglu',
        '--untie-embeddings-and-output-weights',
        '--make-vocab-size-divisible-by', '64',
        '--disable-bias-linear',
        '--use-flash-attn',
        '--bf16',
        '--transformer-impl', 'transformer_engine',
        '--attention-dropout', '0.0',
        '--hidden-dropout', '0.0',
        '--init-method-std', '0.02',
        '--tokenizer-type', 'HuggingFaceTokenizer',
        '--tokenizer-path', MODEL_PATH,
        '--tokenizer-model', MODEL_PATH,
        '--train-iters', '1',
        '--eval-iters', '0',
        '--micro-batch-size', '1',
        '--global-batch-size', '1',
        '--lr', '1e-4',
        '--min-lr', '1e-5',
        '--lr-decay-style', 'cosine',
        '--lr-warmup-iters', '0',
        '--mock-data',
        '--split', '1,0,0',
        '--no-load-optim',
        '--no-load-rng',
        '--wandb-mode', 'disabled',
        '--tensor-model-parallel-size', '1',
        '--pipeline-model-parallel-size', '1',
        '--prompt', PROMPT,
        '--max-new-tokens', str(MAX_NEW_TOKENS),
    ]
    if use_memrift:
        args += [
            '--memrift-enable',
            '--memrift-weight-enable',
            '--memrift-compressed-weight-dir', MEMRIFT_DIR,
            '--memrift-zstd-level', '3',
            '--memrift-prefetch-layers', '12',
            '--memrift-weight-async',
            '--memrift-activation-enable',
            '--memrift-act-async',
            '--memrift-decode-pool-workers', '48',
            '--memrift-compress-pool-workers', '8',
        ]
    return args

def run_bench(use_memrift: bool):
    mode = 'memrift' if use_memrift else 'standard'
    print(f'\n{"="*60}')
    print(f'  Mode: {mode.upper()}')
    print(f'{"="*60}')

    import sys as _sys
    _sys.argv = ['bench'] + build_argv(use_memrift)

    # delayed imports so megatron re-initializes cleanly per run
    from megatron.training import initialize_megatron, get_args, get_tokenizer, get_model
    from megatron.core.enums import ModelType

    t0 = time.perf_counter()

    initialize_megatron(extra_args_provider=_extra_args)
    args      = get_args()
    tokenizer = get_tokenizer()

    t_init = time.perf_counter()
    print(f'[TIMER] megatron init        : {t_init - t0:.2f} s')

    from flagscale.train.models.gpt.gpt_model import GPTModel
    from flagscale.train.train_gpt import model_provider, gpt_builder
    model_list = get_model(
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )
    model = model_list[0]
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    t_model = time.perf_counter()
    print(f'[TIMER] model build          : {t_model - t_init:.2f} s')

    if use_memrift:
        from flagscale.compress.memrift.train_hooks import inject_memrift_for_inference
        inject_memrift_for_inference(model, args)
        t_memrift = time.perf_counter()
        print(f'[TIMER] memrift inject hooks  : {t_memrift - t_model:.2f} s')
        t_ready = t_memrift
    else:
        t_ready = t_model

    t_load_total = t_ready - t0
    print(f'[TIMER] TOTAL LOAD TIME      : {t_load_total:.2f} s')

    # ── generation ─────────────────────────────────────────────────────────
    from flagscale.train.generate_gpt import generate
    generate(model, tokenizer, args)

    del model
    torch.cuda.empty_cache()

def _extra_args(parser):
    parser.add_argument('--prompt', type=str, default=PROMPT)
    parser.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    parser.add_argument('--greedy', action='store_true')
    return parser

if __name__ == '__main__':
    # standard first, then memrift
    run_bench(use_memrift=False)
    run_bench(use_memrift=True)
    print('\n=== BENCH DONE ===')
