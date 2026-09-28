"""Megatron inference backend used by the MemRift benchmark evaluator."""

from __future__ import annotations

import os
import re
import sys
from functools import partial
from pathlib import Path


def adapter_state_dict(state_dict):
    adapters = {key: value for key, value in state_dict.items() if ".adapter." in key}
    if not adapters:
        raise ValueError("checkpoint contains no LoRA adapter parameters")
    return adapters


def _load_memrift_adapters(model, checkpoint_dir: str, torch) -> None:
    root = Path(checkpoint_dir)
    iteration = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    checkpoint = root / f"iter_{iteration:07d}" / "mp_rank_00" / "model_optim_rng.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapters = adapter_state_dict(payload["model"])
    incompatible = model.load_state_dict(adapters, strict=False)
    unexpected_keys = [] if incompatible is None else incompatible.unexpected_keys
    unexpected_adapters = [key for key in unexpected_keys if ".adapter." in key]
    if unexpected_adapters:
        raise RuntimeError(f"checkpoint adapter keys were not accepted: {unexpected_adapters[:5]}")


def _uses_adapter_only_load(mode: str | None) -> bool:
    return mode in {"memrift_async", "lora_adapter_only"}


class MegatronBenchmarkBackend:
    def __init__(self) -> None:
        train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "flagscale", "train"))
        if train_dir not in sys.path:
            sys.path.insert(0, train_dir)

        import torch
        from megatron.core.enums import ModelType
        from megatron.training import get_args, get_model, get_tokenizer
        from megatron.training.checkpointing import load_checkpoint
        from megatron.training.initialize import initialize_megatron
        from gpt_builders import gpt_builder
        from model_provider import model_provider

        initialize_megatron(
            args_defaults={
                "tokenizer_type": "HuggingFaceTokenizer",
                "train_iters": 1,
                "eval_iters": 0,
                "micro_batch_size": 1,
                "global_batch_size": 1,
                "lr": 1e-4,
                "min_lr": 1e-5,
                "lr_decay_style": "cosine",
                "lr_warmup_iters": 0,
                "mock_data": True,
                "split": "1,0,0",
                "no_load_optim": True,
                "no_load_rng": True,
            }
        )
        self.args = get_args()
        self.tokenizer = get_tokenizer()
        models = get_model(partial(model_provider, gpt_builder), ModelType.encoder_or_decoder, wrap_with_ddp=False)
        if self.args.load and _uses_adapter_only_load(os.environ.get("BENCHMARK_MODE")):
            _load_memrift_adapters(models[0], self.args.load, torch)
        elif self.args.load:
            load_checkpoint(models, None, None)
        self.model = models[0]
        self.model.eval()
        self.torch = torch

    def _tokens(self, text: str):
        return self.tokenizer.tokenize(text)

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        from megatron.core import InferenceParams

        torch = self.torch
        device = torch.cuda.current_device()
        outputs = []
        with torch.no_grad():
            for prompt in prompts:
                prompt_ids = self._tokens(prompt)
                tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                context = InferenceParams(max_batch_size=1, max_sequence_length=len(prompt_ids) + max_new_tokens)
                context.enable_prefill_mode()
                positions = torch.arange(len(prompt_ids), dtype=torch.long, device=device).unsqueeze(0)
                logits = self.model(tokens, positions, None, inference_context=context, runtime_gather_output=True)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = [next_id.item()]
                context.sequence_len_offset = len(prompt_ids)
                context.enable_decode_mode()
                for _ in range(max_new_tokens - 1):
                    position = torch.tensor([[context.sequence_len_offset]], dtype=torch.long, device=device)
                    logits = self.model(next_id, position, None, inference_context=context, runtime_gather_output=True)
                    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if next_id.item() == self.tokenizer.eod:
                        break
                    generated.append(next_id.item())
                    context.sequence_len_offset += 1
                    if len(generated) % 8 == 0:
                        partial_text = self.tokenizer.detokenize(generated)
                        if re.search(r"The answer is \-?[0-9.,]+\.", partial_text):
                            break
                outputs.append(self.tokenizer.detokenize(generated))
        return outputs

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[tuple[float, int]]:
        torch = self.torch
        device = torch.cuda.current_device()
        results = []
        with torch.no_grad():
            for context, continuation in requests:
                context_ids = self._tokens(context)
                full_ids = self._tokens(context + continuation)
                continuation_count = len(full_ids) - len(context_ids)
                if continuation_count <= 0:
                    raise ValueError("continuation must add at least one token")
                tokens = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
                positions = torch.arange(len(full_ids), dtype=torch.long, device=device).unsqueeze(0)
                logits = self.model(tokens, positions, None, runtime_gather_output=True)
                start = len(context_ids) - 1
                selected_logits = logits[:, start : len(full_ids) - 1, :].float()
                labels = tokens[:, len(context_ids) :]
                token_log_probs = torch.log_softmax(selected_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                results.append((float(token_log_probs.sum().item()), continuation_count))
        return results
