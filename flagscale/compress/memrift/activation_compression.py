from contextlib import contextmanager
from typing import Optional, Set, Any
import functools
import weakref

import torch
import torch.nn as nn

from flagscale.compress.memrift.async_compressor import PlaceHolderToken, AsyncCompressor


def _should_compress_activation(t: torch.Tensor, skip_storage_ptrs: Optional[Set[int]] = None) -> bool:
    if not t.is_cuda or t.numel() == 0:
        return False
    if t.dtype not in (torch.float32, torch.bfloat16):
        return False
    if t.is_leaf or (not t.requires_grad):
        return False
    if skip_storage_ptrs:
        try:
            if t.untyped_storage().data_ptr() in skip_storage_ptrs:
                return False
        except Exception:
            pass
    return True


def _unpack(tok: Any, compressor: AsyncCompressor):
    if isinstance(tok, PlaceHolderToken):
        tok_act_async = bool(getattr(tok, "act_async", False))
        if tok_act_async and compressor.enable_async:
            if tok.decomped_data is not None and tok.CtoD_copy_evt is None:
                return tok.decomped_data
            # Fallback: if bwd_pre_hook did not schedule async decode yet, do it here.
            if tok.decomped_data is None and getattr(tok, "future", None) is not None:
                compressor.decompress_async(tok, tok.future)
                tok.future = None
            tok.ready_evt.wait()
            tok.decomped_data.record_stream(torch.cuda.current_stream())
            tok._clear_after_recover()
        else:
            if tok.decomped_data is None:
                compressor.decompress_sync(tok)
                tok.decomped_data.record_stream(torch.cuda.current_stream())
        return tok.decomped_data
    return tok


class DecoderLayerWrapper(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        compressor: AsyncCompressor,
        use_async: bool,
        release_after_unpack: bool = True,
        skip_storage_ptrs: Optional[Set[int]] = None,
        do_empty: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self.comp = compressor
        self.use_async = use_async
        self.release_after_unpack = release_after_unpack
        self.skip_storage_ptrs = skip_storage_ptrs or set()
        self.tokens = []
        self.futures = []
        self.do_empty = do_empty

        self.register_full_backward_pre_hook(self._bwd_pre_hook)
        self.register_full_backward_hook(self._bwd_hook)

    def forward(self, *inp, **kw):
        self.tokens.clear()
        self.futures.clear()
        seen = {}

        def _pack(t):
            if not _should_compress_activation(t, self.skip_storage_ptrs):
                return t

            key = (t.data_ptr(), t.nbytes)
            if key in seen:
                tok_ref, t_ref = seen[key]
                if t_ref() is not None:
                    tok = tok_ref()
                    if tok is not None:
                        return tok

            tok = PlaceHolderToken(t.dtype, t.shape, tuple(t.stride()), t.storage_offset())
            tok.act_async = bool(self.use_async)
            seen[key] = (weakref.ref(tok), weakref.ref(t))
            if self.use_async and self.comp.enable_async:
                fut = self.comp.kickoff_async(tok, t)
                self.futures.append(fut)
                tok.fut_id = len(self.futures) - 1
                tok.future = fut
            else:
                self.comp.kickoff_sync(tok, t)
            self.tokens.append(weakref.ref(tok))
            return tok

        unpack_fn = functools.partial(_unpack, compressor=self.comp)
        with torch.autograd.graph.saved_tensors_hooks(_pack, unpack_fn):
            out = self.layer(*inp, **kw)

        if self.do_empty:
            torch.cuda.empty_cache()
        return out

    def _bwd_pre_hook(self, _mod, _grad_in):
        if not (self.use_async and self.comp.enable_async):
            return
        for tok_ptr in self.tokens[::-1]:
            tok = tok_ptr()
            if tok is None:
                continue
            fut = self.futures[tok.fut_id]
            self.comp.decompress_async(tok, fut)
            self.futures[tok.fut_id] = None
            tok.future = None

    def _bwd_hook(self, _mod, _gin, _gout):
        for tok_ptr in self.tokens:
            tok = tok_ptr()
            if tok is not None:
                del tok
        self.tokens.clear()
        self.futures.clear()


@contextmanager
def activation_compression_context(
    compressor: Optional[AsyncCompressor] = None,
    use_async: bool = False,
    release_after_unpack: bool = True,
    zstd_level: int = 18,
    skip_storage_ptrs: Optional[Set[int]] = None,
):
    if compressor is None:
        compressor = AsyncCompressor(
            compress_workers=2,
            decode_workers=2,
            concurrency_limit=4,
            zstd_level=zstd_level,
            enable_async=False,
        )

    seen = {}

    def pack(t):
        if not _should_compress_activation(t, skip_storage_ptrs):
            return t

        key = (t.data_ptr(), t.nbytes)
        if key in seen:
            tok_ref, t_ref = seen[key]
            if t_ref() is not None:
                tok = tok_ref()
                if tok is not None:
                    return tok

        tok = PlaceHolderToken(t.dtype, t.shape, tuple(t.stride()), t.storage_offset())
        tok.act_async = bool(use_async)
        seen[key] = (weakref.ref(tok), weakref.ref(t))
        if use_async and compressor.enable_async:
            fut = compressor.kickoff_async(tok, t)
            tok.fut_id = 0
            tok.future = fut
        else:
            compressor.kickoff_sync(tok, t)
        return tok

    unpack = functools.partial(_unpack, compressor=compressor)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield compressor
