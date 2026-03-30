"""
MemRift 背景解码器 C++ 扩展

这个模块提供了优化的背景解码功能，通过以下方式提高性能：
1. 使用 C++ 而不是 Python 实现核心解码逻辑
2. 在密集操作期间显式释放 GIL
3. 使用 CUDA 流进行高效的异步操作
4. 消除 Python 线程带来的阻塞

使用方式：
    from flagscale.compress.memrift.cpp_extension import BackgroundDecoder

    # 或直接导入到 async_compressor 中
    from .cpp_extension import BackgroundDecoder
"""

import warnings

# 尝试导入编译好的 C++ 扩展
try:
    from .bg_decoder import BackgroundDecoder, decode_worker_wrapper
    CPP_EXTENSION_AVAILABLE = True
    warnings.warn(
        "Successfully loaded MemRift background decoder C++ extension. "
        "This will provide significant performance improvements by reducing "
        "GIL contention during background decoding operations."
    )
except ImportError:
    BackgroundDecoder = None
    decode_worker_wrapper = None
    CPP_EXTENSION_AVAILABLE = False
    warnings.warn(
        "MemRift background decoder C++ extension not available. "
        "Falling back to pure Python implementation. "
        "To enable C++ extension, run `python setup.py build_ext --inplace` "
        "in the cpp_extension directory."
    )


def is_available():
    """
    检查 C++ 扩展是否可用

    Returns:
        bool: 如果 C++ 扩展可用则返回 True，否则返回 False
    """
    return CPP_EXTENSION_AVAILABLE


__all__ = [
    "BackgroundDecoder",
    "decode_worker_wrapper",
    "is_available",
]
