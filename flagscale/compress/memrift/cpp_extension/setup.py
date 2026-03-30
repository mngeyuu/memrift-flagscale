"""
Setup script for building the background decoder C++ extension.

This extension optimizes the background decode worker for MemRift by:
1. Using pybind11 to create Python bindings to C++ code
2. Explicitly releasing the GIL during intensive operations
3. Using CUDA streams for efficient async operations
4. Bypassing Python's threading limitations

Usage:
    cd /share/project/mengyc/flagScale/FlagScale/flagscale/compress/memrift/cpp_extension
    python setup.py build_ext --inplace
"""

import os
import sys
from setuptools import setup, Extension
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
import torch


def get_extension_modules():
    # 获取 CUDA 安装路径
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        cuda_home = os.environ.get("CUDA_PATH")

    # 获取当前目录
    current_dir = os.path.abspath(os.path.dirname(__file__))

    # 源文件
    sources = [
        "bg_decoder.cpp",
    ]

    # 编译参数
    extra_compile_args = [
        "-O3",
        "-std=c++17",
        "-fPIC",
    ]

    # 链接参数
    extra_link_args = []

    # CUDA 相关配置
    if torch.cuda.is_available():
        extra_compile_args.append("-DCUDA_AVAILABLE")
        if cuda_home:
            include_dirs = [
                os.path.join(cuda_home, "include"),
                os.path.join(current_dir, "..", "..", "..", "..", "external", "nvcomp", "include"),
            ]
        else:
            # 如果没有明确设置 CUDA_HOME，尝试自动检测
            import subprocess
            try:
                cuda_path = subprocess.check_output(["which", "nvcc"], universal_newlines=True).strip()
                cuda_home = os.path.abspath(os.path.join(os.path.dirname(cuda_path), ".."))
                include_dirs = [
                    os.path.join(cuda_home, "include"),
                    os.path.join(current_dir, "..", "..", "..", "..", "external", "nvcomp", "include"),
                ]
            except Exception as e:
                print(f"Warning: Failed to find CUDA installation: {e}")
                include_dirs = []

    else:
        include_dirs = []

    # 创建扩展
    ext_modules = [
        Pybind11Extension(
            "bg_decoder",
            sources=sources,
            include_dirs=include_dirs,
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]

    return ext_modules


if __name__ == "__main__":
    setup(
        name="bg_decoder",
        version="0.1.0",
        description="Background decoder extension for MemRift",
        long_description="C++ extension to optimize background decoding for MemRift",
        ext_modules=get_extension_modules(),
        cmdclass={"build_ext": build_ext},
        install_requires=["torch>=1.10.0", "pybind11>=2.6.0"],
        classifiers=[
            "Programming Language :: C++",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: Linux",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        zip_safe=False,
    )
