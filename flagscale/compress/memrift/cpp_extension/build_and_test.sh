#!/bin/bash
# 构建和测试 MemRift 背景解码器 C++ 扩展的脚本

set -e  # 遇到错误立即退出

# 彩色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${GREEN}=========================================="
echo -e "MemRift 背景解码器 C++ 扩展构建脚本"
echo -e "=========================================="

# 检查是否在正确的目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo -e "${GREEN}当前目录: ${SCRIPT_DIR}${NC}"

# 检查是否有 setup.py
if [ ! -f "setup.py" ]; then
    echo -e "${RED}错误: setup.py 不在当前目录中${NC}"
    exit 1
fi

# 检查是否有 src 目录
if [ ! -d "src" ]; then
    echo -e "${YELLOW}警告: 没有找到 src 目录${NC}"
fi

# 检查 Python 版本
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}Python 版本: ${PYTHON_VERSION}${NC}"

# 检查 CUDA 是否可用
echo -e "${GREEN}检查 CUDA 可用性...${NC}"
if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    CUDA_AVAILABLE="true"
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    echo -e "${GREEN}CUDA 版本: ${CUDA_VERSION}${NC}"
else
    CUDA_AVAILABLE="false"
    echo -e "${YELLOW}警告: CUDA 不可用${NC}"
fi

# 构建扩展
echo -e "${GREEN}正在构建 C++ 扩展...${NC}"
if [ "$CUDA_AVAILABLE" = "true" ]; then
    echo -e "${YELLOW}使用 CUDA 支持构建${NC}"
    python3 setup.py build_ext --inplace
else
    echo -e "${YELLOW}CUDA 不可用，将使用 CPU 版本构建${NC}"
    python3 setup.py build_ext --inplace --without-cuda
fi

# 检查构建是否成功
if [ -f "bg_decoder.cpython-*.so" ]; then
    echo -e "${GREEN}构建成功!${NC}"
else
    echo -e "${RED}构建失败!${NC}"
    exit 1
fi

# 运行简单测试
echo -e "${GREEN}正在运行简单的导入测试...${NC}"
if python3 -c "from flagscale.compress.memrift.cpp_extension import BackgroundDecoder, is_available; print(f'C++ 扩展可用: {is_available()}')"; then
    echo -e "${GREEN}测试通过: 成功导入 BackgroundDecoder${NC}"
else
    echo -e "${RED}测试失败: 无法导入 BackgroundDecoder${NC}"
    exit 1
fi

# 运行更详细的测试
echo -e "${GREEN}正在运行详细测试...${NC}"
python3 - <<END
import sys
sys.path.insert(0, '/share/project/mengyc/flagScale/FlagScale')

import torch
from flagscale.compress.memrift.cpp_extension import BackgroundDecoder, is_available

print("="*50)
print("C++ 扩展可用性检查")
print("="*50)
print(f"扩展可用: {is_available()}")

if is_available():
    print("\\n" + "="*50)
    print("测试背景解码器实例化")
    print("="*50)
    try:
        decoder = BackgroundDecoder()
        print("成功: BackgroundDecoder 实例化")
    except Exception as e:
        print(f"失败: BackgroundDecoder 实例化失败 - {e}")
else:
    print("\\n扩展不可用，无法测试实例化")
END

echo -e "\\n${GREEN}=========================================="
echo -e "构建和测试完成！"
echo -e "=========================================="
echo -e "C++ 扩展已准备好使用。"
echo -e ""
echo -e "${YELLOW}注意:${NC}"
echo -e "- 使用前需要确保已正确安装了所有依赖项"
echo -e "- 如果在使用过程中遇到问题，尝试重新构建"
echo -e "- 要完全重新构建，请先运行: rm -f bg_decoder.cpython-*.so && python3 setup.py build_ext --inplace"
