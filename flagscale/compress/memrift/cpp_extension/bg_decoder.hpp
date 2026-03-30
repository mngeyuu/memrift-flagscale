#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

namespace bg_decoder {

// 背景解码工作器类
class BackgroundDecoder {
public:
    BackgroundDecoder();
    ~BackgroundDecoder();

    // 初始化解码工作器（在独立线程中调用）
    void initialize();

    // 执行解码任务
    bool decode_batch(const std::vector<torch::Tensor>& ans_pinned_list,
                     const std::vector<torch::Tensor>& sm_gpu_list,
                     const std::vector<std::vector<int64_t>>& shape_list,
                     const std::vector<std::vector<int64_t>>& stride_list,
                     torch::Tensor& merge_event);

    // 关闭工作器
    void shutdown();

private:
    // CUDA 流和编解码器
    torch::cuda::Stream h2d_stream;
    torch::cuda::Stream decode_stream;
    torch::cuda::Stream merge_stream;

    // NVCOMP 解码器
    void* nvcomp_codec;  // 抽象 nvcomp 解码器指针

    // 浮点拆分合并操作
    void* fs_sp;  // 抽象 float_split_stride_pin 指针

    bool initialized;
};

// 独立的线程函数，用于执行背景解码
void decode_worker_wrapper(const std::shared_ptr<BackgroundDecoder>& decoder,
                          std::function<void()> shutdown_func);

}  // namespace bg_decoder
