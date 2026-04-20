#include "bg_decoder.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <memory>

namespace py = pybind11;
namespace bg_decoder = bg_decoder;

// 解码任务结构体
struct DecodeTask {
    std::vector<torch::Tensor> ans_pinned_list;
    std::vector<torch::Tensor> sm_gpu_list;
    std::vector<std::vector<int64_t>> shape_list;
    std::vector<std::vector<int64_t>> stride_list;
    torch::Tensor merge_event;
    std::function<void(bool)> callback;
};

bg_decoder::BackgroundDecoder::BackgroundDecoder()
    : initialized(false), nvcomp_codec(nullptr), fs_sp(nullptr) {
    // 在构造时创建 CUDA 流
    h2d_stream = torch::cuda::Stream(torch::cuda::Stream::DEFAULT);
    decode_stream = torch::cuda::Stream(torch::cuda::Stream::DEFAULT);
    merge_stream = torch::cuda::Stream(torch::cuda::Stream::DEFAULT);
}

bg_decoder::BackgroundDecoder::~BackgroundDecoder() {
    shutdown();
}

void bg_decoder::BackgroundDecoder::initialize() {
    // 初始化 NVCOMP 解码器（如果可用）
    try {
        // 这里应该是实际的 NVCOMP 初始化代码
        // 暂时留空，需要链接 NVCOMP 库
        nvcomp_codec = nullptr;
    } catch (...) {
        nvcomp_codec = nullptr;
    }

    // 初始化 float_split_stride_pin（如果可用）
    try {
        // 这里应该是实际的 float_split_stride_pin 初始化代码
        // 暂时留空
        fs_sp = nullptr;
    } catch (...) {
        fs_sp = nullptr;
    }

    initialized = true;
}

bool bg_decoder::BackgroundDecoder::decode_batch(const std::vector<torch::Tensor>& ans_pinned_list,
                                                 const std::vector<torch::Tensor>& sm_gpu_list,
                                                 const std::vector<std::vector<int64_t>>& shape_list,
                                                 const std::vector<std::vector<int64_t>>& stride_list,
                                                 torch::Tensor& merge_event) {
    if (!initialized || !nvcomp_codec || !fs_sp) {
        return false;
    }

    auto device = sm_gpu_list[0].device();
    std::vector<torch::Tensor> decompressed_bf16;

    // 阶段 1: H2D 传输（h2d_stream）
    std::vector<torch::Tensor> comp_gpu_list;
    for (const auto& ans_pinned : ans_pinned_list) {
        auto comp_gpu = ans_pinned.to(device, c10::TensorOptions().dtype(torch::kUInt8),
                                   false, true);  // non_blocking=true
        comp_gpu_list.push_back(comp_gpu);
    }

    // 阶段 2: 解码（decode_stream）
    // 等待 h2d 完成
    decode_stream.wait_stream(h2d_stream);

    std::vector<torch::Tensor> decoded_list;
    for (const auto& comp_gpu : comp_gpu_list) {
        // 这里应该是实际的 NVCOMP 解码代码
        auto decoded = comp_gpu.to(torch::kUInt8);
        decoded_list.push_back(decoded);
    }

    // 阶段 3: 合并（merge_stream）
    merge_stream.wait_stream(decode_stream);

    for (size_t i = 0; i < decoded_list.size(); ++i) {
        const auto& decoded = decoded_list[i];
        const auto& sm_gpu = sm_gpu_list[i];
        const auto& shape = shape_list[i];
        const auto& stride = stride_list[i];

        // 这里应该是实际的 float_split_stride_pin.merge 操作
        auto bf16 = decoded.to(torch::kBFloat16);
        decompressed_bf16.push_back(bf16);
    }

    // 记录合并事件
    merge_event.record(merge_stream);

    return true;
}

void bg_decoder::BackgroundDecoder::shutdown() {
    initialized = false;
    // 清理资源
    if (nvcomp_codec) {
        // 这里应该是 NVCOMP 解码器清理代码
        nvcomp_codec = nullptr;
    }

    if (fs_sp) {
        // 这里应该是 float_split_stride_pin 清理代码
        fs_sp = nullptr;
    }
}

void bg_decoder::decode_worker_wrapper(const std::shared_ptr<BackgroundDecoder>& decoder,
                                       std::function<void()> shutdown_func) {
    // 显式释放 GIL，这样主线程可以继续执行
    py::gil_scoped_release release_gil;

    try {
        // 初始化解码器
        decoder->initialize();

        // 解码任务队列
        std::queue<std::shared_ptr<DecodeTask>> task_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool should_stop = false;

        // 工作循环
        while (!should_stop) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [&] {
                return should_stop || !task_queue.empty();
            });

            // 处理任务
            while (!should_stop && !task_queue.empty()) {
                auto task = task_queue.front();
                task_queue.pop();
                lock.unlock();

                bool success = decoder->decode_batch(
                    task->ans_pinned_list,
                    task->sm_gpu_list,
                    task->shape_list,
                    task->stride_list,
                    task->merge_event
                );

                task->callback(success);
                lock.lock();
            }
        }

        // 清理
        decoder->shutdown();
        shutdown_func();
    } catch (const std::exception& e) {
        // 捕获并处理异常
        py::gil_scoped_acquire acquire_gil;
        // 这里可以添加错误报告
        shutdown_func();
    }
}

PYBIND11_MODULE(bg_decoder, m) {
    py::class_<bg_decoder::BackgroundDecoder>(m, "BackgroundDecoder")
        .def(py::init<>())
        .def("initialize", &bg_decoder::BackgroundDecoder::initialize)
        .def("decode_batch", &bg_decoder::BackgroundDecoder::decode_batch)
        .def("shutdown", &bg_decoder::BackgroundDecoder::shutdown);

    m.def("decode_worker_wrapper", &bg_decoder::decode_worker_wrapper);
}
