import time
import torch
import io
import sys
from vllm import LLM, SamplingParams
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
MODEL_PATH = "/share/project/mengyc/flagScale/FlagScale/checkpoints/Qwen2.5-7B-Instruct"

def profile_inference():
    # 初始化模型
    print("Loading model...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        dtype="bfloat16"
    )
    
    # 构造测试 Prompt (稍微长一点，模拟真实输入)
    prompt = "请详细解释一下量子力学中的纠缠态是什么？"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)

    # --- 预热 (Warmup) ---
    # GPU 第一次运行通常会慢，需要预热
    print("Warming up...")
    llm.generate([prompt], sampling_params)
    torch.cuda.synchronize()
    
    # --- 开始测试 ---
    print("Start profiling...")
    start_time = time.perf_counter()
    
    # 运行生成
    outputs = llm.generate([prompt], sampling_params)
    
    torch.cuda.synchronize() # 确保 GPU 计算完成
    end_time = time.perf_counter()

    # --- 计算指标 ---
    output = outputs[0]
    generated_text = output.outputs[0].text
    num_input_tokens = len(output.prompt_token_ids)
    num_output_tokens = len(output.outputs[0].token_ids)
    total_time = end_time - start_time

    # 简单计算 TPS (Tokens Per Second)
    tps = num_output_tokens / total_time

    print("-" * 30)
    print(f"Input Tokens: {num_input_tokens}")
    print(f"Output Tokens: {num_output_tokens}")
    print(f"Total Latency: {total_time:.4f} s")
    print(f"Throughput:    {tps:.2f} tokens/s")
    print("-" * 30)
    print(f"Generated Text Preview:\n{generated_text[:50]}...")

if __name__ == "__main__":
    profile_inference()
