import os
from vllm import LLM, SamplingParams

# 1. 设置模型路径 (这里硬编码或者从环境变量读都可以)
# 注意：确保这个路径和你配置文件里的一致
MODEL_PATH = "/share/project/mengyc/flagScale/FlagScale/checkpoints/Qwen2.5-7B-Instruct"

def main():
    print(">>> 正在启动 vLLM 推理引擎...")
    
    # 初始化 vLLM
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="bfloat16"
    )

    # 准备测试数据
    prompts = [
        "你好，请介绍一下你自己。",
        "量子力学是什么？请用一句话解释。",
        "写一首关于春天的四言绝句。"
    ]
    
    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=200)

    # 执行推理
    print(f">>> 开始推理 {len(prompts)} 条数据...")
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    print("\n" + "="*50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"【提问】: {prompt}")
        print(f"【回答】: {generated_text}")
        print("-" * 50)
    print("="*50 + "\n")
    print(">>> 推理完成！")

if __name__ == "__main__":
    main()
