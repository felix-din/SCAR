import json
import time
from vllm import LLM, SamplingParams

# 1. 初始化 vLLM 引擎
model_path = "./Llama-3.1-8B-Instruct"
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    max_model_len=16384,
    enable_prefix_caching=True,
)

# 2. 采样参数：配合 Few-Shot，max_tokens 保持 64 足够，repetition_penalty 适度防复读
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
    repetition_penalty=1.2,
    stop=[
        "<|eot_id|>", 
        "<|end_of_text|>", 
        "\n",             # 一旦给完第一行的答案想换行，立刻截断！
        "However",        # 一旦想转折唠叨，立刻截断！
        "Note",           # 阻止 Note:
        "Therefore",      # 阻止 Therefore
        "Ref",            # 阻止 Reference
        "According"       # 阻止 According to
    ], 
)

# 3. 读取数据集
with open("processed_model_input.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

prompts = []
for item in dataset:
    question = item["question"]
    passages = item["passages"]

    # 规范化参考文档格式
    context = "\n".join([f"- {p}" for p in passages])

    # 核心：加入 2 个少样本（Few-Shot）示例，给模型树立绝对不写分析、直接给结果的榜样
    prompt = f"""You are a precise QA assistant. Read the references and answer the question concisely and directly. Do not analyze, do not explain. 
If the answer cannot be found in the references, output: I cannot answer based on the provided documents.

Example 1:
References:
- Apple Inc. is an American multinational technology company headquartered in Cupertino, California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.
Question: Who founded Apple Inc.?
Answer: Steve Jobs, Steve Wozniak, and Ronald Wayne

Example 2:
References:
- The capital of France is Paris. It is also the most populous city in the country.
Question: What is the population of Paris?
Answer: I cannot answer based on the provided documents.

Now, answer the following question:

References:
{context}

Question: {question}
Answer:"""

    prompts.append(prompt)

print(f"开始使用 vLLM 批量处理 {len(prompts)} 条数据并统计时间和 Token...")

# 4. 执行推理
start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
total_time = time.time() - start_time

# 5. 统计与保存
total_samples = len(prompts)
avg_time_per_sample = total_time / total_samples

total_prompt_tokens = 0
total_completion_tokens = 0
results = []

for idx, output in enumerate(outputs):
    item = dataset[idx]
    generated_text = output.outputs[0].text.strip()

    prompt_tokens = len(output.prompt_token_ids)
    completion_tokens = len(output.outputs[0].token_ids)
    sample_total_tokens = prompt_tokens + completion_tokens

    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    results.append({
        "original_id": item["original_id"],
        "question": item["question"],
        "model_answer": generated_text,
        "token_stats": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": sample_total_tokens
        }
    })

grand_total_tokens = total_prompt_tokens + total_completion_tokens
avg_tokens_per_sample = grand_total_tokens / total_samples

# 打印报告
print("\n" + "=" * 50)
print("         vLLM 性能与 Token 消耗综合报告          ")
print("=" * 50)
print(f"总样本数           : {total_samples} 条")
print(f"总耗时             : {total_time:.4f} 秒")
print(f"平均每个样本耗时   : {avg_time_per_sample:.4f} 秒")
print(f"平均每个样本 Token : {avg_tokens_per_sample:.2f} tokens")
print("=" * 50)

output_data = {
    "performance_statistics": {
        "total_samples": total_samples,
        "total_time_seconds": round(total_time, 4),
        "average_time_per_sample_seconds": round(avg_time_per_sample, 4)
    },
    "token_statistics": {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "grand_total_tokens": grand_total_tokens,
        "average_tokens_per_sample": round(avg_tokens_per_sample, 2)
    },
    "results": results
}

with open("vllm_output_with_tokens.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("加入 Few-Shot 后的结果已成功保存至: vllm_output_with_tokens.json")