"""
Forward generation script - No tag version - Pure retrieval (vLLM Version)
Experiment: forward_llama_notag_pure
Seed: 1001
Output file: forward_llama_notag_pure.json
"""

import os
import re
import json
import torch
import random
import warnings
import argparse
import gc
import time  # 引入 time 用于精准计算耗时
from datetime import datetime
from vllm import LLM, SamplingParams  # vLLM 批量推理引擎
from thefuzz import fuzz

# Global variable for start time
start_time = None

# ==========================================
# Experiment Configuration
# ==========================================

EXPERIMENT_CONFIG = {
    "name": "forward_llama_notag_pure_modify_yanprompt",
    "model_path": "/root/autodl-tmp/Llama-3.1-8B-Instruct",
    "random_seed": 1001,
    "input_filename": "processed_model_input.json",
    "output_filename": "stage1_output_stage2_input_v2.json",
    "max_new_tokens": 128,
    "temperature": 0.0
}


# ==========================================
# 1. Environment Setup & Hardware Check
# ==========================================

def setup_environment():
    """Set up environment and suppress warnings"""
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore")

    print(f"🚀 Experiment: {EXPERIMENT_CONFIG['name']}")
    print(f"📌 Random Seed: {EXPERIMENT_CONFIG['random_seed']}")
    print(f"💻 Device: {'GPU available' if torch.cuda.is_available() else 'CPU only'}")

    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"📊 Memory: {memory_gb:.1f} GB")

    random.seed(EXPERIMENT_CONFIG['random_seed'])


# ==========================================
# 2. Model Loading (vLLM 引擎初始化)
# ==========================================

def load_model_and_tokenizer():
    """Load model and tokenizer using vLLM"""
    print(f"🔧 Loading vLLM engine: {EXPERIMENT_CONFIG['model_path']}")

    llm = LLM(
        model=EXPERIMENT_CONFIG['model_path'],
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        max_model_len=16384
    )

    print("✅ vLLM engine loaded successfully")
    return llm


# ==========================================
# 3. File Reading and Shuffling
# ==========================================

def read_and_shuffle_retrieval_results(file_path):
    """Read data from retrieval results file and shuffle randomly"""
    print(f"📄 Reading file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for idx, item in enumerate(data):
        passages = item.get("passages", [])

        results.append({
            "original_id": item.get("original_id", item.get("id", f"unknown_{idx}")),
            "original_index": item.get("original_index", idx),
            "question": item.get("question", ""),
            "passages": passages,
            "shuffled_index": idx,
        })

    print(f"📊 Read {len(results)} questions")
    print(f"🎲 Shuffling with seed {EXPERIMENT_CONFIG['random_seed']}...")
    random.shuffle(results)

    for i, item in enumerate(results):
        item["shuffled_index"] = i

    print(f"First 5 original IDs after shuffling: {[r['original_id'] for r in results[:5]]}")
    print(f"First 5 shuffled positions: {[r['shuffled_index'] for r in results[:5]]}")

    return results


# ==========================================
# 4. Prompt Construction Function
# ==========================================

def build_prompt_and_messages(question, passages_list, tokenizer):
    """Build prompt using original logic and templates"""
    formatted_docs = []
    for i, p in enumerate(passages_list[:5]):
        formatted_docs.append(f"[[SOURCE DOCUMENT {i + 1}]]\n{p.strip()}")
    P = "\n\n".join(formatted_docs)

    system_prompt = """
You are a HIGH-FIDELITY DATA EXTRACTOR. Your sole mission is to find evidence and answer questions using ONLY the provided source documents. You must operate like a literal scanner, not a creative writer.

=== EXECUTION PROCESS ===
Step 1: Scan [[SOURCE DOCUMENT 1]] through [[SOURCE DOCUMENT 5]] for facts answering the question.
Step 2: Copy-paste the evidence. Each quote must be a SINGLE, CONTINUOUS string from the source. 
Step 3: Write a concise Answer based ONLY on the collected Quote(s).

=== MANDATORY OUTPUT FORMAT ===
1. IF EVIDENCE EXISTS:
Quote(s): ["verbatim quote 1", "verbatim quote 2", ...]
Answer: Concise answer text

2. IF NO EVIDENCE EXISTS:
Quote(s): []
Answer: I cannot answer based on the provided documents.

=== CRITICAL EXTRACTION RULES (STRICT ENFORCEMENT) ===
1. VERBATIM & NO STITCHING: Quotes must be 100% identical to the source (including punctuation/typos). You are PROHIBITED from merging non-contiguous sentences into one string. If you need facts from different locations, use separate strings.
2. LINEAR & SOURCE PURITY: Quotes must follow the EXACT order of the source documents. Never invert sequence. Each string must belong to one document only; do not blend Text from Doc A and Doc B.
3. SEMANTIC INTEGRITY: Always include the subject/pronoun referent (e.g., if a sentence uses "He", include the preceding sentence that identifies who "He" is). A quote must be a stand-alone proof; do not cut mid-sentence.
4. QUOTE-ANSWER ANCHORING: The 'Answer' must be a synthesis of your 'Quote(s)' list ONLY. If a fact cannot be found in your 'Quote(s)', it MUST be excluded from the 'Answer'.
5. ZERO INFERENCE & NO SUMMARY: No outside knowledge, no logical leaps, and no paraphrasing. The 'Quote(s)' section is for Copy-Pasting only. Any character mismatch or summary in quotes is a system failure.
6. NO CONVERSATIONAL FILLER: Output only the "Quote(s)" and "Answer" fields. Do not add intro/outro text.

=== FORMATTING GUARDRAIL ===
- Use standard double quotes for Quote strings.
- Ensure the Quote(s) list is output BEFORE the Answer.
"""

    few_shot_content = """
===== MULTI-DOCUMENT RETRIEVAL EXAMPLES =====

[EXAMPLE 1: SYNTHESIZED ANSWER FROM MULTIPLE DOCUMENTS]
Documents: 
[[SOURCE DOCUMENT 1]] The new orbital satellite, Helios-X, was successfully launched from the Kennedy Space Center.
[[SOURCE DOCUMENT 2]] Meteorological conditions remained stable.
[[SOURCE DOCUMENT 3]] The primary mission of the craft is to monitor solar flare activity over a five-year period.
[[SOURCE DOCUMENT 4]] Ground teams confirmed signal acquisition.
[[SOURCE DOCUMENT 5]] Funding for Helios-X was primarily secured through a $200 million grant from the International Space Agency.
Question: What are the key details regarding the launch, purpose, and financial backing of the Helios-X mission?
Quote(s): ["The new orbital satellite, Helios-X, was successfully launched from the Kennedy Space Center.","The primary mission of the craft is to monitor solar flare activity over a five-year period.","Funding for Helios-X was primarily secured through a $200 million grant from the International Space Agency."]
Answer: Helios-X was launched from the Kennedy Space Center to monitor solar flare activity over a five-year period, with $200 million in funding from the International Space Agency.

[EXAMPLE 2: SINGLE DOCUMENT WITH PRONOUN RESOLUTION]
Documents: 
[[SOURCE DOCUMENT 1]] Ocean currents play a major role in global climate.
[[SOURCE DOCUMENT 2]] The Gulf Stream is a powerful warm ocean current. It originates in the Gulf of Mexico.
[[SOURCE DOCUMENT 3]] Most currents are driven by wind patterns.
[[SOURCE DOCUMENT 4]] Deep sea temperatures remain stable.
[[SOURCE DOCUMENT 5]] Satellite data helps track current movements.
Question: Where does the Gulf Stream originate?
Quote(s): ["The Gulf Stream is a powerful warm ocean current. It originates in the Gulf of Mexico."]
Answer: It originates in the Gulf of Mexico.

[EXAMPLE 3: STRICT REFUSAL (NO EVIDENCE)]
Documents: 
[[SOURCE DOCUMENT 1]] Mount Everest reaches an elevation of 29,032 feet.
[[SOURCE DOCUMENT 2]] Climbing Everest requires extensive preparation.
[[SOURCE DOCUMENT 3]] The first successful ascent was in 1953.
[[SOURCE DOCUMENT 4]] Sherpas play a critical role as guides.
[[SOURCE DOCUMENT 5]] Base Camp is situated at 17,598 feet.
Question: How many oxygen tanks does a typical climber carry?
Quote(s): []
Answer: I cannot answer based on the provided documents.
"""

    full_system_content = system_prompt + "\n" + few_shot_content

    user_message = (
        f"### INPUT DOCUMENTS (1-5) ###\n"
        f"{P}\n"
        f"### END OF DOCUMENTS ###\n\n"
        f"TARGET QUESTION: {question}\n\n"
        f"Quote(s): ["
    )

    messages = [
        {"role": "system", "content": full_system_content},
        {"role": "user", "content": user_message}
    ]

    f_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    ).strip()

    return f_prompt


# ==========================================
# 5. Matching Process
# ==========================================

def unified_matching_process(quote_text_raw, passages_list):
    raw_quotes = re.findall(r'[\"\'](.*?)[\"\'](?=\s*[,\]]|\s*$)', quote_text_raw.replace('\n', ' '))
    extracted_list = [q.strip(' ,[]\\\'\"').strip() for q in raw_quotes if q.strip()]

    if not extracted_list:
        return "[]", []

    matched_ids = []
    for q_text in extracted_list:
        found_id = "UNKNOWN"
        clean_q = q_text.strip()

        for idx, passage in enumerate(passages_list[:5]):
            if clean_q in passage:
                found_id = f"doc_{idx + 1}"
                break

            q_norm = " ".join(clean_q.split())
            p_norm = " ".join(passage.split())
            if q_norm in p_norm:
                found_id = f"doc_{idx + 1}"
                break

            if "..." in clean_q:
                frags = [f.strip() for f in clean_q.split("...") if len(f.strip()) > 5]
                if frags and all(f in passage for f in frags):
                    found_id = f"doc_{idx + 1}"
                    break

            if fuzz and fuzz.partial_ratio(clean_q, passage) > 90:
                found_id = f"doc_{idx + 1}"
                break

        matched_ids.append(found_id)

    doc_ids_str = "[" + ", ".join([f'"{d}"' for d in matched_ids]) + "]"
    return doc_ids_str, extracted_list


# ==========================================
# 6. Main Processing Function
# ==========================================

def process_experiment(input_path, output_path, llm):
    """Execute experiment processing with vLLM batch generation and structured parsing"""
    print(f"\n{'=' * 60}")
    print(f"Processing: {EXPERIMENT_CONFIG['name']}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"{'=' * 60}")

    all_data = read_and_shuffle_retrieval_results(input_path)
    tokenizer = llm.get_tokenizer()

    prompts = []
    for data_item in all_data:
        prompt = build_prompt_and_messages(data_item['question'], data_item['passages'], tokenizer)
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=EXPERIMENT_CONFIG['temperature'],
        max_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
        repetition_penalty=1.1,
        stop_token_ids=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    )

    print(f"🚀 正在使用 vLLM 一次性批量推理 {len(prompts)} 条数据...")

    # ⏱️ 严格计时开始：包含生成与物理定位全过程
    stage_start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    # ⏱️ 严格计时结束：此时所有样本处理完毕
    stage_end_time = time.time()
    total_time = stage_end_time - stage_start_time
    total_count = len(all_data)

    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # 用于极值统计的变量
    min_tokens = float('inf')
    max_tokens = float('-inf')
    min_token_id = None
    max_token_id = None

    # 第一遍循环：计算总 Token 并找出 Token 极值
    temp_sample_tokens = []
    for i, output in enumerate(outputs, 1):
        data_item = all_data[i - 1]
        p_tokens = len(output.prompt_token_ids)
        c_tokens = len(output.outputs[0].token_ids)
        sample_total_tokens = p_tokens + c_tokens

        total_prompt_tokens += p_tokens
        total_completion_tokens += c_tokens
        temp_sample_tokens.append(sample_total_tokens)

        if sample_total_tokens < min_tokens:
            min_tokens = sample_total_tokens
            min_token_id = data_item['original_id']

        if sample_total_tokens > max_tokens:
            max_tokens = sample_total_tokens
            max_token_id = data_item['original_id']

    grand_total_tokens = total_prompt_tokens + total_completion_tokens

    # 第二遍循环：解析结果、估算单样本耗时并组装
    for i, output in enumerate(outputs, 1):
        data_item = all_data[i - 1]
        p_tokens = len(output.prompt_token_ids)
        c_tokens = len(output.outputs[0].token_ids)
        sample_total_tokens = temp_sample_tokens[i - 1]

        # 估算单个样本的耗时（基于总耗时与该样本 Token 占总 Token 的比例加权）
        sample_time = (sample_total_tokens / grand_total_tokens) * total_time if grand_total_tokens > 0 else (total_time / total_count)

        try:
            raw_gen = output.outputs[0].text.strip()
            clean_content = raw_gen.replace("Quote(s):", "").strip()

            if "Answer:" in clean_content:
                parts = clean_content.split("Answer:")
                quote_part = parts[0].strip()
                answer_part = parts[1].strip()

                doc_ids_str, extracted_list = unified_matching_process(quote_part, data_item['passages'])
                full_response = f"Quote(s): {extracted_list}\nDoc_IDs: {doc_ids_str}\nAnswer: {answer_part}"
            else:
                doc_ids_str, extracted_list = unified_matching_process(clean_content, data_item['passages'])
                fallback_answer = "(No answer tag found) I cannot answer based on the provided documents."
                full_response = f"Quote(s): {extracted_list}\nDoc_IDs: {doc_ids_str}\nAnswer: {fallback_answer}"

            parts_q = full_response.split("Doc_IDs:")
            quo_content = parts_q[0].replace("Quote(s):", "").strip()

            parts_a = parts_q[1].split("Answer:")
            ids_content = parts_a[0].strip()
            ans_content = parts_a[1].strip()

            result = {
                "original_id": data_item['original_id'],
                "original_index": data_item['original_index'],
                "shuffled_index": data_item['shuffled_index'],
                "question": data_item['question'],
                "retrieved_passages": data_item['passages'],
                "final_quote(s)": quo_content,
                "doc_id(s)": ids_content,
                "final_answer": ans_content,
                "processing_order": i,
                "token_statistics": {
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "total_tokens": sample_total_tokens
                },
                "estimated_time_seconds": round(sample_time, 4)
            }

            results.append(result)

        except Exception as e:
            print(f"❌ Processing failed for {data_item['original_id']}: {str(e)}")
            results.append({
                "original_id": data_item['original_id'],
                "original_index": data_item['original_index'],
                "shuffled_index": data_item['shuffled_index'],
                "question": data_item['question'],
                "error": str(e),
                "final_quote(s)": "None",
                "doc_id(s)": "[]",
                "final_answer": "ERROR",
                "processing_order": i,
                "token_statistics": {
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "total_tokens": sample_total_tokens
                },
                "estimated_time_seconds": round(sample_time, 4)
            })

    # 计算耗时极值
    all_times = [r["estimated_time_seconds"] for r in results]
    min_time = min(all_times) if all_times else 0
    max_time = max(all_times) if all_times else 0

    avg_time_per_sample = total_time / total_count if total_count > 0 else 0
    avg_tokens_per_sample = grand_total_tokens / total_count if total_count > 0 else 0

    print("\n" + "=" * 50)
    print("      SCAR 阶段一 (Stage 1) 性能、Token 与极值统计报告      ")
    print("=" * 50)
    print(f"总样本数            : {total_count} 条")
    print(f"总耗时              : {total_time:.4f} 秒 (约 {total_time / 60:.2f} 分钟)")
    print(f"平均每个样本耗时    : {avg_time_per_sample:.4f} 秒")
    print(f"  - 单个样本最短耗时  : {min_time:.4f} 秒")
    print(f"  - 单个样本最长耗时  : {max_time:.4f} 秒")
    print("-" * 50)
    print(f"总输入 Token (Prompt)       : {total_prompt_tokens:,} tokens")
    print(f"总输出 Token (Completion)    : {total_completion_tokens:,} tokens")
    print(f"总消耗 Token (Grand Total)   : {grand_total_tokens:,} tokens")
    print(f"平均每个样本 Token 消耗      : {avg_tokens_per_sample:.1f} tokens")
    print(f"  - 单个样本最少 Token: {min_tokens} tokens (ID: {min_token_id})")
    print(f"  - 单个样本最多 Token: {max_tokens} tokens (ID: {max_token_id})")
    print("=" * 50)

    final_output = {
        "performance_statistics": {
            "total_samples": total_count,
            "total_time_seconds": round(total_time, 4),
            "average_time_per_sample_seconds": round(avg_time_per_sample, 4),
            "min_sample_time_seconds": round(min_time, 4),
            "max_sample_time_seconds": round(max_time, 4)
        },
        "token_statistics": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "grand_total_tokens": grand_total_tokens,
            "average_tokens_per_sample": round(avg_tokens_per_sample, 2),
            "min_sample_tokens": min_tokens,
            "max_sample_tokens": max_tokens
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Results saved to: {output_path}")
    return final_output


# ==========================================
# 7. Cleanup Function
# ==========================================
def cleanup_and_exit(llm, exit_process=True):
    print("\n🧹 Performing deep memory cleanup...")

    try:
        if llm is not None:
            del llm

        for i in range(3):
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("✅ Memory cleanup completed")

    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

    if exit_process:
        print("🔄 Exiting process for complete state reset...")
        import sys
        sys.exit(0)


# ==========================================
# 8. Main Program
# ==========================================

def main():
    global start_time
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description="Forward Generation - vLLM Version")
    parser.add_argument("--input-dir", type=str,
                        default="/root/autodl-tmp/",
                        help="Input file directory")
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/",
                        help="Output file directory")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Maximum number of questions to process")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    setup_environment()
    llm = load_model_and_tokenizer()

    input_path = os.path.join(args.input_dir, EXPERIMENT_CONFIG['input_filename'])
    output_path = os.path.join(args.output_dir, EXPERIMENT_CONFIG['output_filename'])

    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
        cleanup_and_exit(llm)
        return

    try:
        process_experiment(input_path, output_path, llm)
    except Exception as e:
        print(f"❌ Experiment execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

    cleanup_and_exit(llm, exit_process=True)


if __name__ == "__main__":
    print("=" * 60)
    print("Forward Generation Experiment - vLLM Version")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    main()