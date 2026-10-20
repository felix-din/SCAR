"""
Stage 3: QA Alignment Audit - vLLM Version (with Full Time, Token Statistics & Extremes)
Experiment: stage3_alignment_audit_mn12b_batch
Model: Mistral-NeMo-12B-Instruct
Hardware: RTX 5090 (vLLM Accelerated)
"""

import os
import re
import json
import torch
import random
import warnings
import argparse
import gc
import time
import ast
from datetime import datetime
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ==========================================
# Experiment Configuration
# ==========================================

EXPERIMENT_CONFIG = {
    "name": "stage3_alignment_audit_mn12b_batch_v3_vllm",
    "model_path": "/root/autodl-tmp/Mistral-NeMo-12B-Instruct",
    "random_seed": 1029,
    "input_filename": "stage2_input_qwen_output_stage3_input.json",
    "output_filename": "stage3_output_mn_final_output.json",
    "max_new_tokens": 128,
    "temperature": 0.0
}


# ==========================================
# 1. Environment Setup
# ==========================================

def setup_environment():
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore")
    print(f"🚀 Experiment: {EXPERIMENT_CONFIG['name']}")
    print(f"📌 Random Seed: {EXPERIMENT_CONFIG['random_seed']}")
    random.seed(EXPERIMENT_CONFIG['random_seed'])
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")


# ==========================================
# 2. Alignment Model Loading (vLLM 引擎初始化)
# ==========================================

def load_alignment_model():
    print(f"⚖️ Loading Aligner with vLLM: {EXPERIMENT_CONFIG['model_path']}")
    
    # 针对 Mistral-NeMo 在 vLLM 中的显存和长度优化配置
    llm = LLM(
        model=EXPERIMENT_CONFIG['model_path'],
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192  # 结合显存计算的安全上下文长度
    )
    print("✅ Model loaded successfully with vLLM engine")
    return llm


# ==========================================
# 3. JSON Extractor (级联回退逻辑)
# ==========================================
def extract_json_from_llm(text):
    if not text or '{' not in text:
        return None

    md_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if md_blocks:
        for content in md_blocks:
            result = try_parse_all_methods(content.strip())
            if result is not None:
                return result

    json_candidates = re.findall(r'\{.*?\}', text, re.DOTALL)
    if json_candidates:
        for content in json_candidates:
            result = try_parse_all_methods(content.strip())
            if result is not None:
                return result

    return None


def try_parse_all_methods(content):
    if not content:
        return None

    try:
        return json.loads(content)
    except:
        pass

    try:
        temp_content = re.sub(r'\btrue\b', 'True', content)
        temp_content = re.sub(r'\bfalse\b', 'False', temp_content)
        temp_content = re.sub(r'\bnull\b', 'None', temp_content)
        return ast.literal_eval(temp_content)
    except:
        pass

    try:
        clean_content = re.sub(r'[\x00-\x1F\x7F]', '', content)
        clean_content = clean_content.replace('\n', '\\n').replace('\r', '')

        try:
            return json.loads(clean_content)
        except:
            pass

        fixed = re.sub(r'(?<!\\)"', r'\"', clean_content)
        fixed = re.sub(r'(?<=\{)\s*\\"', '"', fixed)
        fixed = re.sub(r'\\"\s*(?=\})', '"', fixed)
        fixed = re.sub(r'\\"\s*:\s*\\"', '": "', fixed)
        fixed = re.sub(r'\\"\s*,\s*\\"', '", "', fixed)

        fixed = fixed.strip()
        if not fixed.startswith('{'): fixed = '{' + fixed
        if not fixed.endswith('}'): fixed = fixed + '}'

        try:
            return json.loads(fixed)
        except:
            try:
                return ast.literal_eval(fixed)
            except:
                pass
    except:
        return None


# ==========================================
# 4. Main Processing Function (带耗时、Token 统计及极值)
# ==========================================

def process_alignment_batch(input_path, output_path, llm):
    print(f"\n{'=' * 60}")
    print(f"Starting vLLM Batch Alignment Check")
    print(f"Input: {input_path}")
    print(f"{'=' * 60}")

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    all_samples = input_data.get("results", [])
    total_count = len(all_samples)
    tokenizer = llm.get_tokenizer()

    SKIP_AUDIT_CATEGORIES = {"ACTIVE_HONESTY", "PASSIVE_HONESTY", "CITATION_HALLUCINATION", "CONTENT_HALLUCINATION"}
    MODEL_FORMAT_ERROR_TAG = "MODEL_FORMAT_ERROR"

    final_checked_results = [None] * total_count
    audit_indices = []
    prepared_prompts = []

    print("🔄 正在预处理并构建所有样本的审计 Prompt...")
    for idx, item in enumerate(all_samples):
        stage2_audit = item.get("stage2_audit", {})
        category = stage2_audit.get("category", "UNKNOWN_CATEGORY")

        if category in SKIP_AUDIT_CATEGORIES:
            judgment = "SKIPPED"
            new_item = item.copy()
            new_item["stage3_audit"] = {
                "judgment": judgment,
                "reason": f"Skipped: Category '{category}' finalized in Stage 2.",
                "raw_output": "SKIPPED_BY_LOGIC"
            }
            new_item["token_statistics"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            new_item["estimated_time_seconds"] = 0.0
            final_checked_results[idx] = new_item
        else:
            audit_indices.append(idx)
            passages = item.get("retrieved_passages", [])
            q = item.get("question", "N/A")

            if category == MODEL_FORMAT_ERROR_TAG:
                a = stage2_audit.get("final_answer", "no answer")
                s2_evidence = "No specific evidence extracted."
            else:
                a = stage2_audit.get("final_answer_verified", "no answer")
                s2_evidence = stage2_audit.get("evidence", "no evidence")

            target_context = ""
            ids_content = item.get("doc_id(s)", "[]")
            try:
                raw_ids = json.loads(ids_content) if isinstance(ids_content, str) else ids_content
            except:
                raw_ids = []

            clean_ids = [str(d).strip() for d in raw_ids if str(d).strip() not in ["UNKNOWN"]]
            valid_idxs = []
            if len(clean_ids) > 0:
                for d in clean_ids:
                    match = re.search(r'doc_(\d+)', d, re.IGNORECASE)
                    if match:
                        doc_num = int(match.group(1)) - 1
                        if 0 <= doc_num < len(passages):
                            valid_idxs.append(doc_num)

                unique_idxs = sorted(list(set(valid_idxs)))
                if unique_idxs:
                    target_context = "\n\n".join([f"[[DOC {i_idx + 1}]] {passages[i_idx]}" for i_idx in unique_idxs])
                else:
                    target_context = "\n\n".join([f"[[DOC {j + 1}]] {p}" for j, p in enumerate(passages)])
            else:
                target_context = "\n\n".join([f"[[DOC {j + 1}]] {p}" for j, p in enumerate(passages)])

            json_format = '{"judgment": "ALIGNED/MISALIGNED", "reason": "Concise analysis of the QA alignment and factual basis."}'

            audit_prompt = f"""
You are a Senior QA Alignment Auditor. Your task is to perform a final semantic check: Does the 'Model Answer' provide a valid and factually grounded response to the 'Question'?

### CORE AUDIT MISSION:
1. **QA Logic**: Does the Answer directly satisfy the intent of the Question?
2. **Factual Grounding**: Cross-check facts (names, dates, numbers, entities) against 'Source Documents'.
   - **Strict Rule**: If a specific fact (e.g., a date or a name) is NOT mentioned or cannot be logically inferred from the documents, it is an 'Unverified Hallucination' (MISALIGNED).
3. **No Overclaim**: Did the model 'hallucinate' conflicting information or claims that contradict the source documents?

### 💡 AUDIT NOTE ON REFERENCES:
- If 'Stage 2 Evidence' is 'No specific evidence extracted' or empty, you MUST perform a full semantic scan of all provided 'Source Documents'.
- Do not penalize the answer solely because pre-extracted evidence is missing.

### JUDGMENT STANDARDS (Calibration):
- **ALIGNED**: The Answer's core facts are accurate and supported. 
  - *Note*: Ignore minor stylistic differences, polite fillers, or safe rephrasing (e.g., "According to the text...").
- **MISALIGNED**:
  - **Factual Error**: Explicitly contradicts the source documents.
  - **Unsupported**: Claims specific data points, entities, or events that are completely absent from the source.
  - **Irrelevant**: The answer talks about the topic but fails to address the specific question asked.

### CRITICAL OUTPUT INSTRUCTION:
1. You must output the result in the following JSON format: {json_format}
2. You MUST start your response with this JSON block.
3. Do not provide any conversational filler or analysis BEFORE the JSON.
4. Detailed step-by-step analysis (Evidence Mapping) MUST be provided AFTER the JSON block to justify your decision.
"""

            user_content = (
                f"### 🎯 AUDIT TARGET\n"
                f"**QUESTION**: {q}\n"
                f"**MODEL ANSWER**: {a}\n\n"
                f"### 📚 REFERENCE (Stage 2 Background)\n"
                f"- **Extracted Evidence**: {s2_evidence}\n"
                f"- **Source Documents**: {target_context}\n\n"
                f"---"
                f"Task: Based on the reference provided, does the Model Answer correctly and accurately address the Question?"
            )

            messages = [
                {"role": "system", "content": audit_prompt},
                {"role": "user", "content": user_content}
            ]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).strip()
            prepared_prompts.append(formatted_prompt)

    sampling_params = SamplingParams(
        temperature=EXPERIMENT_CONFIG['temperature'],
        max_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
        stop_token_ids=[tokenizer.eos_token_id]
    )

    print(f"🚀 正在使用 vLLM 批量高效处理 {len(prepared_prompts)} 条审计请求...")

    # ⏱️ 严格计时开始
    stage_start_time = time.time()
    outputs = llm.generate(prepared_prompts, sampling_params)
    stage_end_time = time.time()

    total_time = stage_end_time - stage_start_time

    # 初始化 Token 统计与极值追踪
    total_prompt_tokens = 0
    supposed_completion_tokens = 0
    min_tokens = float('inf')
    max_tokens = float('-inf')
    min_token_id = None
    max_token_id = None

    temp_tokens_list = []
    for output in outputs:
        p_tokens = len(output.prompt_token_ids)
        c_tokens = len(output.outputs[0].token_ids)
        sample_total_tokens = p_tokens + c_tokens

        total_prompt_tokens += p_tokens
        supposed_completion_tokens += c_tokens
        temp_tokens_list.append(sample_total_tokens)

    grand_total_tokens = total_prompt_tokens + supposed_completion_tokens

    # 遍历结果写回对应索引
    output_idx = 0
    for audit_idx in audit_indices:
        item = all_samples[audit_idx]
        output = outputs[output_idx]
        output_idx += 1

        p_tokens = len(output.prompt_token_ids)
        c_tokens = len(output.outputs[0].token_ids)
        sample_total_tokens = p_tokens + c_tokens

        # 追踪样本 Token 极值
        item_id = item.get("original_id", f"idx_{audit_idx}")
        if sample_total_tokens < min_tokens:
            min_tokens = sample_total_tokens
            min_token_id = item_id
        if sample_total_tokens > max_tokens:
            max_tokens = sample_total_tokens
            max_token_id = item_id

        # 估算单个样本耗时
        sample_time = (sample_total_tokens / grand_total_tokens) * total_time if grand_total_tokens > 0 else (total_time / len(audit_indices))

        raw_gen = output.outputs[0].text.strip()
        res_json = extract_json_from_llm(raw_gen)

        if res_json:
            raw_judgment = str(res_json.get("judgment", "")).strip().upper().replace('.', '')
            final_judgment = raw_judgment if raw_judgment else "EMPTY_JUDGMENT"
        else:
            final_judgment = "PARSE_ERROR"

        processed_item = item.copy()
        processed_item["stage3_audit"] = {
            "judgment": final_judgment,
            "reason": res_json.get("reason", "No reason provided") if res_json else raw_gen,
            "raw_output": raw_gen
        }
        processed_item["token_statistics"] = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": sample_total_tokens
        }
        processed_item["estimated_time_seconds"] = round(sample_time, 4)
        final_checked_results[audit_idx] = processed_item

    # 计算时间极值
    all_times = [r["estimated_time_seconds"] for r in final_checked_results if r is not None and "estimated_time_seconds" in r]
    min_time = min(all_times) if all_times else 0
    max_time = max(all_times) if all_times else 0

    avg_time_per_sample = total_time / total_count if total_count > 0 else 0
    avg_tokens_per_sample = grand_total_tokens / total_count if total_count > 0 else 0

    misaligned_count = len([r for r in final_checked_results if r and r.get("stage3_audit", {}).get("judgment") == "MISALIGNED"])
    parse_errors_count = len([r for r in final_checked_results if r and r.get("stage3_audit", {}).get("judgment") == "PARSE_ERROR"])

    # 打印标准统计报告
    print("\n" + "=" * 50)
    print("   SCAR 阶段三 (Stage 3 Audit) 性能、Token 与极值统计报告   ")
    print("=" * 50)
    print(f"总样本数            : {total_count} 条")
    print(f"总耗时              : {total_time:.4f} 秒 (约 {total_time / 60:.2f} 分钟)")
    print(f"平均每个样本耗时    : {avg_time_per_sample:.4f} 秒")
    print(f"  - 单个样本最短耗时  : {min_time:.4f} 秒")
    print(f"  - 单个样本最长耗时  : {max_time:.4f} 秒")
    print("-" * 50)
    print(f"总输入 Token (Prompt)      : {total_prompt_tokens:,} tokens")
    print(f"总输出 Token (Completion) : {supposed_completion_tokens:,} tokens")
    print(f"总消耗 Token (Grand Total): {grand_total_tokens:,} tokens")
    print(f"平均每个样本 Token 消耗   : {avg_tokens_per_sample:.1f} tokens")
    print(f"  - 单个样本最少 Token: {min_tokens} tokens (ID: {min_token_id})")
    print(f"  - 单个样本最多 Token: {max_tokens} tokens (ID: {max_token_id})")
    print("=" * 50)
    print(f"✅ 审计完成! Misaligned: {misaligned_count}, Parse Errors: {parse_errors_count}")

    final_output = {
        "performance_statistics": {
            "total_samples": total_count,
            "audited_samples": len(audit_indices),
            "total_time_seconds": round(total_time, 4),
            "average_time_per_sample_seconds": round(avg_time_per_sample, 4),
            "min_sample_time_seconds": round(min_time, 4),
            "max_sample_time_seconds": round(max_time, 4)
        },
        "token_statistics": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": supposed_completion_tokens,
            "grand_total_tokens": grand_total_tokens,
            "average_tokens_per_sample": round(avg_tokens_per_sample, 2),
            "min_sample_tokens": min_tokens,
            "max_sample_tokens": max_tokens
        },
        "summary": {
            "total": total_count,
            "audited": len(audit_indices),
            "parse_errors": parse_errors_count,
            "misaligned_in_audited": misaligned_count,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        "results": final_checked_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Results successfully saved to: {output_path}")


# ==========================================
# 5. Cleanup & Exit
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
# Main Entry
# ==========================================

def main():
    setup_environment()
    llm = load_alignment_model()

    INPUT_PATH = "/root/autodl-tmp/" + EXPERIMENT_CONFIG['input_filename']
    OUTPUT_PATH = "/root/autodl-tmp/" + EXPERIMENT_CONFIG['output_filename']

    try:
        process_alignment_batch(INPUT_PATH, OUTPUT_PATH, llm)
    except Exception as e:
        print(f"❌ Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

    cleanup_and_exit(llm, exit_process=True)


if __name__ == "__main__":
    main()