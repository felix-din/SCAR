"""
Stage 2 Audit Script - vLLM Version (with Full Time, Token Statistics & Extremes)
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

# ==========================================
# Experiment Configuration
# ==========================================

EXPERIMENT_CONFIG = {
    "name": "stage2_qwen_audit_results",
    "audit_model_path": "./Mistral-NeMo-12B-Instruct",
    "random_seed": 1001,
    "input_filename": "stage1_output_stage2_input.json",
    "output_filename": "stage2_input_mn_output_stage3_input.json",
    "max_new_tokens": 128,
    "temperature": 0.0
}


# ==========================================
# 1. Environment Setup
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
# 2. Audit Model Loading (vLLM 引擎初始化)
# ==========================================

def load_audit_model():
    print(f"⚖️ Loading Auditor with vLLM: {EXPERIMENT_CONFIG['audit_model_path']}")

    llm = LLM(
        model=EXPERIMENT_CONFIG['audit_model_path'],
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192
    )
    print("✅ Auditor vLLM engine loaded successfully")
    return llm


# ==========================================
# 3. Main Processing Function (带耗时、Token 统计及极值)
# ==========================================

def process_audit(input_path, output_path, llm):
    print(f"\n{'=' * 60}")
    print(f"Starting Stage 2 Audit: {EXPERIMENT_CONFIG['name']}")
    print(f"Input: {input_path}")
    print(f"{'=' * 60}")

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    all_results = input_data.get("results", [])
    total_count = len(all_results)
    tokenizer = llm.get_tokenizer()

    prepared_items = []

    print("🔄 正在预处理并构建所有样本的审计 Prompt...")
    for i, item in enumerate(all_results, 1):
        try:
            question = item.get("question", "")
            passages = item.get("retrieved_passages", [])
            ids_content = item.get("doc_id(s)", "[]")
            final_answer = item.get("final_answer", "")

            # --- 1. ID 解析逻辑 ---
            raw_ids = json.loads(ids_content)
            invalid_markers = ["UNKNOWN"]
            clean_ids = [d.strip() for d in raw_ids if d.strip() not in invalid_markers]
            num_raw = len(raw_ids)
            num_clean = len(clean_ids)

            # --- 2. 状态判定 ---
            is_format_collapse = "(No answer tag found)" in final_answer
            refusal_keywords = ["cannot answer"]
            is_real_refusal = not is_format_collapse and any(kw in final_answer.lower() for kw in refusal_keywords)

            # --- 3. 初始化维度标记 ---
            audit_meta = {
                "source": "FORMAT_COLLAPSE" if is_format_collapse else "FORMAT_NORMAL",
                "intent": "",
                "category": ""
            }

            # --- 4. 逻辑分流与级联拦截 ---
            if is_format_collapse:
                audit_meta["intent"] = "PASSIVE_REFUSAL"
            elif is_real_refusal:
                audit_meta["intent"] = "ACTIVE_REFUSAL"
            else:
                audit_meta["intent"] = "GIVE_ANSWER"

            # 物理层级联拦截 (全是 UNKNOWN 的垃圾引用)
            if num_raw >= 2 and num_clean == 0:
                audit_meta["category"] = "CITATION_HALLUCINATION"
                audit_meta["decision"] = "REJECT"
                audit_meta["reason"] = f"Physical Block: {num_raw} IDs provided but all are UNKNOWN/Invalid."
                audit_meta['audit_raw_output'] = "PHYSICAL_BLOCK_SKIPPED"

                prepared_items.append({
                    "item": item,
                    "audit_meta": audit_meta,
                    "bypass_model": True
                })
                continue

            # --- 5. 确定 Context 范围 ---
            if num_clean == 0:
                context = "\n\n".join([f"[[DOC {j + 1}]] {p}" for j, p in enumerate(passages)])
            else:
                valid_idxs = []
                for d in clean_ids:
                    match = re.search(r'doc_(\d+)', d)
                    if match and 0 <= (int(match.group(1)) - 1) < len(passages):
                        valid_idxs.append(int(match.group(1)) - 1)
                unique_idxs = sorted(list(set(valid_idxs)))
                context = "\n\n".join(
                    [f"[[DOC {idx + 1}]] {passages[idx]}" for idx in unique_idxs]) if unique_idxs else "\n\n".join(
                    [f"[[DOC {j + 1}]] {p}" for j, p in enumerate(passages)])

            has_citation = (num_clean > 0)
            json_format = '{"decision": "PASS/REJECT", "final_answer_verified": "...", "reason": "..."}'

            # --- 6. 构建针对性 Audit Prompt ---
            if audit_meta["intent"] in ["PASSIVE_REFUSAL", "ACTIVE_REFUSAL"]:
                if has_citation:
                    audit_prompt = f"""You are a senior Auditor. Your goal is to determine if the question can be answered strictly based on the provided [References].

【Question】: {question}
【References】: 
{context}

### Audit Logic:
1. **Extraction Check**: Based on the question, can you find/extract a specific answer from the provided [References]?
2. **Decision & Action**: 
   - **YES**: Set decision to "PASS", extract a concise answer to final_answer_verified, and provide your reason.
   - **NO**: Set decision to "REJECT", set final_answer_verified to "NOT_FOUND_IN_ID", and state the reason as "Citation Hallucination: IDs provided but no answer found in them."

Output JSON: {json_format}"""
                else:
                    audit_prompt = f"""You are a senior Auditor. Your goal is to determine if the question can be answered strictly based on the provided [Full Context].

【Question】: {question}
【Full Context】: 
{context}

### Audit Logic:
1. **Extraction Check**: Based on the question, can you find/extract a specific answer from the provided [Full Context]?
2. **Decision & Action**: 
   - **YES**: Set decision to "PASS", extract a concise answer to final_answer_verified, and provide your reason.
   - **NO**: Set decision to "PASS", set final_answer_verified to "NOT_FOUND_IN_FULL_CONTEXT", and state the reason as "Correct silence: Full context does not have answer."

Output JSON: {json_format}"""

            elif audit_meta["intent"] == "GIVE_ANSWER":
                if has_citation:
                    audit_prompt = f"""You are a senior Auditor. Your goal is to determine if the [Model Answer] can be found in or is strictly supported by the provided [References].

【Question】: {question}
【Model Answer】: {final_answer}
【References】: 
{context}

### Audit Logic:
1. **Verification Check**: Based on the question, can the [Model Answer] be found or directly supported by the provided [References]?
2. **Decision & Action**: 
   - **YES**: Set decision to "PASS", set final_answer_verified to the original [Model Answer], and provide your reason.
   - **NO**: Set decision to "REJECT", set final_answer_verified to "NOT_FOUND_IN_ID", and state the reason as "Citation Hallucination: IDs provided but no answer found in them."

Output JSON: {json_format}"""
                else:
                    audit_prompt = f"""You are a senior Auditor. Your goal is to determine if the [Model Answer] can be found in or is supported by the provided [Full Context].

【Question】: {question}
【Model Answer】: {final_answer}
【Full Context】: 
{context}

### Audit Logic:
1. **Verification Check**: Based on the question, can the [Model Answer] be found or supported by the provided [Full Context]?
2. **Decision & Action**:
   - **YES**: Set decision to "PASS", set final_answer_verified to the original [Model Answer], and provide your reason.
   - **NO**: Set decision to "REJECT", set final_answer_verified to "NOT_FOUND_IN_FULL_CONTEXT", and state the reason as "Content Hallucination: The answer is not supported by the full context."

Output JSON: {json_format}"""

            messages = [{"role": "user", "content": audit_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,           # 👈 改为 True
                add_generation_prompt=True
            ).strip()

            prepared_items.append({
                "item": item,
                "audit_meta": audit_meta,
                "has_citation": has_citation,
                "prompt": formatted_prompt,
                "bypass_model": False
            })

        except Exception as e:
            item["stage2_audit"] = {"decision": "SYSTEM_ERROR", "reason": str(e)}
            prepared_items.append({
                "item": item,
                "audit_meta": {"decision": "SYSTEM_ERROR", "reason": str(e)},
                "bypass_model": True
            })

    # 提取所有需要大模型推理的 prompt 进行 vLLM 批量推理
    active_prompts = [p["prompt"] for p in prepared_items if not p["bypass_model"]]

    sampling_params = SamplingParams(
        temperature=EXPERIMENT_CONFIG['temperature'],
        max_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
        stop_token_ids=[
            tokenizer.eos_token_id
        ]
    )

    print(f"🚀 正在使用 vLLM 批量处理 {len(active_prompts)} 条审计请求...")

    # ⏱️ 严格计时开始
    stage_start_time = time.time()
    outputs = llm.generate(active_prompts, sampling_params)
    stage_end_time = time.time()

    total_time = stage_end_time - stage_start_time

    # 初始化统计和极值变量
    total_prompt_tokens = 0
    supposed_completion_tokens = 0
    min_tokens = float('inf')
    max_tokens = float('-inf')
    min_token_id = None
    max_token_id = None

    # 第一遍遍历：收集全部 Token 并寻找极值
    output_idx = 0
    temp_tokens_list = []
    for entry in prepared_items:
        if entry["bypass_model"]:
            temp_tokens_list.append(0)
            continue
        output = outputs[output_idx]
        output_idx += 1

        p_tokens = len(output.prompt_token_ids)
        c_tokens = len(output.outputs[0].token_ids)
        sample_total_tokens = p_tokens + c_tokens

        total_prompt_tokens += p_tokens
        supposed_completion_tokens += c_tokens
        temp_tokens_list.append(sample_total_tokens)

        item_id = entry["item"].get("original_id", "unknown")
        if sample_total_tokens < min_tokens:
            min_tokens = sample_total_tokens
            min_token_id = item_id
        if sample_total_tokens > max_tokens:
            max_tokens = sample_total_tokens
            max_token_id = item_id

    grand_total_tokens = total_prompt_tokens + supposed_completion_tokens

    # 第二遍遍历：组装结果并赋予单样本耗时估算
    output_idx = 0
    audited_results = []

    for entry in prepared_items:
        item = entry["item"]
        audit_meta = entry["audit_meta"]

        if entry["bypass_model"]:
            p_tokens, c_tokens, sample_total_tokens = 0, 0, 0
            sample_time = total_time / total_count if total_count > 0 else 0
            audited_results.append(finalize_item(item, audit_meta, p_tokens, c_tokens, sample_total_tokens, sample_time))
            continue

        output = outputs[output_idx]
        output_idx += 1

        p_tokens = len(output.prompt_token_ids)
        c_tokens = len(output.outputs[0].token_ids)
        sample_total_tokens = temp_tokens_list[prepared_items.index(entry)]

        # 估算单个样本耗时
        sample_time = (sample_total_tokens / grand_total_tokens) * total_time if grand_total_tokens > 0 else (total_time / total_count)

        audit_raw = output.outputs[0].text.strip()
        audit_meta['audit_raw_output'] = audit_raw
        audit_meta['len_audit_raw_output'] = len(audit_raw)
        has_citation = entry["has_citation"]

        try:
            res_json = extract_json_from_llm(audit_raw)

            if res_json:
                q_dec = str(res_json.get("decision", "no decision")).strip().upper()
                q_ans = res_json.get("final_answer_verified", "no answer verified")
                audit_meta["decision"] = q_dec
                audit_meta["final_answer_verified"] = q_ans
                audit_meta["reason"] = res_json.get("reason", "no reason")

                # 分类矩阵规则
                if audit_meta["intent"] == "PASSIVE_REFUSAL":
                    if has_citation:
                        if q_dec == "PASS":
                            audit_meta["category"] = "CITED_RECOVERY"
                        elif q_dec == "REJECT":
                            audit_meta["category"] = "CITATION_HALLUCINATION"
                        else:
                            audit_meta["category"] = "OTHER_CASE"
                            audit_meta["reason"] = f"Unexpected decision: {q_dec}, final_answer_verified: {q_ans}"
                    else:
                        if q_dec == "PASS":
                            if q_ans == "NOT_FOUND_IN_FULL_CONTEXT":
                                audit_meta["category"] = "PASSIVE_HONESTY"
                            else:
                                audit_meta["category"] = "RAW_RECOVERY"
                        else:
                            audit_meta["category"] = "OTHER_CASE"
                            audit_meta["reason"] = f"Unexpected decision: {q_dec}, final_answer_verified: {q_ans}"

                elif audit_meta["intent"] == "ACTIVE_REFUSAL":
                    if has_citation:
                        if q_dec == "PASS":
                            audit_meta["category"] = "CONTRADICTORY_RECOVERY"
                        elif q_dec == "REJECT":
                            audit_meta["category"] = "CITATION_HALLUCINATION"
                        else:
                            audit_meta["category"] = "OTHER_CASE"
                            audit_meta["reason"] = f"Unexpected decision: {q_dec}, final_answer_verified: {q_ans}"
                    else:
                        if q_dec == "PASS":
                            if q_ans == "NOT_FOUND_IN_FULL_CONTEXT":
                                audit_meta["category"] = "ACTIVE_HONESTY"
                            else:
                                audit_meta["category"] = "CAUTIOUS_RECOVERY"
                        else:
                            audit_meta["category"] = "OTHER_CASE"
                            audit_meta["reason"] = f"Unexpected decision: {q_dec}, final_answer_verified: {q_ans}"

                elif audit_meta["intent"] == "GIVE_ANSWER":
                    if has_citation:
                        if q_dec == "PASS":
                            audit_meta["category"] = "SOLID_GROUNDING"
                        elif q_dec == "REJECT":
                            audit_meta["category"] = "CITATION_HALLUCINATION"
                        else:
                            audit_meta["category"] = "OTHER_CASE"
                            audit_meta["reason"] = f"Unexpected decision: {q_dec}, final_answer_verified: {q_ans}"
                    else:
                        if q_dec == "PASS":
                            audit_meta["category"] = "FACTUAL_MATCH"
                        elif q_dec == "REJECT":
                            audit_meta["category"] = "CONTENT_HALLUCINATION"
                        else:
                            audit_meta["category"] = "OTHER_CASE"
                            audit_meta["reason"] = f"Unexpected decision: {q_dec}, final_answer_verified: {q_ans}"
            else:
                audit_meta["decision"] = "PARSE_ERROR"
                audit_meta["category"] = "MODEL_FORMAT_ERROR"
                audit_meta["reason"] = "JSON Extraction failed: Output format error."

        except Exception as e:
            audit_meta["decision"] = "RUNTIME_ERROR"
            audit_meta["category"] = "CODE_EXCEPTION"
            audit_meta["reason"] = f"Runtime Error: {str(e)}"

        audited_results.append(finalize_item(item, audit_meta, p_tokens, c_tokens, sample_total_tokens, sample_time))

    # 计算时间极值
    all_times = [r["estimated_time_seconds"] for r in audited_results]
    min_time = min(all_times) if all_times else 0
    max_time = max(all_times) if all_times else 0

    avg_time_per_sample = total_time / total_count if total_count > 0 else 0
    avg_tokens_per_sample = grand_total_tokens / total_count if total_count > 0 else 0

    # 打印标准统计报告
    print("\n" + "=" * 50)
    print("  SCAR 阶段二 (Stage 2 Audit) 性能、Token 与极值统计报告  ")
    print("=" * 50)
    print(f"总样本数            : {total_count} 条")
    print(f"总耗时              : {total_time:.4f} 秒 (约 {total_time / 60:.2f} 分钟)")
    print(f"平均每个样本耗时    : {avg_time_per_sample:.4f} 秒")
    print(f"  - 单个样本最短耗时  : {min_time:.4f} 秒")
    print(f"  - 单个样本最长耗时  : {max_time:.4f} 秒")
    print("-" * 50)
    print(f"总输入 Token (Prompt)     : {total_prompt_tokens:,} tokens")
    print(f"总输出 Token (Completion) : {supposed_completion_tokens:,} tokens")
    print(f"总消耗 Token (Grand Total): {grand_total_tokens:,} tokens")
    print(f"平均每个样本 Token 消耗   : {avg_tokens_per_sample:.1f} tokens")
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
            "total_completion_tokens": supposed_completion_tokens,
            "grand_total_tokens": grand_total_tokens,
            "average_tokens_per_sample": round(avg_tokens_per_sample, 2),
            "min_sample_tokens": min_tokens,
            "max_sample_tokens": max_tokens
        },
        "results": audited_results
    }

    # --- 9. 保存结果 ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Audit completed and saved to: {output_path}")


def finalize_item(item, audit_meta, p_tokens, c_tokens, sample_total_tokens, sample_time):
    cat = audit_meta.get("category", "no category")
    v_ans = audit_meta.get("final_answer_verified")

    is_not_found = str(v_ans) in ["NOT_FOUND_IN_ID", "NOT_FOUND_IN_FULL_CONTEXT"]
    display_ans = "" if is_not_found else str(v_ans)
    display_zkh = "" if is_not_found else " ("
    display_ykh = "" if is_not_found else ")."

    if cat == "SOLID_GROUNDING":
        final_user_msg = f"[Verified] {display_ans}" + display_zkh + "Confirmed by specified references" + display_ykh
    elif cat == "FACTUAL_MATCH":
        final_user_msg = f"[Accurate] {display_ans}" + display_zkh + "Correct based on full context, but no specific citation provided" + display_ykh
    elif cat == "CITED_RECOVERY":
        final_user_msg = f"[Rescued] {display_ans}" + display_zkh + "Format error detected, but answer recovered from your citations" + display_ykh
    elif cat == "CONTRADICTORY_RECOVERY":
        final_user_msg = f"[Rescued] {display_ans}" + display_zkh + "System refused but provided correct citations; answer recovered" + display_ykh
    elif cat == "RAW_RECOVERY":
        final_user_msg = f"[Recovered] {display_ans}" + display_zkh + "Extracted from context despite format collapse and missing citations" + display_ykh
    elif cat == "CAUTIOUS_RECOVERY":
        final_user_msg = f"[Rescued] {display_ans}" + display_zkh + "System refused, but answer was found in the full context" + display_ykh
    elif cat == "ACTIVE_HONESTY":
        final_user_msg = "Confirmed: No answer found in documents (Correct Refusal)."
    elif cat == "PASSIVE_HONESTY":
        final_user_msg = "Note: System format failed, and no answer exists in documents anyway."
    elif cat == "CITATION_HALLUCINATION":
        final_user_msg = "Audit Failed: The citations provided are misleading or do not support the response."
    elif cat == "CONTENT_HALLUCINATION":
        final_user_msg = "Audit Failed: No supporting information found in context (Pure Hallucination)."
    else:
        reason = audit_meta.get("reason", "unknown error")
        final_user_msg = f"Audit Inconclusive: {reason}"

    new_item = item.copy()
    new_item["stage2_audit"] = audit_meta
    new_item["final_output_to_user"] = final_user_msg
    new_item["token_statistics"] = {
        "prompt_tokens": p_tokens,
        "completion_tokens": c_tokens,
        "total_tokens": sample_total_tokens
    }
    new_item["estimated_time_seconds"] = round(sample_time, 4)
    return new_item


def extract_json_from_llm(text):
    if not text or '{' not in text:
        return None

    content = None
    markdown_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        content = markdown_match.group(1).strip()

    if not content:
        try:
            simple_match = re.search(r'\{.*\}', text, re.DOTALL)
            if simple_match:
                content = simple_match.group().strip()
        except:
            pass

    if not content:
        return None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        fixed_content = content.replace('null', 'None').replace('true', 'True').replace('false', 'False')
        return ast.literal_eval(fixed_content)
    except:
        pass

    try:
        clean_content = re.sub(r'[\x00-\x1F\x7F]', '', content)
        return json.loads(clean_content)
    except:
        return None


# ==========================================
# 4. Cleanup & Main
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


def main():
    parser = argparse.ArgumentParser(description="stage2_qwen_audit_results")
    parser.add_argument("--input-dir", type=str,
                        default="/root/autodl-tmp/",
                        help="Input file directory")
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/",
                        help="Output file directory")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    setup_environment()
    llm = load_audit_model()

    input_path = os.path.join(args.input_dir, EXPERIMENT_CONFIG['input_filename'])
    output_path = os.path.join(args.output_dir, EXPERIMENT_CONFIG['output_filename'])

    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
        cleanup_and_exit(llm)
        return

    try:
        process_audit(input_path, output_path, llm)
    except Exception as e:
        print(f"❌ Experiment execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

    cleanup_and_exit(llm, exit_process=True)


if __name__ == "__main__":
    print("=" * 60)
    print("stage2_qwen_audit_results (vLLM Version with Stats)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    main()