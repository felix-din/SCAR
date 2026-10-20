import os
import re
import json
import torch
import random
import warnings
import argparse
import gc
import ast
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# Experiment Configuration
# ==========================================

EXPERIMENT_CONFIG = {
    "model_path": "/root/autodl-tmp/SCAR/models/Mistral-NeMo-12B-Instruct",
    "random_seed": 1029,
    "input_filename": "stage2_input_qwen_output_stage3_input.json",
    "output_filename": "stage3_output_mn_final_output.json",
    "inference_batch": 32,  # 5090 建议 16-32，根据显存占用调整
    "batch_size": 192,
    "max_new_tokens": 512
}


# ==========================================
# 1. Environment Setup
# ==========================================

def setup_environment():
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore")
    print(f"🚀 Experiment: {EXPERIMENT_CONFIG['name']}")
    print(f"📌 Random Seed: {EXPERIMENT_CONFIG['random_seed']}")
    print(f"⚡ Inference Batch Size: {EXPERIMENT_CONFIG['inference_batch']}")
    random.seed(EXPERIMENT_CONFIG['random_seed'])
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")


# ==========================================
# 2. Model Loading (Batching Optimized)
# ==========================================

def load_alignment_model():
    print(f"⚖️ Loading Aligner: {EXPERIMENT_CONFIG['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        EXPERIMENT_CONFIG['model_path'],
        trust_remote_code=True,
        use_fast=True,  # 必须开启，C++ 实现的分词器快得多
        fix_mistral_regex=True  # ✨ 修复 Mistral 官方正则 Bug，提升理解力
    )

    # 批处理关键：设置左填充，确保生成时的位置向量正确
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. 物理对齐 ID (这是最关键的一步) ---
    # 确保 tokenizer 认为的 pad_token_id 和 model 生成时用的 ID 是同一个物理值
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        EXPERIMENT_CONFIG['model_path'],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 5090 强力加速
        low_cpu_mem_usage=True  # 加快加载速度
    )
    model.eval()
    print("✅ Model loaded with Flash-Attention 2")
    return model, tokenizer


# ==========================================
# 3. JSON Extractor (级联回退逻辑)
# ==========================================
def extract_json_from_llm(text):
    """
    高效提取 LLM 输出中最后一个（通常是最完整的）JSON 块
    """
    if not text or '{' not in text:
        return None

    # 1. 优先匹配 Markdown 块 (```json ... ```)
    md_blocks = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if md_blocks:
        for content in md_blocks:
            result = try_parse_all_methods(content.strip())
            if result is not None:
                return result

    # 2. 如果没找到 MD 块，尝试匹配所有 {} 结构
    # 贪婪匹配会抓取第一个 { 到最后一个 } 之间的所有内容
    json_candidates = re.findall(r'\{.*?\}', text, re.DOTALL)

    if json_candidates:
        for content in json_candidates:
            result = try_parse_all_methods(content.strip())
            if result is not None:
                return result

    return None


def try_parse_all_methods(content):
    """
    多级智能回退解析引擎 (满血加固版)
    """
    if not content:
        return None

    # --- 第一级：标准解析 ---
    try:
        return json.loads(content)
    except:
        pass

    # --- 第二级：Python 风格兼容 (处理 True/False/None/单引号) ---
    try:
        # 使用 \b 确保只替换独立的关键字
        temp_content = re.sub(r'\btrue\b', 'True', content)
        temp_content = re.sub(r'\bfalse\b', 'False', temp_content)
        temp_content = re.sub(r'\bnull\b', 'None', temp_content)
        return ast.literal_eval(temp_content)
    except:
        pass

    # --- 第三级：极端字符清洗与暴力修复 ---
    # 先定义一个基础的清洗版本，作为后续修复的基准
    try:
        clean_content = re.sub(r'[\x00-\x1F\x7F]', '', content)
        # 将真实的换行符替换为转义字符 \n，这是 JSON 规范要求的
        clean_content = clean_content.replace('\n', '\\n').replace('\r', '')

        # 尝试清洗后的解析
        try:
            return json.loads(clean_content)
        except:
            pass

        # --- 暴力修复：处理类似 "Manchester United" 导致的嵌套引号 ---
        # 1. 先给所有非转义的双引号加上斜杠
        fixed = re.sub(r'(?<!\\)"', r'\"', clean_content)

        # 2. 用正则精准还原 JSON 结构的引号 (使用 \s* 覆盖所有空格/换行情况)
        fixed = re.sub(r'(?<=\{)\s*\\"', '"', fixed)  # 还原 {"
        fixed = re.sub(r'\\"\s*(?=\})', '"', fixed)  # 还原 "}
        fixed = re.sub(r'\\"\s*:\s*\\"', '": "', fixed)  # 还原 "key": "value"
        fixed = re.sub(r'\\"\s*,\s*\\"', '", "', fixed)  # 还原 "value", "key"

        # 3. 修正可能被误伤的边界
        fixed = fixed.strip()
        if not fixed.startswith('{'): fixed = '{' + fixed
        if not fixed.endswith('}'): fixed = fixed + '}'

        try:
            return json.loads(fixed)
        except:
            # 最后的倔强：尝试用 Python 引擎解析暴力修复后的结果
            try:
                return ast.literal_eval(fixed)
            except:
                pass  # 这里不写任何逻辑，让它自然走到最下面的 return None

    except:
        # 到这里说明连 clean_content 或 fixed 的生成都失败了
        return None


# ==========================================
# 4. Main Processing Function
# ==========================================

def process_alignment_batch(input_path, output_path, model, tokenizer):
    print(f"\n{'=' * 60}")
    print(f"Starting Batch Alignment Check")
    print(f"Input: {input_path}")
    print(f"{'=' * 60}")

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    all_samples = input_data.get("results")
    total_count = len(all_samples)

    # --- 1. 定义分类集合 ---
    # 需要精准审计（仅限引用的 DOC）
    PRECISE_AUDIT_TAGS = {"SOLID_GROUNDING", "CITED_RECOVERY", "CONTRADICTORY_RECOVERY"}
    # 需要全量审计（1-5 全量扫描）
    GLOBAL_AUDIT_TAGS = {"RAW_RECOVERY", "CAUTIOUS_RECOVERY", "FACTUAL_MATCH"}
    # 自动结案（不进 Stage 3）
    SKIP_AUDIT_CATEGORIES = {"ACTIVE_HONESTY", "PASSIVE_HONESTY", "CITATION_HALLUCINATION", "CONTENT_HALLUCINATION"}

    MODEL_FORMAT_ERROR_TAG = "MODEL_FORMAT_ERROR"

    # 2. 初始化一个等长的列表，用来装最终结果
    final_checked_results = [None] * total_count
    # 3. 筛选出需要审计的【索引】，而不是拷贝对象
    audit_indices = []

    for idx, item in enumerate(all_samples):
        audit_s2 = item.get("stage2_audit", {})
        category = audit_s2.get("category", "UNKNOWN_CATEGORY")

        # 1. 自动结案：如果在跳过名单中，直接记录结果
        if category in SKIP_AUDIT_CATEGORIES:
            # 自动分类逻辑
            judgment = "SKIPPED"
            new_item = item.copy()
            new_item["stage3_audit"] = {
                "judgment": judgment,
                "reason": f"Skipped: Category '{category}' finalized in Stage 2.",
                "raw_output": "SKIPPED_BY_LOGIC"
            }
            final_checked_results[idx] = new_item
        # 2. 审计准入：除跳过名单外的所有情况（包含已知和未知分类）全部进入审计索引
        else:
            audit_indices.append(idx)

    inf_batch = EXPERIMENT_CONFIG['inference_batch']
    cleanup_threshold = EXPERIMENT_CONFIG['batch_size']

    # json_format = '{"judgment": "ALIGNED/MISALIGNED", "reason": "Concise analysis of the QA alignment and factual basis."}'
    #
    # SYSTEM_PROMPT = """
    #     You are a Senior QA Alignment Auditor. Your task is to perform a final semantic check: Does the 'Model Answer' provide a valid and factually grounded response to the 'Question'?
    #
    #     ### CORE AUDIT MISSION:
    #     1. **QA Logic**: Does the Answer directly satisfy the intent of the Question?
    #     2. **Factual Grounding**: Are the specific facts (names, dates, numbers) in the Answer consistent with the provided 'Source Documents' and 'Evidence'?
    #     3. **No Overclaim**: Did the model 'hallucinate' extra information that wasn't asked for or isn't in the references?
    #
    #     ### 💡 AUDIT NOTE ON REFERENCES:
    #     - If 'Stage 2 Evidence' is 'No specific evidence extracted' or empty, you MUST perform a full semantic scan of all provided 'Source Documents'.
    #     - Do not penalize the answer solely because pre-extracted evidence is missing.
    #
    #     ### JUDGMENT STANDARDS:
    #     - **ALIGNED**: The Answer is a high-quality, relevant, and factually accurate response to the Question.
    #     - **MISALIGNED**: The Answer is irrelevant, contains factual errors compared to the source, or claims things not supported by the context.
    #
    #     ### CRITICAL OUTPUT INSTRUCTION:
    #     1. You must output the result in the following JSON format:{json_format}
    #     2. You MUST start your response with this JSON block.
    #     3. Do not provide any conversational filler or analysis BEFORE the JSON.
    #     4. Detailed step-by-step analysis should only be provided AFTER the JSON block.
    # """

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

    # --- 3. 批处理推理 (仅针对 to_audit_samples) ---
    pbar = tqdm(range(0, len(audit_indices), inf_batch), desc="Auditing Batches", unit="batch")
    for i in pbar:
        current_batch_indices = audit_indices[i: i + inf_batch]
        prompts = []

        # 构造本批次的 Prompts
        for idx in current_batch_indices:
            item = all_samples[idx]
            stage2_audit = item.get("stage2_audit", {})
            category = stage2_audit.get("category", "UNKNOWN_CATEGORY")
            passages = item.get("retrieved_passages", [])
            q = item.get("question", "N/A")

            if category == MODEL_FORMAT_ERROR_TAG:
                a = stage2_audit.get("final_answer", "no answer")
                s2_evidence = "No specific evidence extracted."
            else:
                a = stage2_audit.get("final_answer_verified", "no answer")
                s2_evidence = stage2_audit.get("evidence", "no evidence")


            # --- 动态 Context 构造逻辑 ---
            target_context = ""

            # 提取并解析 ID
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
                    target_context = "\n\n".join([f"[[DOC {idx + 1}]] {passages[idx]}" for idx in unique_idxs])
                else:
                    # 引用解析失败则回退全量
                    target_context = "\n\n".join([f"[[DOC {j + 1}]] {p}" for j, p in enumerate(passages)])

            else:
                target_context = "\n\n".join([f"[[DOC {j + 1}]] {p}" for j, p in enumerate(passages)])


            # 3. 构造 User Content (原文 + 证据 + 问题 + 答案)
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
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted)

        # 编码并填充 (Padding)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # 批处理推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
                do_sample=False,
                use_cache=True,  # 开启 KV Cache
                num_beams=1,  # 确保是 greedy search，最快
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # 减少 CPU-GPU 之间的同步等待
                return_dict_in_generate=False,
                output_scores=False
            )

        input_len = inputs.input_ids.shape[1]
        for b_idx, original_idx in enumerate(current_batch_indices):
            raw_gen = tokenizer.decode(outputs[b_idx][input_len:], skip_special_tokens=True).strip()
            res_json = extract_json_from_llm(raw_gen)
            # --- ✨ 关键改进：归一化处理 ---
            if res_json:
                raw_judgment = str(res_json.get("judgment", "")).strip().upper().replace('.', '')
                # 模糊匹配：只要包含 MISALIGNED 就是不一致
                final_judgment = raw_judgment if raw_judgment else "EMPTY_JUDGMENT"  # 非标结果默认降级为不一致
            else:
                final_judgment = "PARSE_ERROR"

            processed_item = all_samples[original_idx].copy()
            processed_item["stage3_audit"] = {
                "judgment": final_judgment,
                "reason": res_json.get("reason", "No reason provided") if res_json else raw_gen,
                "raw_output": raw_gen
            }
            final_checked_results[original_idx] = processed_item

        current_processed = i + len(current_batch_indices)
        if current_processed % cleanup_threshold == 0 or current_processed == len(audit_indices):
            torch.cuda.empty_cache()
            gc.collect()
            pbar.set_postfix({"status": "GC_Done", "last_idx": current_processed})

    final_checked_results = [r for r in final_checked_results if r is not None]

    misaligned_count = len([
        r for r in final_checked_results
        if r["stage3_audit"]["judgment"] == "MISALIGNED"
    ])

    parse_errors_count = len([r for r in final_checked_results if r["stage3_audit"]["judgment"] == "PARSE_ERROR"])

    # 汇总并保存
    final_output = {
        "summary": {
            "total": total_count,  # 总数 (3785)
            "audited": len(audit_indices),  # 真正进模型的数量
            "parse_errors": parse_errors_count,
            "misaligned_in_audited": misaligned_count,  # 审计发现的冲突数
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        "results": final_checked_results  # 顺序与输入文件完全一致
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Finished! Misaligned: {misaligned_count}, Parse Errors: {parse_errors_count}")


# ==========================================
# 5. Cleanup & Exit (保持不动)
# ==========================================

def cleanup_and_exit(model, tokenizer, exit_process=True):
    """
    Thoroughly clean GPU memory and optionally exit process

    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        exit_process: If True, exit process after cleanup
    """
    print("\n🧹 Performing deep memory cleanup...")

    try:
        # 1. 将模型移到CPU（关键步骤！）
        if model is not None and hasattr(model, 'to'):
            try:
                print("  Moving model to CPU...")
                model.to('cpu')
                model = None
            except Exception as e:
                print(f"  ⚠️ Warning moving model to CPU: {e}")

        # 2. 删除模型引用
        if model is not None:
            print("  Deleting model references...")
            del model

        # 3. 删除tokenizer引用
        if tokenizer is not None:
            print("  Deleting tokenizer references...")
            del tokenizer

        # 4. 强制垃圾回收
        print("  Running garbage collection...")
        for i in range(3):
            gc.collect()

        # 5. 清理CUDA缓存
        if torch.cuda.is_available():
            print("  Clearing CUDA cache...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # 显示清理后的内存使用
            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(
                f"  📊 GPU memory after cleanup: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

        print("✅ Memory cleanup completed")

    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. 可选：退出进程（使用sys.exit而不是os._exit）
    if exit_process:
        print("🔄 Exiting process for complete state reset...")
        import sys
        sys.exit(0)


# ==========================================
# Main Entry
# ==========================================

def main():
    setup_environment()
    model, tokenizer = load_alignment_model()

    # 路径配置
    INPUT_PATH = "/root/autodl-tmp/SCAR/data/output/stage2_qwen/" + EXPERIMENT_CONFIG['input_filename']
    OUTPUT_PATH = "/root/autodl-tmp/SCAR/data/output/stage3_mn/" + EXPERIMENT_CONFIG['output_filename']

    try:
        process_alignment_batch(INPUT_PATH, OUTPUT_PATH, model, tokenizer)
    except Exception as e:
        print(f"❌ Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

    cleanup_and_exit(model, tokenizer, exit_process=True)


if __name__ == "__main__":
    main()
