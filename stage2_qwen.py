import os
import re
import json
import torch
import random
import warnings
import argparse
import gc
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ==========================================
# Experiment Configuration (配置保持不动)
# ==========================================

EXPERIMENT_CONFIG = {
    "audit_model_path": "/root/autodl-tmp/SCAR/models/Qwen2.5-14B-Instruct",
    "random_seed": 1001,
    "input_filename": "stage1_output_stage2_input.json",
    "output_filename": "stage2_input_qwen_output_stage3_input.json",
    "batch_size": 200,
    "inference_batch": 20  # 5090 物理推理批次，完美对齐 200
}


# ==========================================
# 1. Environment Setup (保持不动)
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

    # Set random seed for reproducibility
    random.seed(EXPERIMENT_CONFIG['random_seed'])


# ==========================================
# 2. Audit Model Loading (保持不动)
# ==========================================

def load_audit_model():
    print(f"⚖️ Loading Auditor: {EXPERIMENT_CONFIG['audit_model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        EXPERIMENT_CONFIG['audit_model_path'],
        trust_remote_code=True,
        use_fast=True  # 必须开启，C++ 实现的分词器快得多
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        EXPERIMENT_CONFIG['audit_model_path'],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True  # 加快加载速度
    )
    print("✅ Auditor loaded successfully")
    model.eval()
    return model, tokenizer


# ==========================================
# 3. Main Processing Function (重点修改逻辑)
# ==========================================

def process_audit(input_path, output_path, model, tokenizer):
    print(f"\n{'=' * 60}")
    print(f"Starting Stage 2 Audit: {EXPERIMENT_CONFIG['name']}")
    print(f"Input: {input_path}")
    print(f"{'=' * 60}")

    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    all_results = input_data.get("results", [])
    audited_results = []
    total_count = len(all_results)

    for i, item in enumerate(all_results, 1):
        if i % 100 == 0 or i == 1:
            print(f"🔄 Auditing: {i}/{total_count} ({(i / total_count) * 100:.1f}%)")

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
                "intent": "",  # 稍后判定
                "category": ""
            }

            # --- 4. 逻辑分流与级联拦截 ---

            # 第一步：无论引用好坏，先定意图（基于正文）
            if is_format_collapse:
                audit_meta["intent"] = "PASSIVE_REFUSAL"
            elif is_real_refusal:
                audit_meta["intent"] = "ACTIVE_REFUSAL"
            else:
                audit_meta["intent"] = "GIVE_ANSWER"

            # 第二步：物理层级联拦截 (全是 UNKNOWN 的垃圾引用)
            # 逻辑：只要提供了 ID (num_raw > = 2) 但有效 ID 为 0 (num_clean == 0)
            if num_raw >= 2 and num_clean == 0:
                audit_meta["category"] = "CITATION_HALLUCINATION"
                audit_meta["decision"] = "REJECT"
                audit_meta["reason"] = f"Physical Block: {num_raw} IDs provided but all are UNKNOWN/Invalid."
                # 此处直接结束，跳过后续所有模型调用代码

            else:
                # 第三步：进入模型审计层 (有效引用 或 无引用)
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

                # --- 6. 构建针对性 Audit Prompt (分语义审计版) ---
                has_citation = (num_clean > 0)
                # 统一输出格式，确保下游处理逻辑唯一
                json_format = '{"decision": "PASS/REJECT", "final_answer_verified": "...", "reason": "..."}'

                # 处理 拒答类意图 (无论被动 PASSIVE 还是主动 ACTIVE)
                if audit_meta["intent"] in ["PASSIVE_REFUSAL", "ACTIVE_REFUSAL"]:
                    # 动作定性为：RESCUED (尝试从 Context 中救回答案)
                    if has_citation:
                        # 场景：有引用。逻辑：证据指路但拒答，找不着答案属于引用幻觉。
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
                        # 场景：无引用。逻辑：全量搜索。
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

                # 处理 答题类意图
                elif audit_meta["intent"] == "GIVE_ANSWER":
                    # 动作定性为：VERIFIED (核实模型给出的答案是否属实)
                    if has_citation:
                        # 场景：有引用。逻辑：核实答案是否源自指定的这几篇 References。
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
                        # 场景：无引用。逻辑：核实答案是否在 Full Context 全量库中。
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

                # --- 7. 调用 Qwen 推理与解析 ---
                inputs = tokenizer(audit_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
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
                audit_raw = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                audit_meta['audit_raw_output'] = audit_raw
                audit_meta['len_audit_raw_output'] = len(audit_raw)

                try:
                    res_json = extract_json_from_llm(audit_raw)

                    if res_json:
                        q_dec = str(res_json.get("decision", "no decision")).strip().upper()
                        q_ans = res_json.get("final_answer_verified", "no answer verified")
                        audit_meta["decision"] = q_dec
                        audit_meta["final_answer_verified"] = q_ans
                        audit_meta["reason"] = res_json.get("reason", "no reason")

                        # --- 严格执行用户定义的分类矩阵 (Category Matrix) ---
                        # 1. 拒答类意图 (PASSIVE_REFUSAL / ACTIVE_REFUSAL)
                        if audit_meta["intent"] == "PASSIVE_REFUSAL":
                            if has_citation:
                                # 轨道：指路了但拒答
                                if q_dec == "PASS":
                                    audit_meta["category"] = "CITED_RECOVERY"  # 捞回来了
                                elif q_dec == "REJECT":
                                    audit_meta["category"] = "CITATION_HALLUCINATION"  # 乱指路 (REJECT)
                                else:
                                    audit_meta["category"] = "OTHER_CASE"  # 或者保持默认
                                    audit_meta[
                                        "reason"] = f"Unexpected decision value from LLM: {q_dec},final_answer_verified:{q_ans} "
                            else:
                                # 轨道：没指路且拒答
                                if q_dec == "PASS":
                                    if q_ans == "NOT_FOUND_IN_FULL_CONTEXT":
                                        audit_meta["category"] = "PASSIVE_HONESTY"  # 确认全量无货
                                    else:
                                        audit_meta["category"] = "RAW_RECOVERY"  # 确认捞回答案
                                else:
                                    audit_meta["category"] = "OTHER_CASE"
                                    audit_meta[
                                        "reason"] = f"Unexpected decision value from LLM: {q_dec},final_answer_verified:{q_ans} "


                        elif audit_meta["intent"] == "ACTIVE_REFUSAL":
                            if has_citation:
                                if q_dec == "PASS":
                                    audit_meta["category"] = "CONTRADICTORY_RECOVERY"  # 确认全量无货，诚实
                                elif q_dec == "REJECT":
                                    audit_meta["category"] = "CITATION_HALLUCINATION"  # 全量有货，救回
                                else:
                                    audit_meta["category"] = "OTHER_CASE"  # 或者保持默认
                                    audit_meta[
                                        "reason"] = f"Unexpected decision value from LLM: {q_dec},final_answer_verified:{q_ans} "

                            else:
                                if q_dec == "PASS":
                                    if q_ans == "NOT_FOUND_IN_FULL_CONTEXT":
                                        audit_meta["category"] = "ACTIVE_HONESTY"  # 确认全量无货，诚实
                                    else:
                                        audit_meta["category"] = "CAUTIOUS_RECOVERY"  # 全量有货，救回
                                else:
                                    audit_meta["category"] = "OTHER_CASE"  # 或者保持默认
                                    audit_meta[
                                        "reason"] = f"Unexpected decision value from LLM: {q_dec},final_answer_verified:{q_ans} "

                        # 2. 答题类意图 (GIVE_ANSWER)
                        elif audit_meta["intent"] == "GIVE_ANSWER":
                            if has_citation:
                                # 轨道：给答案且有指路
                                if q_dec == "PASS":
                                    audit_meta["category"] = "SOLID_GROUNDING"
                                elif q_dec == "REJECT":
                                    audit_meta["category"] = "CITATION_HALLUCINATION"  # 证据对不上
                                else:
                                    audit_meta["category"] = "OTHER_CASE"  # 或者保持默认
                                    audit_meta[
                                        "reason"] = f"Unexpected decision value from LLM: {q_dec},final_answer_verified:{q_ans} "
                            else:
                                # 轨道：给答案但没指路
                                if q_dec == "PASS":
                                    audit_meta["category"] = "FACTUAL_MATCH"  # 蒙对了或全量匹配成功
                                elif q_dec == "REJECT":
                                    audit_meta["category"] = "CONTENT_HALLUCINATION"  # 纯瞎编
                                else:
                                    audit_meta["category"] = "OTHER_CASE"  # 或者保持默认
                                    audit_meta[
                                        "reason"] = f"Unexpected decision value from LLM: {q_dec},final_answer_verified:{q_ans} "

                    else:
                        # 如果解析返回 None，手动标记为解析失败
                        audit_meta["decision"] = "PARSE_ERROR"
                        audit_meta["category"] = "MODEL_FORMAT_ERROR"
                        audit_meta["reason"] = "JSON Extraction failed: Output format error."


                except Exception as e:
                    audit_meta["decision"] = "RUNTIME_ERROR"
                    audit_meta["category"] = "CODE_EXCEPTION"
                    audit_meta["reason"] = f"Runtime Error: {str(e)}"

            # --- 8. 仲裁与最终合成 ---
            cat = audit_meta.get("category", "no category")
            v_ans = audit_meta.get("final_answer_verified")

            # 处理显示逻辑：如果是标签，则转换为描述性文字；如果是答案，则保留

            is_not_found = str(v_ans) in ["NOT_FOUND_IN_ID", "NOT_FOUND_IN_FULL_CONTEXT"]
            # 2. 只有当它是真正的“答案”时，display_ans 才拿值
            if is_not_found:
                display_ans = ""  # 既然没找到，就不在括号里重复那句“我找不到”了
            else:
                display_ans = str(v_ans)

            display_zkh = "" if is_not_found else " ("
            display_ykh = "" if is_not_found else ")."

            # if "NOT_FOUND_IN_ID" == str(v_ans):
            #     # 对于“诚实”类，v_ans 实际上是审计确认的结果
            #     display_ans = "I cannot answer this question based on the provided references."
            # elif "NOT_FOUND_IN_FULL_CONTEXT" == str(v_ans):
            #     display_ans = "I cannot answer this question based on the full context."
            # else:
            #     display_ans = str(v_ans)

            # 定义不同标签对应的用户提示语
            if cat == "SOLID_GROUNDING":
                final_user_msg = f"[Verified] {display_ans}" + display_zkh + "Confirmed by specified references" + display_ykh

            elif cat == "FACTUAL_MATCH":
                final_user_msg = f"[Accurate] {display_ans}" + display_zkh + "Correct based on full context, but no specific citation provided" + display_ykh

            elif cat == "CITED_RECOVERY":
                # 针对格式崩塌：强调“捞回”
                final_user_msg = f"[Rescued] {display_ans}" + display_zkh + "Format error detected, but answer recovered from your citations" + display_ykh

            elif cat == "CONTRADICTORY_RECOVERY":
                # 针对主动拒答+有引有答：强调“嘴硬”但“指对了路”
                final_user_msg = f"[Rescued] {display_ans}" + display_zkh + "System refused but provided correct citations; answer recovered" + display_ykh

            elif cat == "RAW_RECOVERY":
                final_user_msg = f"[Recovered] {display_ans}" + display_zkh + "Extracted from context despite format collapse and missing citations" + display_ykh

            elif cat == "CAUTIOUS_RECOVERY":
                # 针对无引用拒答：强调“太怂”
                final_user_msg = f"[Rescued] {display_ans}" + display_zkh + "System refused, but answer was found in the full context" + display_ykh

            elif cat == "ACTIVE_HONESTY":
                final_user_msg = "Confirmed: No answer found in documents (Correct Refusal)."

            elif cat == "PASSIVE_HONESTY":
                final_user_msg = "Note: System format failed, and no answer exists in documents anyway."

            elif cat == "CITATION_HALLUCINATION":
                # 现在包含了：给出错答案+错引用，以及 主动拒答+错引用（噪音）
                final_user_msg = "Audit Failed: The citations provided are misleading or do not support the response."

            elif cat == "CONTENT_HALLUCINATION":
                final_user_msg = "Audit Failed: No supporting information found in context (Pure Hallucination)."

            else:
                reason = audit_meta.get("reason", "unknown error")
                final_user_msg = f"Audit Inconclusive: {reason}"

            new_item = item.copy()
            new_item["stage2_audit"] = audit_meta
            new_item["final_output_to_user"] = final_user_msg
            audited_results.append(new_item)

        except Exception as e:
            # print(f"❌ Error at index {i}: {e}")
            item["stage2_audit"] = {"decision": "SYSTEM_ERROR", "reason": str(e)}
            audited_results.append(item)

        if i % EXPERIMENT_CONFIG['batch_size'] == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    # --- 9. 保存结果 ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": audited_results}, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Audit completed: {output_path}")


import ast


def extract_json_from_llm(text):
    """
    层级回退式 JSON 提取器：
    1. 优先抓取 ```json 块
    2. 其次利用正则抓取最外层花括号 {}
    3. 增加 ast.literal_eval 修复单引号/Python 类型问题
    4. 增加控制字符清洗保底
    """
    if not text or '{' not in text:
        return None

    content = None

    # --- 1. 尝试提取被 Markdown 标签包裹的内容 ---
    # markdown_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    markdown_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        content = markdown_match.group(1).strip()

    # --- 2. 如果没标签，或标签内解析失败，使用你原来的正则逻辑 (作为核心保底) ---
    if not content:
        try:
            # 这里的 \{.*\} 是贪婪匹配，能抓到最外层的花括号
            simple_match = re.search(r'\{.*\}', text, re.DOTALL)
            if simple_match:
                content = simple_match.group().strip()
        except:
            pass

    if not content:
        return None

    # --- 3. 尝试解析逻辑链 ---

    # 第一跳：标准 json.loads
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 第二跳：处理单引号、None、True/False (LLM 经常吐出 Python 格式)
    try:
        # 统一转成 Python 字面量格式，由 ast 处理
        fixed_content = content.replace('null', 'None').replace('true', 'True').replace('false', 'False')
        return ast.literal_eval(fixed_content)
    except:
        pass

    # 第三跳：终极清洗 (处理多余的控制字符、不可见字符)
    try:
        clean_content = re.sub(r'[\x00-\x1F\x7F]', '', content)
        return json.loads(clean_content)
    except:
        return None


# ==========================================
# 4. Cleanup & Main (保持不动)
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


def main():
    global start_time
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description="stage2_qwen_audit_results")
    parser.add_argument("--input-dir", type=str,
                        default="/root/autodl-tmp/SCAR/data/output/stage1",
                        help="Input file directory")
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/SCAR/data/output/stage2_qwen",
                        help="Output file directory")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Maximum number of questions to process")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Set up environment
    setup_environment()

    # 2. Load model
    model, tokenizer = load_audit_model()

    # 3. Build file paths
    input_path = os.path.join(args.input_dir, EXPERIMENT_CONFIG['input_filename'])
    output_path = os.path.join(args.output_dir, EXPERIMENT_CONFIG['output_filename'])

    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
        cleanup_and_exit(model, tokenizer)
        return

    # 4. Execute experiment
    try:
        process_audit(input_path, output_path, model, tokenizer)
    except Exception as e:
        print(f"❌ Experiment execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # 5. Clean up and exit
    cleanup_and_exit(model, tokenizer, exit_process=True)


if __name__ == "__main__":
    print("=" * 60)
    print("stage2_qwen_audit_results")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    main()
