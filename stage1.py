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
# 这里的 thefuzz 需要安装 (pip install thefuzz)，如果不想装可以只用碎片匹配
from thefuzz import fuzz

# Global variable for start time
start_time = None

# ==========================================
# Experiment Configuration
# ==========================================

EXPERIMENT_CONFIG = {
    "model_path": "/root/autodl-tmp/SCAR/models/Meta-Llama-3.1-8B-Instruct",
    "random_seed": 1001,  # Unique random seed
    "input_filename": "pure_dense.json",  # Only process pure retrieval file
    "output_filename": "stage1_output_stage2_input.json",
    "max_new_tokens": 1024,
    "batch_size": 10,
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

    # Set random seed for reproducibility
    random.seed(EXPERIMENT_CONFIG['random_seed'])


# ==========================================
# 2. Model Loading
# ==========================================

def load_model_and_tokenizer():
    """Load model and tokenizer"""
    print(f"🔧 Loading model: {EXPERIMENT_CONFIG['model_path']}")

    tokenizer = AutoTokenizer.from_pretrained(EXPERIMENT_CONFIG['model_path'])

    # Load model with bfloat16 precision to GPU
    model = AutoModelForCausalLM.from_pretrained(
        EXPERIMENT_CONFIG['model_path'],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("✅ Model loaded successfully")
    model.eval()
    return model, tokenizer


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
        # Extract content from top_5 as retrieval passages
        passages = []
        for passage_item in item.get("top_5", []):
            if "content" in passage_item and passage_item["content"]:
                passages.append(passage_item["content"].strip())

        results.append({
            "original_id": item.get("id", f"unknown_{idx}"),
            "original_index": idx,  # Save original order
            "question": item.get("question", ""),
            "passages": passages,
            "shuffled_index": idx,  # Will be updated after shuffling
        })

    print(f"📊 Read {len(results)} questions")

    # Shuffle order using fixed seed
    print(f"🎲 Shuffling with seed {EXPERIMENT_CONFIG['random_seed']}...")
    random.shuffle(results)

    # Update shuffled positions
    for i, item in enumerate(results):
        item["shuffled_index"] = i

    # Show shuffling effect
    print(f"First 5 original IDs after shuffling: {[r['original_id'] for r in results[:5]]}")
    print(f"First 5 shuffled positions: {[r['shuffled_index'] for r in results[:5]]}")

    return results


# ==========================================
# 4. Generation Function (No tag version)
# ==========================================

def generate_final_answer(question, passages_list, tokenizer, model):
    """Generate final answer (no tag version)"""

    try:
        # Merge passages
        # 1. 结构化文档输入：明确编号 1-5，这是防止模型在长文本中迷失的关键
        formatted_docs = []
        for i, p in enumerate(passages_list[:5]):
            formatted_docs.append(f"[[SOURCE DOCUMENT {i + 1}]]\n{p.strip()}")
        P = "\n\n".join(formatted_docs)

        # 2. 修改 System Prompt：翻转顺序 + 强制列表
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
            f"Quote(s): ["  # 关键点：将 Answer: 改为 Quote(s): 引导模型先列证据
        )

        messages = [
            {"role": "system", "content": full_system_content},
            {"role": "user", "content": user_message}
        ]

        # Apply chat template
        f_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ).strip()

        inputs = tokenizer(f_prompt, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=EXPERIMENT_CONFIG['max_new_tokens'],
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]
            )

        raw_gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # 1. 清理标签
        clean_content = raw_gen.replace("Quote(s):", "").strip()

        if "Answer:" in clean_content:
            parts = clean_content.split("Answer:")
            quote_part = parts[0].strip()
            answer_part = parts[1].strip()

            doc_ids_str, extracted_list = unified_matching_process(quote_part, passages_list)
            full_response = f"Quote(s): {extracted_list}\nDoc_IDs: {doc_ids_str}\nAnswer: {answer_part}"

        else:
            # --- 修改点 2：在兜底逻辑里也加入 Doc_IDs 占位符，保持格式对齐 ---
            doc_ids_str, extracted_list = unified_matching_process(clean_content, passages_list)

            # 【关键区分点】：在 Answer 里加一个微小的固定前缀
            # 下游脚本如果是按文本匹配 Answer 内容，这里就分开了
            fallback_answer = "(No answer tag found) I cannot answer based on the provided documents."
            full_response = f"Quote(s): {extracted_list}\nDoc_IDs: {doc_ids_str}\nAnswer: {fallback_answer}"

        return full_response

    except Exception as e:
        print(f"⚠️ Generation error: {str(e)}")
        return f"ERROR: {str(e)}"


def unified_matching_process(quote_text_raw, passages_list):
    """
    统一处理：清洗引用 -> 匹配文档 -> 返回 ID 列表字符串和提取后的列表
    """
    # 1. 提取引用 (支持单双引号)
    raw_quotes = re.findall(r'[\"\'](.*?)[\"\'](?=\s*[,\]]|\s*$)', quote_text_raw.replace('\n', ' '))
    extracted_list = [q.strip(' ,[]\\\'\"').strip() for q in raw_quotes if q.strip()]

    if not extracted_list:
        return "[]", []

    matched_ids = []
    for q_text in extracted_list:
        found_id = "UNKNOWN"
        clean_q = q_text.strip()

        # 核心回捞匹配逻辑 (级联匹配)
        for idx, passage in enumerate(passages_list[:5]):
            # A. 精确匹配
            if clean_q in passage:
                found_id = f"doc_{idx + 1}"
                break

            # B. 规范化匹配 (处理空格/换行)
            q_norm = " ".join(clean_q.split())
            p_norm = " ".join(passage.split())
            if q_norm in p_norm:
                found_id = f"doc_{idx + 1}"
                break

            # C. 碎片化匹配 (处理省略号)
            if "..." in clean_q:
                frags = [f.strip() for f in clean_q.split("...") if len(f.strip()) > 5]
                if frags and all(f in passage for f in frags):
                    found_id = f"doc_{idx + 1}"
                    break

            # D. 模糊匹配 (处理标点差异)
            if fuzz and fuzz.partial_ratio(clean_q, passage) > 90:
                found_id = f"doc_{idx + 1}"
                break

        # 加上前缀以区分分支
        matched_ids.append(found_id)

    doc_ids_str = "[" + ", ".join([f'"{d}"' for d in matched_ids]) + "]"
    return doc_ids_str, extracted_list


# ==========================================
# 5. Main Processing Function
# ==========================================

def process_experiment(input_path, output_path, model, tokenizer):
    """Execute experiment processing with structured parsing"""
    print(f"\n{'=' * 60}")
    print(f"Processing: {EXPERIMENT_CONFIG['name']}")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"{'=' * 60}")

    # Read and shuffle data
    all_data = read_and_shuffle_retrieval_results(input_path)

    # Process each question
    results = []
    total_count = len(all_data)

    for i, data_item in enumerate(all_data, 1):
        # Progress display
        if i % 100 == 0 or i == 1 or i == total_count:
            progress_percent = i / total_count * 100
            print(f"🔄 Progress: {i}/{total_count} ({progress_percent:.1f}%)")
            print(f"   Current ID: {data_item['original_id']} (Original position: {data_item['original_index']})")

            # Monitor memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024 ** 3
                print(f"   GPU Memory: {memory_used:.2f} GB")

        try:
            # 1. 生成原始回复 (包含 Answer: 和 Quote: )
            raw_response = generate_final_answer(
                data_item['question'],
                data_item['passages'],
                tokenizer,
                model
            )

            # 2. 结构化解析逻辑
            # 我们预设模型输出格式为 "Answer: [内容] \n Quote: [内容]"
            ans_content = ""
            ids_content = ""
            quo_content = ""

            parts_q = raw_response.split("Doc_IDs:")
            quo_content = parts_q[0].replace("Quote(s):", "").strip()

            # 第二次拆分：剥离 Doc_IDs 和 Answer
            parts_a = parts_q[1].split("Answer:")
            ids_content = parts_a[0].strip()  # 这里拿到的就是 ["doc_1", "doc_2"]
            ans_content = parts_a[1].strip()

            # if "Doc_IDs:" in raw_response and "Answer:" in raw_response:
            #     # 第一次拆分：剥离 Quote 部分
            #     parts_q = raw_response.split("Doc_IDs:")
            #     quo_content = parts_q[0].replace("Quote(s):", "").strip()
            #
            #     # 第二次拆分：剥离 Doc_IDs 和 Answer
            #     parts_a = parts_q[1].split("Answer:")
            #     ids_content = parts_a[0].strip()  # 这里拿到的就是 ["doc_1", "doc_2"]
            #     ans_content = parts_a[1].strip()
            # elif "Answer:" in raw_response:
            #     # 兜底逻辑：如果模型没吐 Doc_IDs 标签 (旧格式)
            #     parts = raw_response.split("Answer:")
            #     quo_content = parts[0].replace("Quote(s):", "").strip()
            #     ans_content = parts[1].strip()
            #     ids_content = "[]"
            # else:
            #     # 完全格式错误
            #     quo_content = raw_response.replace("Quote(s):", "").strip()
            #     ans_content = "Format Error: Answer label missing."
            #     ids_content = "[]"

            # 3. 构建结果字典 (新增 final_quote 和 raw_generation)
            result = {
                "original_id": data_item['original_id'],
                "original_index": data_item['original_index'],
                "shuffled_index": data_item['shuffled_index'],
                "question": data_item['question'],
                "retrieved_passages": data_item['passages'],
                "final_quote(s)": quo_content,  # 存储对应的证据原文
                "doc_id(s)": ids_content,  # 【新增】存储匹配到的文档编号列表字符串
                "final_answer": ans_content,  # 存储纯净答案
                "raw_generation": raw_response,  # 保留原始生成文本备查
                "processing_order": i
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
                "processing_order": i
            })

        # Regular cache cleanup
        if i % EXPERIMENT_CONFIG['batch_size'] == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # --- 后续的 Build final output 和保存逻辑保持不变 ---
    final_output = {
        "experiment_info": {
            "experiment_name": EXPERIMENT_CONFIG['name'],
            "model": "llama-3.1-8B-instruct",
            "generation_type": "forward_generation",
            "prompt_type": "notag",
            "retrieval_type": "pure",
            "random_seed": EXPERIMENT_CONFIG['random_seed'],
            "input_filename": os.path.basename(input_path),
            "output_filename": os.path.basename(output_path),
            "total_questions": total_count,
            "processed_questions": len(results),
            "start_time": start_time,
            "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "shuffle_method": "random.shuffle with fixed seed"
        },
        "data_statistics": {
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "success_rate": f"{len([r for r in results if 'error' not in r]) / len(results) * 100:.1f}%" if results else "0%"
        },
        "results": results
    }

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"📊 Statistics:")
    print(f"  Total questions: {len(results)}")
    print(f"  Successful: {final_output['data_statistics']['successful']}")
    print(f"  Failed: {final_output['data_statistics']['failed']}")
    print(f"  Success rate: {final_output['data_statistics']['success_rate']}")

    return final_output


# ==========================================
# 6. Cleanup Function
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
# 7. Main Program
# ==========================================

def main():
    global start_time
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description="Forward Generation - No Tag Pure Retrieval")
    parser.add_argument("--input-dir", type=str,
                        default="/root/autodl-tmp/SCAR/data/output/dense_retreive",
                        help="Input file directory")
    parser.add_argument("--output-dir", type=str,
                        default="/root/autodl-tmp/SCAR/data/output/stage1",
                        help="Output file directory")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Maximum number of questions to process")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Set up environment
    setup_environment()

    # 2. Load model
    model, tokenizer = load_model_and_tokenizer()

    # 3. Build file paths
    input_path = os.path.join(args.input_dir, EXPERIMENT_CONFIG['input_filename'])
    output_path = os.path.join(args.output_dir, EXPERIMENT_CONFIG['output_filename'])

    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
        cleanup_and_exit(model, tokenizer)
        return

    # 4. Execute experiment
    try:
        process_experiment(input_path, output_path, model, tokenizer)
    except Exception as e:
        print(f"❌ Experiment execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

    # 5. Clean up and exit
    cleanup_and_exit(model, tokenizer, exit_process=True)


# ==========================================
# Script Entry Point
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Forward Generation Experiment - No Tag Pure Retrieval Version")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    main()
