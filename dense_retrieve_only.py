"""
dense_retrieve_only.py — 纯向量检索脚本（只出 Top-K 结果，不做评估）

功能概述
--------
加载 vector_db_build.py 构建好的 Wikipedia 全量段落 FAISS 索引，
对测试问题集逐条做稠密向量检索（Top-K），回查原文并做句子边界自愈，
最终把结果按题目组织成 JSON 落盘，供下游（挖 hard negative / 训练 /
Reader / 生成等）直接使用。本脚本**不计算任何命中率或评估指标**。

在流水线中的位置
----------------
[vector_db_build.py]                  [★ 本脚本]                         [下游]
passage → BGE-M3 → FAISS index  →  query 编码 → index.search Top-K  →  results.json
     + passages.jsonl（原文元数据）      回查原文 → 边界自愈 → 结构化结果

输入
----
- MODEL_PATH : BGE-M3 模型目录
- INDEX_PATH : wikipedia_full.index（FAISS IndexFlatIP，1024 维，与 jsonl 行号对齐）
- JSONL_PATH : wikipedia_full_passages.jsonl（每行 {id,title,text}，id 即行号-1）
- CSV_PATH   : 待检索问题集，需含列 id / question / source / all_reference

输出
----
OUTPUT_JSON_PATH（vector_retrieve_results.json），每题一条：
{
  "id": "...", "question": "...", "source": "...", "all_reference": [...],
  "top_5": [
    {"passage_id": 123, "title": "...", "content": "...(已自愈)",
     "score": 0.87, "rank": 1}, ...
  ]
}

关键实现说明
------------
1. 索引用 mmap + READ_ONLY 加载：80GB 索引不整体进内存，按需读页，省内存。
2. 启动时用零向量跑一次 index.search 强制预热，避免首条查询异常慢。
3. query 编码后做 faiss.normalize_L2，保证内积 == 余弦相似度。
4. 索引里只有向量没有原文，需按 passage_id 回查 jsonl 取 text/title；
   用 linecache 按行号随机读，避免把 2100 万行整文件读进内存。
5. heal_text_boundaries 做"上下文自愈"：psgs_w100 按固定词数切段，
   常把句子切断——若段落首字符小写则拼前一段末句，若结尾无 .!? 则拼后一段首句。
   仅影响展示/回查文本，不改索引与 jsonl。

注意事项
--------
- TOTAL_ENTRIES 必须与建库时最终条目数严格一致，否则 heal_text_boundaries
  的越界判断会不准（建议改为从 jsonl 行数动态获取或加断言）。
- 若题目量大且结果只给程序读，可把 json.dump 的 indent=2 去掉以减小体积。
- 单卡 cuda:0 推理；批量大/并发高时可改 devices 或做 batch 编码。
"""


import os
import faiss
import numpy as np
import json
import time
import re
import linecache
import pandas as pd
from FlagEmbedding import BGEM3FlagModel

# ====================== 1. 硬件与路径配置 ======================
os.environ["NUMEXPR_MAX_THREADS"] = "84"
os.environ["OMP_NUM_THREADS"] = "80"
os.environ["MKL_NUM_THREADS"] = "80"

MODEL_PATH = "/root/autodl-tmp/SCAR/models/bge-m3"
INDEX_PATH = "/root/autodl-tmp/SCAR/data/wiki/output/wikipedia_full.index"
JSONL_PATH = "/root/autodl-tmp/SCAR/data/wiki/output/wikipedia_full_passages.jsonl"
CSV_PATH = "/root/autodl-tmp/SCAR/data/question/extracted_source_questions.csv"
OUTPUT_JSON_PATH = "/root/autodl-tmp/SCAR/data/output/vector_retrieve_results.json"

TOTAL_ENTRIES = 21015299
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ====================== 2. 核心原子功能 ======================

def get_raw_text(tid):
    """获取原始文本和标题"""
    if tid < 0 or tid >= TOTAL_ENTRIES:
        return "", ""
    line = linecache.getline(JSONL_PATH, tid + 1)
    if not line:
        return "", ""
    try:
        data = json.loads(line)
        return data.get('text', ""), data.get('title', "")
    except:
        return "", ""


def heal_text_boundaries(center_id):
    """上下文自愈补全"""
    text, title = get_raw_text(center_id)
    current_text = text.strip()
    if not current_text:
        return "", title

    # 前向补全
    if current_text.lstrip()[:1].islower(): # 如果文本开头第一个字符是小写
        # 就进行前向补全
        prev_text, _ = get_raw_text(center_id - 1)
        if prev_text:
            sentences = re.split(r'(?<=[.!?])\s+', prev_text.strip())
            if sentences:
                current_text = sentences[-1] + " " + current_text

    # 后向补全
    if current_text.rstrip()[-1:] not in ['.', '!', '?']:
        next_text, _ = get_raw_text(center_id + 1)
        if next_text:
            sentences = re.split(r'(?<=[.!?])\s+', next_text.strip())
            if sentences:
                current_text = current_text + " " + sentences[0]

    return current_text, title


# ====================== 3. 检索引擎 ======================

def run_vector_retrieve(model, index, df, top_k=5):
    """
    执行向量检索并保存结果
    """
    print(f"\nStarting vector retrieval (Top-{top_k})...")

    results = []
    start_time = time.time()

    for idx, row in df.iterrows():
        # 使用CSV文件中原有的id字段
        question_id = str(row['id']) if 'id' in row else str(idx + 1)
        source = row['source']
        query = row['question']

        # 解析参考信息
        try:
            refs = json.loads(row['all_reference'])
            if not isinstance(refs, list):
                refs = [refs]
        except:
            refs = [str(row['all_reference'])]

        # --- 向量检索 ---
        query_vec = model.encode([query], return_dense=True)['dense_vecs'].astype('float32')
        faiss.normalize_L2(query_vec)
        distances, indices = index.search(query_vec, top_k)

        # --- 构建Top-5结果 ---
        top_5_results = []

        for rank, (tid, score) in enumerate(zip(indices[0], distances[0]), 1):
            tid_int = int(tid)

            # 获取并处理文本
            text, title = heal_text_boundaries(tid_int)

            # 构建结果字典
            result_item = {
                "passage_id": tid_int,
                "title": title,
                "content": text,
                "score": float(score),
                "rank": rank
            }
            top_5_results.append(result_item)

        # --- 保存结果 ---
        results.append({
            "id": question_id,
            "question": query,
            "source": source,
            "all_reference": refs,
            "top_5": top_5_results
        })

        # 进度显示
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {idx + 1}/{len(df)} | Time: {elapsed:.1f}s")

    # 保存为JSON文件
    save_as_json(results, start_time)


def save_as_json(results, start_time):
    """保存结果为JSON格式"""
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{'=' * 60}")
        print(f"✅ Retrieval completed!")
        print(f"📊 Total questions: {len(results)}")
        print(f"⏱️  Total time: {duration:.2f} seconds")
        print(f"📁 Results saved to: {OUTPUT_JSON_PATH}")

        # 文件大小
        if os.path.exists(OUTPUT_JSON_PATH):
            file_size = os.path.getsize(OUTPUT_JSON_PATH) / (1024 * 1024)
            print(f"💾 File size: {file_size:.2f} MB")

        print(f"{'=' * 60}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")


# ====================== 4. 启动逻辑 ======================
if __name__ == "__main__":
    # 加载资源
    print("Loading BGE-M3 model and 80GB index...")
    model = BGEM3FlagModel(MODEL_PATH, use_fp16=True, devices=['cuda:0'])
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

    # 强制预热
    _ = index.search(np.zeros((1, 1024), dtype='float32'), 1)
    print("Model and index loaded successfully!")

    # 加载测试数据
    print(f"Loading question data: {CSV_PATH}")
    df_test = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df_test)} questions")

    # 执行检索
    run_vector_retrieve(model, index, df_test, top_k=5)