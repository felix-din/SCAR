"""
hybrid_retrieve.py — 混合检索系统（Dense + BM25 + BGE-Reranker）三阶段流水线

========================================================================
功能概述
========================================================================
在纯 dense 检索（test_dense_retrieve_only.py）的基础上，引入 BM25 稀疏检索
与 BGE-Reranker 交叉编码重排，通过 alpha 加权融合，构建 Dense+BM25 混合检索，
并用 reranker 对候选做最终重排，输出 Top-5 结果。支持多组 alpha 一次性跑完，
便于后续做 alpha 敏感性实验。

========================================================================
三阶段设计（用命令行参数切换，互斥）
========================================================================
阶段一 --build-index
    从 wikipedia_full_passages.jsonl 生成 Pyserini 输入（用 heal_text_boundaries
    修复断句）→ 调 pyserini.index.lucene 构建 Lucene BM25 索引 → 验证。
    文档量大（2100 万），耗时约 3-5 小时。生成文件较大，属于一次性离线操作。

阶段二 --retrieve-only [--save-candidates]
    对每个问题同时跑 Dense（FAISS）与 BM25（Pyserini）检索，各自召回 25 条，
    归一化分数后按 alpha 加权融合，取 Top-K 落盘到 normal/ 目录。
    若加 --save-candidates，还会把两路原始候选池（含原文、原始分、归一化分）
    按问题存到 candidates/question_<id>/，供阶段三复用，避免阶段三重跑检索。

阶段三 --rerank-only
    读取阶段二保存的候选池，用 BGE-Reranker-v2-m3（transformers 加载）对
    Dense 与 BM25 候选分别重排，再按 alpha 加权融合重排分数，取 Top-5 落盘到
    reranked/ 目录。此阶段不碰 80GB FAISS 索引，内存压力小。

========================================================================
为什么拆三阶段（设计动机）
========================================================================
FAISS 索引 80GB + BGE-M3 + Reranker + Pyserini 同时驻留内存极易 OOM。
拆开后：阶段二只加载检索所需资源；阶段三只加载 reranker + 读候选池；
两者互不叠加，且候选池可复用，调 alpha / 换 reranker 时无需重跑检索。

========================================================================
输入
========================================================================
- MODEL_PATH         : BGE-M3（Dense 编码）
- RERANKER_PATH      : bge-reranker-v2-m3（交叉编码重排）
- INDEX_PATH         : wikipedia_full.index（FAISS，与 jsonl 行号对齐）
- JSONL_PATH         : wikipedia_full_passages.jsonl（原文元数据）
- CSV_PATH           : 问题集，含 id/question/source/all_reference
- PYSERINI_INDEX_DIR : BM25 Lucene 索引目录（阶段一产物）

========================================================================
输出
========================================================================
OUTPUT_BASE_DIR/
├── normal/                      # 阶段二：混合检索（未重排）结果
│   ├── pure_bm25.json           # alpha=0（纯 BM25）
│   ├── pure_dense.json          # alpha=1（纯 Dense）
│   └── hybrid_alpha_*.json      # 各 alpha 混合结果
├── candidates/                  # 阶段二（--save-candidates）：候选池
│   └── question_<id>/
│       ├── dense_candidates.json
│       ├── bm25_candidates.json
│       └── metadata.json
└── reranked/                    # 阶段三：重排后结果
    ├── rerank_pure_bm25.json
    ├── rerank_pure_dense.json
    └── rerank_alpha_*.json

========================================================================
分数归一化策略
========================================================================
- Dense：BGE-M3 余弦相似度 ∈ [-1,1]，先线性映射到 [0,2]，再做 Min-Max 归一化到 [0,1]。
- BM25 ：原始分 ∈ [0,∞)，直接 Min-Max 归一化到 [0,1]。
- 统一用 normalize_scores_batch()，保证两路分数可加权融合；
  所有分数相同的退化情况返回全 0，单分数返回 0.5。
- 混合分 hybrid = alpha * dense_norm + (1-alpha) * bm25_norm（alpha=1 纯 dense，=0 纯 BM25）。

========================================================================
关键实现说明
========================================================================
1. 索引一律用 mmap + READ_ONLY 加载，避免 80GB 全量驻留内存。
2. 候选池落盘时同时保留 raw_score 与 norm_score，便于阶段三追溯。
3. 内容统一走 get_consistent_content()（内部调 heal_text_boundaries），
   保证 Dense/BM25/Rerank 三处看到的文本完全一致。
4. 阶段三重排：分别对 dense 池与 bm25 池各取前 10 做 rerank，再取并集，
   对每个 alpha 用 rerank 分加权融合排序取 Top-5；同时保留 Stage2 原始分数
   字段（original_dense/bm25/hybrid_score）便于对比分析。
5. Reranker 用 transformers 直接加载（非 FlagReranker），单值 logits 走 sigmoid，
   双值走 softmax 取正类，统一截断到 [0,1]。
6. 阶段三开始时重置 OMP/MKL/TOKENIZERS 线程为 1，避免与 CUDA 多线程冲突。

========================================================================
命令行用法
========================================================================
python test_hybrid_retrieve.py --build-index
python test_hybrid_retrieve.py --retrieve-only --save-candidates
python test_hybrid_retrieve.py --retrieve-only --save-candidates --start-idx 0 --end-idx 500
python test_hybrid_retrieve.py --rerank-only
python test_hybrid_retrieve.py --rerank-only --start-idx 500 --end-idx 1000
python test_hybrid_retrieve.py --build-index --force-build

========================================================================
注意事项
========================================================================
- TOTAL_ENTRIES 必须与建库最终条目数一致，否则 heal_text_boundaries 越界判断不准。
- 阶段二不加 --save-candidates 时脚本会警告并询问，因为阶段三无法运行。
- 阶段一转换 jsonl → Pyserini 格式时会预加载 linecache，内存占用较高。
- 修改 alpha 默认值 [0,0.3,0.5,0.7,1] 时注意输出文件名按 alpha 生成。
- 依赖：faiss、FlagEmbedding、pyserini、transformers、torch、pandas。
"""

import os
import faiss
import numpy as np
import json
import time
import re
import linecache
import pandas as pd
import subprocess
import argparse
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
from pyserini.search.lucene import LuceneSearcher

# ====================== 1. 配置参数 ======================
# 环境配置：优化多线程性能
os.environ["NUMEXPR_MAX_THREADS"] = "84"
os.environ["OMP_NUM_THREADS"] = "80"
os.environ["MKL_NUM_THREADS"] = "80"

# 路径配置：所有数据、模型、索引的路径
MODEL_PATH = "/root/autodl-tmp/SCAR/models/bge-m3"
RERANKER_PATH = "/root/autodl-tmp/SCAR/models/bge-reranker-v2-m3"
INDEX_PATH = "/root/autodl-tmp/SCAR/data/wiki/output/wikipedia_full.index"
JSONL_PATH = "/root/autodl-tmp/SCAR/data/wiki/output/wikipedia_full_passages.jsonl"
CSV_PATH = "/root/autodl-tmp/SCAR/data/question/extracted_source_questions.csv"
PYSERINI_INDEX_DIR = "/root/autodl-tmp/SCAR/data/wiki/output/pyserini_index"
OUTPUT_BASE_DIR = "/root/autodl-tmp/SCAR/data/output/hybrid_retrieve"

TOTAL_ENTRIES = 21015299  # 总文档数
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一张GPU

# 检索参数
DENSE_CANDIDATES = 25  # Dense检索召回数量
BM25_CANDIDATES = 25  # BM25检索召回数量
RERANK_CANDIDATES = 10  # 重排候选数量
FINAL_TOP_K = 5  # 最终输出top-K结果


# ====================== 2. 命令行参数解析 ======================
def parse_arguments():
    """解析命令行参数，支持三阶段分离模式"""
    parser = argparse.ArgumentParser(
        description='混合检索系统 (Dense + BM25) + BGE-Reranker重排 - 三阶段模式',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 三阶段模式参数（互斥）
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument('--build-index', action='store_true',
                             help='阶段一: 仅构建Pyserini索引并退出（需2-4小时）')
    stage_group.add_argument('--retrieve-only', action='store_true',
                             help='阶段二: 仅检索模式 - 执行检索并保存候选池和结果')
    stage_group.add_argument('--rerank-only', action='store_true',
                             help='阶段三: 仅重排模式 - 加载候选池进行重排')

    # 候选池管理参数
    parser.add_argument('--save-candidates', action='store_true',
                        help='保存完整候选池（阶段二必需）')
    parser.add_argument('--candidates-dir', type=str, default=None,
                        help='候选池目录路径（用于阶段三）')

    # 检索与重排参数
    parser.add_argument('--alpha', type=float, nargs='+',
                        default=[0, 0.3, 0.5, 0.7, 1],
                        help='混合权重参数（0=纯BM25，1=纯Dense）')
    parser.add_argument('--top-k', type=int, default=20,
                        help='混合检索保留的候选数量（默认：20）')
    parser.add_argument('--rerank-top-n', type=int, default=10,
                        help='重排候选数量（默认：10）')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_BASE_DIR,
                        help='输出目录')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='进度显示间隔')

    # 索引构建参数
    parser.add_argument('--force-build', action='store_true',
                        help='如果索引不存在则强制构建，不询问')

    # 增量处理参数
    parser.add_argument('--start-idx', type=int, default=0,
                        help='起始处理索引（用于断点续跑）')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='结束处理索引（用于分批处理）')

    return parser.parse_args()


# ====================== 3. Pyserini索引管理 ======================
# ========== 修改点：修复检查函数 ==========
def check_pyserini_index():
    """检查Pyserini索引状态，返回是否存在、是否完整、大小等信息"""
    if not os.path.exists(PYSERINI_INDEX_DIR):
        return {'exists': False, 'complete': False, 'size_gb': 0}

    # ========== 修正：Pyserini索引的关键文件 ==========
    # segments_N 是必须的（N是数字，如segments_1）
    segments_files = [f for f in os.listdir(PYSERINI_INDEX_DIR)
                      if f.startswith('segments_')]

    # 至少需要一个si文件（段信息文件）
    si_files = [f for f in os.listdir(PYSERINI_INDEX_DIR)
                if f.endswith('.si')]

    # 至少需要一个fnm文件（字段信息文件）
    fnm_files = [f for f in os.listdir(PYSERINI_INDEX_DIR)
                 if f.endswith('.fnm')]

    has_segments = len(segments_files) > 0
    has_si_files = len(si_files) > 0
    has_fnm_files = len(fnm_files) > 0

    print(f" Pyserini index check:")
    print(f"   Segments file: {'✓' if has_segments else '✗'} {segments_files}")
    print(f"   SI files: {'✓' if has_si_files else '✗'} {si_files[:3]}{'...' if len(si_files) > 3 else ''}")
    print(f"   FNM files: {'✓' if has_fnm_files else '✗'} {fnm_files[:3]}{'...' if len(fnm_files) > 3 else ''}")

    complete = has_segments and has_si_files and has_fnm_files

    if complete:
        try:
            # 计算索引总大小
            total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk(PYSERINI_INDEX_DIR)
                             for filename in filenames)
            return {
                'exists': True,
                'complete': True,
                'size_gb': total_size / (1024 ** 3),
                'segments_count': len(segments_files),
                'total_files': len([f for f in os.listdir(PYSERINI_INDEX_DIR)
                                    if os.path.isfile(os.path.join(PYSERINI_INDEX_DIR, f))])
            }
        except Exception as e:
            print(f"   Warning calculating size: {e}")
            return {'exists': True, 'complete': True, 'size_gb': 0}
    else:
        return {'exists': True, 'complete': False, 'size_gb': 0}


def build_pyserini_index():
    """构建Pyserini索引（阶段一），使用现有的heal_text_boundaries函数"""
    print(f"�� Building Pyserini index from {JSONL_PATH}...")
    print(f"   This may take 3-5 hours for {TOTAL_ENTRIES:,} documents")
    print(f"   Index will be saved to: {PYSERINI_INDEX_DIR}")

    start_time = time.time()

    # 1. 创建输入目录
    input_dir = os.path.join(os.path.dirname(JSONL_PATH), "pyserini_input")
    os.makedirs(input_dir, exist_ok=True)

    # 2. 转换数据格式 - 使用现有的 heal_text_boundaries 函数
    print(f"�� Step 1: Converting data to Pyserini format...")
    print(f"   Using existing heal_text_boundaries() for context repair")

    converted_file = os.path.join(input_dir, "wikipedia_converted.jsonl")

    # 首先，预加载一些数据到 linecache 以便 heal_text_boundaries 能工作
    print("   Preloading data for linecache...")
    linecache.updatecache(JSONL_PATH)

    converted_count = 0
    start_convert = time.time()

    with open(JSONL_PATH, 'r', encoding='utf-8') as f_in, \
            open(converted_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in):
            try:
                # 解析原始数据
                data = json.loads(line.strip())
                doc_id_int = data.get('id', line_num)
                original_title = data.get('title', '').strip()
                original_text = data.get('text', '').strip()

                # 关键：使用现有的 heal_text_boundaries 函数修复文本
                healed_text, healed_title = heal_text_boundaries(doc_id_int)

                # 如果修复失败，使用原始文本
                if not healed_text:
                    healed_text = original_text
                if not healed_title:
                    healed_title = original_title

                # 修复后的文本
                contents = f"{healed_text}".strip()

                # 写入Pyserini格式
                pyserini_doc = {
                    "id": str(doc_id_int),  # Pyserini需要字符串ID
                    # 可选：保存原始字段用于调试
                    "_original_title": original_title,
                    "contents": contents,
                }

                f_out.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")

                converted_count += 1

                # 进度显示
                if (line_num + 1) % 1000000 == 0:
                    elapsed = time.time() - start_convert
                    progress = (line_num + 1) / TOTAL_ENTRIES * 100
                    speed = (line_num + 1) / elapsed
                    print(f"   Converted {line_num + 1:,} docs ({progress:.1f}%) | "
                          f"Speed: {speed:.0f} docs/sec")

            except json.JSONDecodeError:
                print(f"⚠️  JSON decode error at line {line_num}, skipping")
                continue
            except Exception as e:
                print(f"⚠️  Error at line {line_num}: {e}")
                continue

    print(f"✅ Step 1 complete: Converted {converted_count:,} documents")
    print(f"   Conversion time: {time.time() - start_convert:.1f}s")

    # 检查转换后的文件
    if os.path.exists(converted_file):
        size_mb = os.path.getsize(converted_file) / (1024 * 1024)
        print(f"   Converted file size: {size_mb:.1f} MB")

    # 3. 构建索引
    print(f"\n�� Step 2: Building Pyserini index...")
    os.makedirs(PYSERINI_INDEX_DIR, exist_ok=True)

    cmd = [
        'python', '-m', 'pyserini.index.lucene',
        '--input', input_dir,  # 包含转换后的文件
        '--collection', 'JsonCollection',
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--index', PYSERINI_INDEX_DIR,
        '--threads', '80',
        '--storePositions', '--storeDocvectors', '--storeRaw'
        # 注意：不再需要 --fields 参数！
    ]

    print(f"   Command: {' '.join(cmd)}")
    print(f"   Starting index construction...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            total_time = time.time() - start_time
            print(f"✅ Pyserini index built successfully!")
            print(f"   Index directory: {PYSERINI_INDEX_DIR}")
            print(f"   Total time: {total_time:.1f}s ({total_time / 3600:.1f} hours)")
            print(f"   Speed: {converted_count / total_time:.1f} docs/sec")

            # 验证索引
            print(f"   Validating index...")
            try:
                searcher = LuceneSearcher(PYSERINI_INDEX_DIR)
                searcher.set_bm25()

                # 测试搜索
                test_queries = ["the", "science", "history", "computer", "test"]
                for query in test_queries:
                    test_hits = searcher.search(query, k=5)
                    if len(test_hits) > 0:
                        print(f"   Query '{query}': ✓ Found {len(test_hits)} hits")
                        # 显示第一个结果
                        doc = searcher.doc(test_hits[0].docid)
                        try:
                            doc_data = json.loads(doc.raw())
                            print(f"      First result ID: {doc_data.get('id')}")
                            print(f"      Preview: {doc_data.get('contents', '')[:100]}...")
                        except:
                            pass
                        break
                else:
                    print(f"⚠️  No hits found for any test query")

                searcher.close()
                print(f"   Index validation: PASS")

            except Exception as e:
                print(f"   Index validation: FAIL - {e}")

            return True
        else:
            print(f"❌ Index construction failed!")
            print(f"   Error code: {result.returncode}")
            if result.stdout:
                print(f"   Output (last 500 chars):\n{result.stdout[-500:]}")
            if result.stderr:
                print(f"   Error message (last 500 chars):\n{result.stderr[-500:]}")
            return False

    except Exception as e:
        print(f"❌ Exception during index construction: {e}")
        import traceback
        traceback.print_exc()
        return False



def load_pyserini_searcher():
    """加载Pyserini搜索器，用于BM25检索"""
    print(f"📂 Loading Pyserini searcher from {PYSERINI_INDEX_DIR}")
    start_time = time.time()

    try:
        searcher = LuceneSearcher(PYSERINI_INDEX_DIR)
        searcher.set_bm25()

        load_time = time.time() - start_time
        print(f"✅ Pyserini searcher loaded in {load_time:.1f}s")

        # 快速测试搜索功能
        test_hits = searcher.search("test", k=1)
        if test_hits:
            print(f"   Test search successful: {len(test_hits)} hits")

        return searcher
    except Exception as e:
        print(f"❌ Failed to load Pyserini searcher: {e}")
        raise


# ====================== 4. 核心文本处理函数 ======================
def get_raw_text(tid):
    """根据文档ID获取原始文本和标题"""
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
    """上下文自愈补全：如果文本开头小写或结尾无标点，补全上下文"""
    text, title = get_raw_text(center_id)
    current_text = text.strip()
    if not current_text:
        return "", title

    # 向前补全：如果开头是小写字母，说明是半句话
    if current_text.lstrip()[:1].islower():
        prev_text, _ = get_raw_text(center_id - 1)
        if prev_text:
            sentences = re.split(r'(?<=[.!?])\s+', prev_text.strip())
            if sentences:
                current_text = sentences[-1] + " " + current_text

    # 向后补全：如果结尾没有标点，说明句子不完整
    if current_text.rstrip()[-1:] not in ['.', '!', '?']:
        next_text, _ = get_raw_text(center_id + 1)
        if next_text:
            sentences = re.split(r'(?<=[.!?])\s+', next_text.strip())
            if sentences:
                current_text = current_text + " " + sentences[0]

    return current_text, title


def get_consistent_content(tid_int, content_cache):
    """统一内容获取接口，确保Dense和BM25使用相同的自愈后文本"""
    if tid_int not in content_cache:
        text, title = heal_text_boundaries(tid_int)
        content_cache[tid_int] = {'text': text, 'title': title}
    return content_cache[tid_int]


# ====================== 5. 分数归一化工具 ======================
# def normalize_bm25_scores(bm25_scores):
#     """BM25分数归一化到[0,1]，防除零处理"""
#     if len(bm25_scores) == 0:
#         return np.array([])
#
#     bm25_min = np.min(bm25_scores)
#     bm25_max = np.max(bm25_scores)
#
#     # 处理所有分数相同的情况（无区分度）
#     if bm25_max - bm25_min < 1e-9:
#         return np.zeros_like(bm25_scores)
#
#     normalized = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
#     return np.clip(normalized, 0.0, 1.0)
#
#
# def normalize_dense_score(score):
#     """Dense分数（余弦相似度）归一化到[0,1]"""
#     # BGE-M3返回的是余弦相似度，范围[-1, 1]
#     # 线性映射到[0, 1]：normalized = (cos_sim + 1) / 2
#     return max(0.0, min(1.0, (float(score) + 1) / 2))

# ========== 修改点1：新增统一归一化函数 ==========
def normalize_scores_batch(scores):
    """统一归一化函数 - Min-Max归一化到[0,1]，处理所有类型分数"""
    if len(scores) == 0:
        return np.array([])

    # 转换为numpy数组
    scores_array = np.array(scores, dtype=np.float32)

    # 处理单个分数的情况
    if scores_array.size == 1:
        return np.array([0.5])  # 单个分数给中间值

    s_min = np.min(scores_array)
    s_max = np.max(scores_array)

    # 处理所有分数相同的情况
    if s_max - s_min < 1e-9:
        return np.zeros_like(scores_array)

    normalized = (scores_array - s_min) / (s_max - s_min)
    return np.clip(normalized, 0.0, 1.0)


# ====================== 6. 检索函数 ======================
def dense_retrieve(model, faiss_index, query, top_k=DENSE_CANDIDATES):
    """Dense检索：使用BGE-M3编码查询，在FAISS索引中搜索"""
    query_vec = model.encode([query], return_dense=True)['dense_vecs'].astype('float32')
    faiss.normalize_L2(query_vec)  # L2归一化，余弦相似度检索
    scores, indices = faiss_index.search(query_vec, top_k)
    return scores[0], indices[0]  # 返回分数和文档ID


def bm25_retrieve_pyserini(searcher, query, top_k=BM25_CANDIDATES):
    """BM25检索：使用Pyserini进行关键词检索"""
    hits = searcher.search(query, k=top_k)
    bm25_ids = []
    bm25_scores = []

    for hit in hits:
        doc = searcher.doc(hit.docid)
        try:
            raw_data = json.loads(doc.raw())
            # 优先使用文档中的id字段，否则使用docid
            passage_id = int(raw_data.get('id', hit.docid))
            bm25_ids.append(passage_id)
            bm25_scores.append(hit.score)  # 原始BM25分数
        except Exception as e:
            print(f"⚠️  Failed to parse document {hit.docid}: {e}")
            continue

    return np.array(bm25_scores), np.array(bm25_ids)


# ====================== 7. 候选池保存与加载函数 ======================
def save_candidate_pools(question_id, query, dense_results, bm25_results, content_cache, output_dir, source=None,
                         refs=None):
    """保存Dense和BM25的原始候选池（阶段二的核心功能）"""
    candidates_dir = os.path.join(output_dir, "candidates", f"question_{question_id}")
    os.makedirs(candidates_dir, exist_ok=True)

    dense_scores, dense_ids = dense_results
    bm25_scores, bm25_ids = bm25_results

    # ========== 修改点2：使用统一归一化函数处理Dense分数 ==========
    # 1. 保存Dense候选池
    dense_candidates = []

    # 批量归一化Dense分数（余弦相似度范围[-1,1]，先映射到[0,2]再归一化）
    # 首先将余弦相似度从[-1,1]线性映射到[0,2]
    dense_scores_mapped = [(float(score) + 1.0) for score in dense_scores]  # 映射到[0,2]
    dense_norm_scores = normalize_scores_batch(dense_scores_mapped)  # 再进行Min-Max归一化到[0,1]

    for rank, (raw_score, doc_id, norm_score) in enumerate(zip(dense_scores, dense_ids, dense_norm_scores), 1):
        content = get_consistent_content(int(doc_id), content_cache)
        dense_candidates.append({
            "passage_id": int(doc_id),
            "title": content['title'],
            "content": content['text'],
            "dense_raw_score": float(raw_score),  # 原始余弦相似度[-1,1]
            "dense_norm_score": float(norm_score),  # 统一Min-Max归一化后的分数[0,1]
            "dense_rank": rank
        })

    with open(os.path.join(candidates_dir, "dense_candidates.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "question_id": question_id,
            "query": query,
            "candidates": dense_candidates
        }, f, ensure_ascii=False, indent=2)

    # ========== 修改点3：使用统一归一化函数处理BM25分数 ==========
    # 2. 保存BM25候选池
    bm25_candidates = []

    # BM25原始分数范围通常为[0, ∞)，使用统一归一化
    bm25_norm_scores = normalize_scores_batch(bm25_scores)

    for rank, (raw_score, norm_score, doc_id) in enumerate(zip(bm25_scores, bm25_norm_scores, bm25_ids), 1):
        content = get_consistent_content(int(doc_id), content_cache)
        bm25_candidates.append({
            "passage_id": int(doc_id),
            "title": content['title'],
            "content": content['text'],
            "bm25_raw_score": float(raw_score),  # 原始BM25分数
            "bm25_norm_score": float(norm_score),  # 统一Min-Max归一化后的分数[0,1]
            "bm25_rank": rank
        })

    with open(os.path.join(candidates_dir, "bm25_candidates.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "question_id": question_id,
            "query": query,
            "candidates": bm25_candidates
        }, f, ensure_ascii=False, indent=2)

    # 3. 保存元数据（新增source和all_reference字段）
    dense_ids_set = set(int(doc_id) for doc_id in dense_ids)
    bm25_ids_set = set(int(doc_id) for doc_id in bm25_ids)

    metadata = {
        "question_id": question_id,
        "query": query,
        "source": source,  # 新增：问题来源
        "all_reference": refs if refs else [],  # 新增：参考文档列表
        "total_dense_candidates": len(dense_ids),
        "total_bm25_candidates": len(bm25_ids),
        "overlap_ids": list(dense_ids_set & bm25_ids_set),
        "unique_dense_ids": list(dense_ids_set - bm25_ids_set),
        "unique_bm25_ids": list(bm25_ids_set - dense_ids_set),
        "save_time": time.time(),
        "normalization_method": "min_max"  # 新增：记录归一化方法
    }

    with open(os.path.join(candidates_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return candidates_dir

def load_candidate_pools(question_id, candidates_dir):
    """加载指定问题的候选池（阶段三的核心功能）"""
    question_dir = os.path.join(candidates_dir, f"question_{question_id}")

    if not os.path.exists(question_dir):
        raise FileNotFoundError(f"Candidate pool not found for question {question_id}")

    # 加载Dense候选池
    with open(os.path.join(question_dir, "dense_candidates.json"), 'r', encoding='utf-8') as f:
        dense_data = json.load(f)

    # 加载BM25候选池
    with open(os.path.join(question_dir, "bm25_candidates.json"), 'r', encoding='utf-8') as f:
        bm25_data = json.load(f)

    # 加载元数据
    with open(os.path.join(question_dir, "metadata.json"), 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return dense_data, bm25_data, metadata


# ====================== 8. BGE-Reranker重排函数 ======================
def bge_rerank_candidates(query, candidates, reranker, top_n=RERANK_CANDIDATES, batch_size=32):
    """使用BGE-Reranker对候选进行重排"""
    if not candidates or len(candidates) == 0:
        return []

    # 限制重排数量
    candidates_to_rerank = candidates[:top_n]

    # 分批处理以避免内存问题
    reranked_candidates = []

    for i in range(0, len(candidates_to_rerank), batch_size):
        batch = candidates_to_rerank[i:i + batch_size]

        # 准备query-passage对
        pairs = [(query, cand['content']) for cand in batch]

        try:
            # 计算重排分数
            scores = reranker.compute_score(pairs, max_length=512)

            # 更新候选分数
            for cand, score in zip(batch, scores):
                cand['rerank_score'] = float(score)
                reranked_candidates.append(cand)

        except Exception as e:
            print(f"⚠️  Reranking batch failed: {e}, using original ranking")
            # 失败时保留原始候选
            for cand in batch:
                cand['rerank_score'] = cand.get('hybrid_score', 0.0)
                reranked_candidates.append(cand)

    # 按重排分数排序
    reranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

    # 只返回top-5
    return reranked_candidates[:FINAL_TOP_K]


# ====================== 9. 阶段二：仅检索函数 ======================
def run_retrieve_only(args):
    """阶段二：仅执行检索，保存候选池和normal结果"""
    print("=" * 60)
    print("STAGE 2: RETRIEVE ONLY (Saving Candidate Pools & Normal Results)")
    print("=" * 60)

    try:
        # 加载模型和索引
        print("\n Loading FAISS index...")
        faiss_index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
        print(f"   ✅ FAISS index loaded (dimension: {faiss_index.d})")

        print("\n烙 Loading BGE-M3 model...")
        model = BGEM3FlagModel(MODEL_PATH, use_fp16=True, devices=['cuda:0'])
        print("   ✅ BGE-M3 model loaded")

        print("\n Loading Pyserini BM25 index...")
        pyserini_searcher = load_pyserini_searcher()

        print("\n Loading questions...")
        df_test = pd.read_csv(CSV_PATH)

        # 确定处理范围
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx is not None else len(df_test)
        df_processed = df_test.iloc[start_idx:end_idx]

        print(f"   ✅ Loaded {len(df_processed)} questions (indices {start_idx}-{end_idx - 1})")

        # 创建输出目录
        normal_dir = os.path.join(args.output_dir, "normal")
        candidates_base_dir = os.path.join(args.output_dir, "candidates")
        os.makedirs(normal_dir, exist_ok=True)
        if args.save_candidates:
            os.makedirs(candidates_base_dir, exist_ok=True)

        print(f"\n Starting retrieve-only mode...")
        print(f"   Dense candidates: {DENSE_CANDIDATES}")
        print(f"   BM25 candidates: {BM25_CANDIDATES}")
        print(f"   Alphas: {args.alpha}")
        print(f"   Output directory: {args.output_dir}")
        print(f"   Saving candidate pools: {args.save_candidates}")

        # 初始化结果存储
        all_normal_results = {f"alpha_{alpha:.1f}": [] for alpha in args.alpha}

        start_time = time.time()
        total_questions = len(df_processed)

        for idx, row in enumerate(df_processed.itertuples(), start=start_idx):
            # 生成问题ID：优先使用CSV中的id，否则使用索引
            if hasattr(row, 'id') and row.id:
                question_id = str(row.id)
            else:
                question_id = f"{idx:04d}"  # 0000, 0001, ...

            source = row.source
            query = row.question

            try:
                refs = json.loads(row.all_reference)
                if not isinstance(refs, list):
                    refs = [refs]
            except:
                refs = [str(row.all_reference)]

            # 执行检索
            dense_scores, dense_ids = dense_retrieve(model, faiss_index, query)
            bm25_scores, bm25_ids = bm25_retrieve_pyserini(pyserini_searcher, query)

            # 创建内容缓存
            content_cache = {}

            # 保存候选池
            if args.save_candidates:
                save_candidate_pools(question_id, query,
                                     (dense_scores, dense_ids),
                                     (bm25_scores, bm25_ids),
                                     content_cache, args.output_dir,
                                     source=source, refs=refs)

            # ========== 修改点4：修改混合分数计算部分 ==========
            # 合并候选池用于计算混合分数
            candidate_pool = {}

            # 添加Dense结果（使用与save_candidate_pools相同的归一化方法）
            # 先将余弦相似度从[-1,1]映射到[0,2]，再进行Min-Max归一化
            dense_scores_mapped = [(float(score) + 1.0) for score in dense_scores]
            dense_norm_scores_batch = normalize_scores_batch(dense_scores_mapped)

            for tid, norm_score in zip(dense_ids, dense_norm_scores_batch):
                tid_int = int(tid)
                candidate_pool[tid_int] = {
                    'dense_score': float(norm_score),  # 统一Min-Max归一化分数[0,1]
                    'bm25_score': 0.0,
                    'content': get_consistent_content(tid_int, content_cache),
                    'dense_raw_score': float(dense_scores[np.where(dense_ids == tid)[0][0]])  # 保存原始分数
                }

            # 添加BM25结果（使用相同的归一化方法）
            bm25_norm_scores_batch = normalize_scores_batch(bm25_scores)

            for tid, norm_score in zip(bm25_ids, bm25_norm_scores_batch):
                tid_int = int(tid)
                bm25_norm = float(norm_score)

                if tid_int in candidate_pool:
                    candidate_pool[tid_int]['bm25_score'] = bm25_norm
                    candidate_pool[tid_int]['bm25_raw_score'] = float(
                        bm25_scores[np.where(bm25_ids == tid)[0][0]])  # 保存原始分数
                else:
                    candidate_pool[tid_int] = {
                        'dense_score': 0.0,
                        'bm25_score': bm25_norm,
                        'content': get_consistent_content(tid_int, content_cache),
                        'bm25_raw_score': float(bm25_scores[np.where(bm25_ids == tid)[0][0]])  # 保存原始分数
                    }
            # ========== 修改点4结束 ==========

            # 对每个alpha计算混合分数并保存结果
            for alpha in args.alpha:
                weight_name = f"alpha_{alpha:.1f}"

                # 计算混合分数
                hybrid_items = []
                for tid, scores in candidate_pool.items():
                    hybrid_score = alpha * scores['dense_score'] + (1 - alpha) * scores['bm25_score']
                    hybrid_items.append((tid, hybrid_score, scores['dense_score'], scores['bm25_score'],
                                         scores.get('dense_raw_score', 0.0), scores.get('bm25_raw_score', 0.0)))

                # 排序取top-k
                hybrid_items.sort(key=lambda x: x[1], reverse=True)
                top_items = hybrid_items[:args.top_k]

                # 准备normal结果
                normal_candidates = []
                for rank, (tid, hybrid_score, dense_score, bm25_score, dense_raw, bm25_raw) in enumerate(
                        top_items[:FINAL_TOP_K], 1):
                    content = candidate_pool[tid]['content']
                    normal_candidates.append({
                        "passage_id": tid,
                        "title": content['title'],
                        "content": content['text'],
                        "hybrid_score": float(hybrid_score),
                        "dense_score": float(dense_score),  # Min-Max归一化的分数
                        "bm25_score": float(bm25_score),  # Min-Max归一化的分数
                        "dense_raw_score": float(dense_raw),  # 原始余弦相似度
                        "bm25_raw_score": float(bm25_raw),  # 原始BM25分数
                        "rank": rank,
                        "alpha": alpha
                    })

                # 添加到结果列表
                all_normal_results[weight_name].append({
                    "id": question_id,
                    "question": query,
                    "source": source,
                    "all_reference": refs,
                    "top_5": normal_candidates
                })

            # 进度显示
            if (idx + 1) % args.batch_size == 0 or (idx + 1 - start_idx) == total_questions:
                elapsed = time.time() - start_time
                progress = (idx + 1 - start_idx) / total_questions * 100
                print(f"[{idx + 1:4d}/{start_idx + total_questions}] {progress:5.1f}% | Elapsed: {elapsed:.0f}s")

        # 保存normal结果
        saved_files = []
        for weight_name, results in all_normal_results.items():
            alpha_str = weight_name.replace('alpha_', '')
            filename = f"hybrid_alpha_{alpha_str}.json"
            if alpha_str == "0.0":
                filename = "pure_bm25.json"
            elif alpha_str == "1.0":
                filename = "pure_dense.json"

            filepath = os.path.join(normal_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            saved_files.append((alpha_str, filepath, len(results)))

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"✅ STAGE 2 COMPLETED!")
        print(f"{'=' * 60}")
        print(f" Total questions processed: {total_questions}")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f" Avg time per question: {total_time / total_questions:.2f}s")

        print(f"\n Output files:")
        for alpha_str, filepath, count in saved_files:
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                if alpha_str == "0.0":
                    name = "Pure BM25"
                elif alpha_str == "1.0":
                    name = "Pure Dense"
                else:
                    name = f"α={alpha_str}"
                print(f"  {name}: {filepath} ({file_size:.1f} MB, {count} questions)")

        if args.save_candidates:
            # 统计候选池文件
            candidate_dirs = [d for d in os.listdir(candidates_base_dir)
                              if d.startswith("question_")]
            print(f"  Candidate pools: {candidates_base_dir}/ ({len(candidate_dirs)} pools)")

        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error in retrieve-only mode: {e}")
        import traceback
        traceback.print_exc()

# ====================== 10. 阶段三：仅重排函数 ======================
def run_rerank_only(args):
    """阶段三：仅执行重排，复用候选池"""

    # 重置线程设置避免冲突
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("=" * 60)
    print("STAGE 3: RERANK ONLY (Using Saved Candidate Pools)")
    print("=" * 60)

    # 确定候选池目录
    if args.candidates_dir is None:
        args.candidates_dir = os.path.join(args.output_dir, "candidates")

    if not os.path.exists(args.candidates_dir):
        print(f"❌ Candidate pools directory not found: {args.candidates_dir}")
        print("   Please run stage 2 first: python hybrid_retrieve.py --retrieve-only --save-candidates")
        exit(1)

    # ========== 修改点1：使用transformers加载BGE-Reranker模型 ==========
    print("\n�� Loading BGE-Reranker-v2-m3 via transformers...")

    # 导入必要的transformers组件
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # 清理GPU显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 加载tokenizer和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_PATH)
        model = model.to('cuda:0')
        model.eval()  # 设置为评估模式
        print("   ✅ BGE-Reranker loaded via transformers")

        # 测试模型输出格式
        print("   Testing model output format...")
        test_inputs = tokenizer("test query", "test document", return_tensors="pt", max_length=512, truncation=True)
        test_inputs = {k: v.to('cuda:0') for k, v in test_inputs.items()}
        with torch.no_grad():
            test_output = model(**test_inputs)
        print(f"   Test output shape: {test_output.logits.shape}")

    except Exception as e:
        print(f"❌ Failed to load BGE-Reranker: {e}")
        exit(1)
    # ========== 修改点1结束 ==========

    # 创建输出目录
    reranked_dir = os.path.join(args.output_dir, "reranked")
    os.makedirs(reranked_dir, exist_ok=True)

    # 获取所有问题目录
    question_dirs = [d for d in os.listdir(args.candidates_dir)
                     if d.startswith("question_") and os.path.isdir(os.path.join(args.candidates_dir, d))]

    # 按问题ID排序
    question_dirs.sort(key=lambda x: x.replace("question_", ""))

    # 确定处理范围
    if args.start_idx > 0 or args.end_idx is not None:
        # 按列表位置过滤，而不是按ID
        question_dirs = question_dirs[args.start_idx:args.end_idx]

    print(f"\n�� Found {len(question_dirs)} candidate pools")
    print(f"�� Alphas to process: {args.alpha}")
    print(f"�� Reranking top-{args.rerank_top_n} candidates for each alpha")
    print(f"�� Batch size: {args.batch_size}")

    # 初始化结果存储
    all_reranked_results = {f"alpha_{alpha:.1f}": [] for alpha in args.alpha}

    # ========== 修改点2：定义新的重排函数（增大batch_size） ==========
    def simple_rerank_candidates(query, candidates, model, tokenizer, top_n=10, batch_size=32):  # 从8增大到32
        """使用transformers模型进行重排 - 返回所有重排候选"""
        if not candidates or len(candidates) == 0:
            return []

        # 重排所有候选（最多25个）
        candidates_to_rerank = candidates[:25]  # 最多处理25个

        reranked_candidates = []

        # 分批处理
        for i in range(0, len(candidates_to_rerank), batch_size):
            batch = candidates_to_rerank[i:i + batch_size]

            # 准备query-passage对
            texts_a = [query] * len(batch)
            texts_b = [cand['content'] for cand in batch]

            try:
                # 编码输入
                inputs = tokenizer(
                    texts_a, texts_b,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # 移到GPU
                inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

                # 计算分数
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # BGE-Reranker-v2-m3输出单值logits，使用sigmoid转换为概率
                    if logits.shape[-1] == 1:
                        batch_scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy().tolist()
                    elif logits.shape[-1] == 2:
                        batch_scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy().tolist()
                    else:
                        batch_scores = torch.sigmoid(logits[:, 0]).cpu().numpy().tolist()

                # 确保分数在合理范围内
                batch_scores = [max(0.0, min(1.0, float(score))) for score in batch_scores]

                # 更新候选分数
                for cand, score in zip(batch, batch_scores):
                    cand['rerank_score'] = float(score)
                    reranked_candidates.append(cand)

            except Exception as e:
                print(f"⚠️  Reranking batch failed: {e}, using default score")
                # 失败时保留原始候选
                for cand in batch:
                    cand['rerank_score'] = 0.0  # 使用默认分数
                    reranked_candidates.append(cand)

        # 按重排分数排序
        reranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        # 返回指定数量的候选
        return reranked_candidates[:top_n]

    # ========== 修改点2结束 ==========

    start_time = time.time()

    for idx, question_dir in enumerate(question_dirs):
        question_path = os.path.join(args.candidates_dir, question_dir)

        try:
            # 提取问题ID
            question_id = question_dir.replace("question_", "")

            # 加载候选池
            dense_data, bm25_data, metadata = load_candidate_pools(question_id, args.candidates_dir)

            query = metadata['query']

            # ========== 修改点3：获取原始分数映射 ==========
            # 构建文档ID到原始分数的映射
            doc_original_scores = {}

            # 从Dense候选池获取原始分数
            for cand in dense_data['candidates']:
                doc_id = cand['passage_id']
                if doc_id not in doc_original_scores:
                    doc_original_scores[doc_id] = {'dense': 0.0, 'bm25': 0.0}
                doc_original_scores[doc_id]['dense'] = cand.get('dense_norm_score', 0.0)  # 归一化Dense分数

            # 从BM25候选池获取原始分数
            for cand in bm25_data['candidates']:
                doc_id = cand['passage_id']
                if doc_id not in doc_original_scores:
                    doc_original_scores[doc_id] = {'dense': 0.0, 'bm25': 0.0}
                doc_original_scores[doc_id]['bm25'] = cand.get('bm25_norm_score', 0.0)  # 归一化BM25分数
            # ========== 修改点3结束 ==========

            # ========== 修改点4：分别重排Dense和BM25候选池 ==========
            # 1. 重排Dense候选池（最多25个）并取前10个
            dense_candidates = dense_data['candidates']
            dense_reranked = simple_rerank_candidates(
                query,
                dense_candidates,
                model,
                tokenizer,
                top_n=10  # 取前10个
            )

            # 2. 重排BM25候选池（最多25个）并取前10个
            bm25_candidates = bm25_data['candidates']
            bm25_reranked = simple_rerank_candidates(
                query,
                bm25_candidates,
                model,
                tokenizer,
                top_n=10  # 取前10个
            )

            # 构建文档ID到重排分数的映射
            dense_rerank_scores = {}
            bm25_rerank_scores = {}

            # 记录Dense重排分数
            for cand in dense_reranked:
                doc_id = cand['passage_id']
                dense_rerank_scores[doc_id] = cand['rerank_score']

            # 记录BM25重排分数
            for cand in bm25_reranked:
                doc_id = cand['passage_id']
                bm25_rerank_scores[doc_id] = cand['rerank_score']

            # 获取所有文档ID（两个列表的并集）
            all_doc_ids = set(list(dense_rerank_scores.keys()) + list(bm25_rerank_scores.keys()))
            # ========== 修改点4结束 ==========

            # 对每个alpha进行处理
            for alpha in args.alpha:
                weight_name = f"alpha_{alpha:.1f}"

                # ========== 修改点5：加权求和混合（添加原始分数） ==========
                mixed_results = []

                for doc_id in all_doc_ids:
                    # 获取Dense重排分数（如果没有则为0）
                    dense_rerank = dense_rerank_scores.get(doc_id, 0.0)
                    # 获取BM25重排分数（如果没有则为0）
                    bm25_rerank = bm25_rerank_scores.get(doc_id, 0.0)

                    # ========== 新增：获取原始分数 ==========
                    original_scores = doc_original_scores.get(doc_id, {'dense': 0.0, 'bm25': 0.0})
                    original_dense = original_scores['dense']
                    original_bm25 = original_scores['bm25']

                    # 计算Stage 2的混合分数（原始分数）
                    original_hybrid = alpha * original_dense + (1 - alpha) * original_bm25
                    # ========== 新增结束 ==========

                    # 加权求和：alpha * dense_rerank_score + (1-alpha) * bm25_rerank_score
                    rerank_hybrid = alpha * dense_rerank + (1 - alpha) * bm25_rerank

                    # 获取文档信息（优先从Dense候选池获取，如果没有则从BM25候选池获取）
                    doc_info = None
                    # 从Dense候选池查找
                    for cand in dense_candidates:
                        if cand['passage_id'] == doc_id:
                            doc_info = cand
                            break
                    # 如果没找到，从BM25候选池查找
                    if doc_info is None:
                        for cand in bm25_candidates:
                            if cand['passage_id'] == doc_id:
                                doc_info = cand
                                break

                    if doc_info:
                        mixed_results.append({
                            "passage_id": doc_id,
                            "title": doc_info['title'],
                            "content": doc_info['content'],
                            "rerank_score": rerank_hybrid,  # Stage 3：加权混合后的重排分数
                            "dense_rerank_score": dense_rerank,
                            "bm25_rerank_score": bm25_rerank,
                            "original_dense_score": original_dense,  # Stage 2：原始Dense分数
                            "original_bm25_score": original_bm25,  # Stage 2：原始BM25分数
                            "original_hybrid_score": original_hybrid,  # Stage 2：混合分数
                            "alpha": alpha
                        })

                # 按重排分数排序，取top-5
                mixed_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                final_results = mixed_results[:FINAL_TOP_K]
                # ========== 修改点5结束 ==========

                # 格式化结果
                formatted_results = []
                for rank, cand in enumerate(final_results, 1):
                    formatted_results.append({
                        "passage_id": cand['passage_id'],
                        "title": cand['title'],
                        "content": cand['content'],
                        "rerank_score": cand['rerank_score'],  # Stage 3：加权混合后的重排分数
                        "dense_rerank_score": cand.get('dense_rerank_score', 0.0),
                        "bm25_rerank_score": cand.get('bm25_rerank_score', 0.0),
                        "original_dense_score": cand.get('original_dense_score', 0.0),  # Stage 2：原始Dense分数
                        "original_bm25_score": cand.get('original_bm25_score', 0.0),  # Stage 2：原始BM25分数
                        "original_hybrid_score": cand.get('original_hybrid_score', 0.0),  # Stage 2：混合分数
                        "rank": rank,
                        "alpha": alpha
                    })

                # 添加到结果（保持与normal结果相同的结构）
                all_reranked_results[weight_name].append({
                    "id": question_id,
                    "question": query,
                    "source": metadata.get('source', 'unknown'),  # 从metadata获取
                    "all_reference": metadata.get('all_reference', []),  # 从metadata获取
                    "top_5": formatted_results
                })

            # 进度显示
            if (idx + 1) % args.batch_size == 0 or (idx + 1) == len(question_dirs):
                elapsed = time.time() - start_time
                progress = (idx + 1) / len(question_dirs) * 100
                avg_time = elapsed / (idx + 1) if idx > 0 else 0
                remaining = avg_time * (len(question_dirs) - idx - 1) if idx > 0 else 0

                print(f"[{idx + 1:4d}/{len(question_dirs)}] {progress:5.1f}% | "
                      f"Elapsed: {elapsed:.0f}s | "
                      f"Avg: {avg_time:.2f}s/question | "
                      f"Remaining: {remaining:.0f}s")

        except Exception as e:
            print(f"⚠️  Error processing {question_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存重排结果
    try:
        saved_files = []
        for weight_name, results in all_reranked_results.items():
            alpha_str = weight_name.replace('alpha_', '')
            filename = f"rerank_alpha_{alpha_str}.json"
            if alpha_str == "0.0":
                filename = "rerank_pure_bm25.json"
            elif alpha_str == "1.0":
                filename = "rerank_pure_dense.json"

            filepath = os.path.join(reranked_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                saved_files.append((alpha_str, filepath, len(results), file_size))

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"✅ STAGE 3 COMPLETED!")
        print(f"{'=' * 60}")
        print(f"�� Total questions processed: {len(question_dirs)}")
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"�� Avg time per question: {total_time / len(question_dirs):.2f}s")

        print(f"\n�� Reranked results saved to:")
        for alpha_str, filepath, count, file_size in saved_files:
            if alpha_str == "0.0":
                name = "Rerank Pure BM25"
            elif alpha_str == "1.0":
                name = "Rerank Pure Dense"
            else:
                name = f"Rerank α={alpha_str}"
            print(f"  {name}: {filepath} ({file_size:.1f} MB, {count} questions)")

        print(f"{'=' * 60}")

    except Exception as e:
        print(f"❌ Error saving reranked results: {e}")
        import traceback
        traceback.print_exc()

# ====================== 11. 主程序调度 ======================
def main():
    """主程序入口"""
    args = parse_arguments()

    # 显示程序信息
    print("=" * 60)
    print("HYBRID RETRIEVAL SYSTEM - THREE-STAGE PIPELINE")
    print("=" * 60)
    print("Stage 1: Build BM25 index only (--build-index)")
    print("Stage 2: Retrieve and save candidate pools & normal results (--retrieve-only --save-candidates)")
    print("Stage 3: Rerank using saved pools (--rerank-only)")
    print("=" * 60)

    # ========== 修复：确保 index_status 总是被定义 ==========
    # 初始化一个默认的索引状态
    index_status = {'exists': False, 'complete': False, 'size_gb': 0}

    # 检查索引状态（除了构建索引模式）
    if not args.build_index:
        index_status = check_pyserini_index()
        print(f"📊 Pyserini index status:")
        print(f"   Exists: {index_status['exists']}")
        print(f"   Complete: {index_status['complete']}")
        if index_status['complete']:
            print(f"   Size: {index_status['size_gb']:.1f} GB")

    # 阶段一：构建索引
    if args.build_index:
        print("\n🔄 STAGE 1: BUILDING PYSERINI INDEX")
        print("-" * 40)

        # 检查实际的索引状态（无论之前如何）
        index_status = check_pyserini_index()  # 重新检查
        if index_status['exists'] and index_status['complete']:
            print(f"✅ Index already exists at: {PYSERINI_INDEX_DIR}")
            print(f"   Size: {index_status['size_gb']:.1f} GB")

            if args.force_build:
                print("🔄 --force-build flag set, rebuilding index...")
                rebuild = True
            else:
                response = input("\n⚠️  Index already exists. Rebuild? (y/N): ")
                rebuild = response.lower() == 'y'

            if not rebuild:
                print("Exiting. Using existing index.")
                exit(0)

        success = build_pyserini_index()
        if success:
            print("\n✅ Stage 1 complete.")
            print("\n📋 Next steps:")
            print("   1. Run stage 2: python hybrid_retrieve.py --retrieve-only --save-candidates")
            print("   2. Or run stage 2 with specific range:")
            print("      python hybrid_retrieve.py --retrieve-only --save-candidates --start-idx 0 --end-idx 500")
        else:
            print("\n❌ Stage 1 failed.")
        exit(0)

    # 阶段二：仅检索
    elif args.retrieve_only:
        # 检查索引是否存在
        if not index_status['exists']:
            print(f"❌ Pyserini index not found at: {PYSERINI_INDEX_DIR}")
            if args.force_build:
                print("🔄 --force-build flag set, auto-building index...")
                success = build_pyserini_index()
                if not success:
                    exit(1)
            else:
                print("\nPlease build index first:")
                print("  python hybrid_retrieve.py --build-index")
                print("\nOr use --force-build to auto-build:")
                print("  python hybrid_retrieve.py --retrieve-only --save-candidates --force-build")
                exit(1)
        elif not index_status['complete']:
            print(f"⚠️  Index exists but may be incomplete")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                exit(1)

        # 检查保存候选池标志
        if not args.save_candidates:
            print(f"⚠️  WARNING: --save-candidates flag not set!")
            print(f"   Stage 3 will not be able to run without candidate pools.")
            response = input("Continue without saving candidate pools? (y/N): ")
            if response.lower() != 'y':
                print("Exiting. Use --save-candidates flag.")
                exit(1)

        run_retrieve_only(args)

    # 阶段三：仅重排
    elif args.rerank_only:
        run_rerank_only(args)

    # 默认：显示使用说明
    else:
        print("\n📋 USAGE EXAMPLES:")
        print("=" * 40)
        print("1. Build index (stage 1):")
        print("   python hybrid_retrieve.py --build-index")
        print("\n2. Retrieve with candidate pools (stage 2):")
        print("   python hybrid_retrieve.py --retrieve-only --save-candidates")
        print("\n3. Retrieve specific range (stage 2):")
        print("   python hybrid_retrieve.py --retrieve-only --save-candidates --start-idx 0 --end-idx 500")
        print("\n4. Rerank (stage 3):")
        print("   python hybrid_retrieve.py --rerank-only")
        print("\n5. Rerank specific range (stage 3):")
        print("   python hybrid_retrieve.py --rerank-only --start-idx 500 --end-idx 1000")
        print("\n6. Force rebuild existing index:")
        print("   python hybrid_retrieve.py --build-index --force-build")
        print("=" * 40)

        print("\n⚠️  Please specify a stage mode:")
        print("   --build-index, --retrieve-only, or --rerank-only")
        exit(0)


if __name__ == "__main__":
    main()
