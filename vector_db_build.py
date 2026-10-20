"""
vector_db_build.py — Wikipedia 全量段落向量库构建脚本

功能概述
--------
使用 BGE-M3 模型（双卡 fp16）对 Wikipedia psgs_w100 段落集进行批量向量化，
并构建 FAISS IndexFlatIP 向量检索索引，用于后续的稠密检索（Dense Retrieval）。

处理流程
--------
1. 读取 DATA_DIR 下所有 *.parquet 文件（每行含 title / text 字段）
2. 用 BGEM3FlagModel 将 text 编码为 1024 维归一化向量（batch = 48 × 2 卡）
3. 向量逐文件 add 进 faiss.IndexFlatIP（归一化后内积 == 余弦相似度）
4. 同步写出 SAVE_DIR/wikipedia_full_passages.jsonl（id/title/text，与索引行号对齐）
5. 全部完成后写出 SAVE_DIR/wikipedia_full.index

输出文件
--------
- SAVE_DIR/wikipedia_full_passages.jsonl : 段落原文元数据（与 index 行号一一对应）
- SAVE_DIR/wikipedia_full.index          : FAISS 向量索引

温控 / 静音机制（本脚本特色）
----------------------------
- 每处理完一个文件读取 nvidia-smi 最高温度
- > 80°C  ：强制停工，冷却至 65°C 以下才恢复
- 75~80°C ：休眠 8s
- 70~75°C ：休眠 5s
- 65~70°C ：休眠 3s
- < 65°C  ：休眠 1s（让风扇喘口气）
- 每文件结束执行 gc + torch.cuda.empty_cache() 深度清理显存

可调参数
--------
- MODEL_PATH / DATA_DIR / SAVE_DIR : 模型与数据路径
- BATCH_SIZE_PER_GPU : 单卡 batch（总 batch = 该值 × 2）
- MAX_LENGTH         : 编码最大长度（默认 512）
- HIGH_TEMP_THRESHOLD / SAFE_TEMP_THRESHOLD : 温控阈值

运行环境
--------
- 依赖: FlagEmbedding, faiss, torch(CUDA), pandas, pyarrow(读 parquet)
- 硬件: 至少 2 张 GPU（脚本默认 CUDA_VISIBLE_DEVICES=0,1）

注意事项
--------
- 脚本末尾使用 os._exit(0) 强制退出，属有意为之（规避多进程池/守护线程卡死），
  若需在外部框架内调用请注意该行为。
- 中途异常仅跳过当前文件并继续，不会中断整体任务。
"""



import os
import pandas as pd
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import json
import glob
import time
import torch
import gc
import subprocess
from tqdm import tqdm

# ====================== 双卡极致静音 & 恒温配置区 ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "/root/autodl-tmp/SCAR/models/bge-m3"
DATA_DIR = "/root/autodl-tmp/SCAR/data/wiki/data/psgs_w100/no_embeddings"
SAVE_DIR = "/root/autodl-tmp/SCAR/data/wiki/output"

BATCH_SIZE_PER_GPU = 48
MAX_LENGTH = 512
HIGH_TEMP_THRESHOLD = 80  # 超过此温度触发强制停工
SAFE_TEMP_THRESHOLD = 65  # 降到此温度才恢复工作


# ====================================================================

def get_gpu_max_temp():
    """实时获取所有可见 GPU 中的最高温度"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                                capture_output=True, text=True)
        temps = [int(t) for t in result.stdout.strip().split('\n') if t.strip()]
        return max(temps) if temps else 0
    except:
        return 0


def thermal_control_wait():
    """智能冷却：根据温度动态调整等待时间"""
    current_temp = get_gpu_max_temp()

    if current_temp > HIGH_TEMP_THRESHOLD:
        print(f"\n🌡️  温度警报: {current_temp}°C > {HIGH_TEMP_THRESHOLD}°C")
        print("💤 强制冷却...")
        cool_time = 0
        while get_gpu_max_temp() > SAFE_TEMP_THRESHOLD:
            time.sleep(5)
            cool_time += 5
            temp_now = get_gpu_max_temp()
            print(f"  等待{cool_time}s，当前温度: {temp_now}°C")
        print(f"✅ 冷却完成，恢复工作")
    elif current_temp > 75:
        wait = 8
        print(f"🌡️  {current_temp}°C，散热{wait}s")
        time.sleep(wait)
    elif current_temp > 70:
        wait = 5
        print(f"🌡️  {current_temp}°C，散热{wait}s")
        time.sleep(wait)
    elif current_temp > 65:
        wait = 3
        print(f"🌡️  {current_temp}°C，散热{wait}s")
        time.sleep(wait)
    else:
        # 温度很低，短暂休息让风扇喘口气
        time.sleep(1)


def memory_cleanup():
    """深度清理内存与显存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


if __name__ == '__main__':
    print("=" * 60)
    print("🎯 双卡静音模式 (针对psgs_w100优化)")
    print(f"   段落长度: ~100词 | 批次: {BATCH_SIZE_PER_GPU}×2")
    print(f"   目标温度: <{HIGH_TEMP_THRESHOLD}°C")
    print("=" * 60)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    TARGET_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    print(f"📁 找到 {len(TARGET_FILES)} 个文件")

    # 初始化模型
    print("🚀 加载模型...")
    start_load = time.time()

    # 1. 初始化模型
    model = BGEM3FlagModel(
        MODEL_PATH,
        use_fp16=True,
        devices=['cuda:0', 'cuda:1'],
        normalize_embeddings=True
    )
    print(f"✅ 模型加载: {time.time() - start_load:.1f}秒")

    # 2. 初始化 FAISS (IndexFlatIP)
    index = faiss.IndexFlatIP(1024)

    # 输出文件
    output_jsonl = os.path.join(SAVE_DIR, "wikipedia_full_passages.jsonl")
    output_index = os.path.join(SAVE_DIR, "wikipedia_full.index")

    global_id_counter = 0
    start_time = time.time()

    pbar = tqdm(TARGET_FILES, desc="双卡编码进度", unit="file",
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for file_path in pbar:
            file_name = os.path.basename(file_path)
            try:
                # A. 读取数据
                df = pd.read_parquet(file_path, columns=["title", "text"])
                passages = df["text"].astype(str).tolist()
                titles = df["title"].astype(str).tolist()
                del df

                # B. 推理 (内部已处理多进程分发)
                total_batch_size = BATCH_SIZE_PER_GPU * 2
                outputs = model.encode(
                    passages,
                    batch_size=total_batch_size,
                    max_length=MAX_LENGTH,
                    return_dense=True
                )

                embeddings = outputs['dense_vecs']
                if embeddings is not None:
                    index.add(embeddings.astype(np.float32))

                # C. 写入结果
                lines = []
                for title, text in zip(titles, passages):
                    line = {"id": global_id_counter, "title": title, "text": text}
                    lines.append(json.dumps(line, ensure_ascii=False) + "\n")
                    global_id_counter += 1
                f_out.writelines(lines)

                # D. 更新状态
                elapsed = time.time() - start_time
                speed = global_id_counter / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    "已处理": f"{global_id_counter:,}",
                    "速度": f"{speed:.0f}条/秒",
                    "温度": f"{get_gpu_max_temp()}°C"
                })

                # E. 散热与清理
                del passages, titles, outputs, embeddings
                memory_cleanup()
                thermal_control_wait()  # 这里的逻辑比单纯 sleep 更稳

            except Exception as e:
                print(f"\n❌ 处理文件 {file_name} 时报错: {e}")
                continue

    # 3. 保存索引
    print(f"\n💾 正在保存 FAISS 索引 (当前条数: {index.ntotal:,})...")
    try:
        faiss.write_index(index, output_index)
        print(f"✅ 索引已安全写入: {output_index}")
    except Exception as e:
        print(f"❌ 索引保存失败: {e}")

    # ====================== 新增：优雅退出逻辑 ======================
    print("\n🧹 正在清理系统资源...")

    # 1. 显式关闭 BGE 的多进程池，防止 __del__ 时找不到 SIGTERM
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'stop_multi_process_pool'):
            model.model.stop_multi_process_pool()
        # 彻底释放模型对象
        del model
    except:
        pass

    # 2. 深度清理内存
    memory_cleanup()

    # 3. 强制给系统一点时间回收信号量 (解决 UserWarning)
    time.sleep(2)

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"🎉 任务完美结束！")
    print(f"   总耗时: {total_time / 3600:.2f} 小时")
    print(f"   最终条目: {global_id_counter:,} 条")
    print(f"   存储路径: {SAVE_DIR}")
    print("=" * 60)

    # 4. 强制退出，防止某些守护线程卡死
    os._exit(0)
