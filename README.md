# SCAR: A Cascaded Auditing Framework

This repository contains the official code, system prompts, and evaluation schemas for the SCAR framework.

## Contents

```text
.
├── data/                          # Data, questions, and outputs
│   ├── output/                    # Example retrieval outputs
│   ├── question/                  # Evaluation datasets extracted and curated from NQ and ASQA
│   └── time_and_cost/             # vLLM-based Cost/Latency benchmark scripts & outputs
├── vector_db_build.py             # Offline: Wikipedia dense vector DB construction
├── dense_retrieve_only.py         # Retrieval: Pure dense vector retrieval
├── hybrid_retrieve.py             # Retrieval: Hybrid (Dense + BM25 + Reranker) retrieval
├── stage1.py                      # SCAR (Transformers): Stage 1
├── stage2_qwen.py                 # SCAR (Transformers): Stage 2 (Qwen)
├── stage3_mn.py                   # SCAR (Transformers): Stage 3 (Mistral-NeMo)
└── command.txt                    # Setup commands for downloading data and models
```

## ⚠️ Important Notes

1. **Backbone Models (Transformers vs. vLLM)**: 
   - The root-level scripts (e.g., `stage1.py`, `stage2_qwen.py`, `stage3_mn.py`) use the **HuggingFace Transformers** library for model inference. 
   - The scripts in the `data/time_and_cost/` directory use **vLLM** for highly efficient inference, designed to benchmark latency and cost comparisons against the vanilla generator.
2. **Pipeline Execution Order**: 
   - **Root Directory**: The execution order of the model variants in the root-level scripts is **reversed** compared to the order presented in our paper. The scripts are highly modular; users can easily change the execution sequence, adjust model hyper-parameters, and modify the system prompts within the code to suit their specific needs.
   - **`time_and_cost` Directory**: The scripts and data in this directory contain **all the execution orders** (including the correct ones described in the paper) for the cost and latency evaluation.
3. **Setup and Data Preparation**:
   - `command.txt` contains the bash commands for downloading the Wikipedia dataset, downloading the required models (e.g., BGE-M3, Llama-3.1, Qwen2.5, Mistral-NeMo) via `modelscope` or `huggingface-cli`, and setting up the environment.
   - The `data/question/` directory contains the evaluation question sets that have been extracted and curated from the NQ (Natural Questions) and ASQA datasets.
4. **Cost & Latency Analysis**:
   - The `data/time_and_cost/` directory is dedicated to evaluating the computational cost and latency of the SCAR cascaded pipeline versus a standard vanilla VLM baseline. It contains complete intermediate inputs/outputs (e.g., `stage1_output_stage2_input.json`) to showcase the data flow.
