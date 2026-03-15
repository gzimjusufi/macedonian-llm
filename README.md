# Macedonian LLM

Fine-tuning a large language model for the Macedonian language — one of the most underserved languages in NLP.

## Project Goal
Train a high-quality Macedonian language model by fine-tuning an open-source LLM (Qwen2.5-7B) on a curated Macedonian corpus, then benchmark it against existing multilingual models.

## Dataset
- **Macedonian Wikipedia** — 131,892 articles
- **mC4 Macedonian** — 48,033 web documents  
- **Helsinki-NLP** — 204,246 translation pairs
- **Total** — 370,732 clean documents

## Stack
- Python 3.11, PyTorch 2.10, CUDA 13.2
- HuggingFace Transformers + Datasets
- PEFT (QLoRA), TRL, Unsloth
- Weights & Biases (experiment tracking)

## Progress
- [x] Phase 1 — Data collection
- [x] Phase 2 — Data cleaning
- [ ] Phase 3 — Base model selection
- [ ] Phase 4 — Fine-tuning
- [ ] Phase 5 — Evaluation & benchmarking
- [ ] Phase 6 — Demo & deployment

## Hardware
NVIDIA RTX 5070 Laptop GPU (8GB VRAM), CUDA 13.2