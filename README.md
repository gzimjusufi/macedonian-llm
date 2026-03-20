# Macedonian LLM

Fine-tuning a large language model for the Macedonian language — one of the most underserved languages in NLP.

## Project Goal
Train a high-quality Macedonian language model by fine-tuning an open-source LLM (Qwen2.5-7B) on a curated Macedonian corpus, then benchmark it against existing multilingual models.

## Model
🤗 [GzimJusufi/macedonian-qwen2.5-7b](https://huggingface.co/GzimJusufi/macedonian-qwen2.5-7b)

## Dataset
- **Macedonian Wikipedia** — 131,892 articles
- **mC4 Macedonian** — 48,033 web documents
- **Helsinki-NLP** — 204,246 translation pairs
- **Total** — 370,732 clean documents
- **Training subset** — 10,000 documents (first fine-tune run)

## Results
- Starting loss: 1.909
- Final loss: 1.087
- Training time: ~3.5 hours on Tesla T4

## Stack
- Python 3.11, PyTorch 2.10, CUDA 13.2
- HuggingFace Transformers + Datasets
- PEFT (QLoRA), TRL, Unsloth
- Weights & Biases (experiment tracking)

## Evaluation Results
| Model | Perplexity | 
|-------|-----------|
| Qwen2.5-7B (base) | 10.64 |
| Macedonian fine-tune (ours) | 5.21 |
| **Improvement** | **51.0%** |

Evaluated on 50 held-out Macedonian Wikipedia articles.

## Progress
- [x] Phase 1 — Data collection
- [x] Phase 2 — Data cleaning
- [x] Phase 3 — Base model selection (Qwen2.5-7B)
- [x] Phase 4 — Fine-tuning
- [x] Phase 5 — Evaluation & benchmarking
- [ ] Phase 6 — Demo & deployment

## Hardware
NVIDIA RTX 5070 Laptop GPU (8GB VRAM), CUDA 13.2
Training on Tesla T4 (Google Colab)