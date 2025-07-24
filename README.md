# 🧠 Phase 5 Mini Project: TinyLlama LoRA Fine-Tuning on StackExchange QA

This project demonstrates fine-tuning a small Large Language Model (LLM) using LoRA (Low-Rank Adaptation), evaluated with BERTScore, and deployed via Gradio Chat UI.

## 📌 Overview

- **Model**: [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Dataset**: [lvwerra/stack-exchange-paired](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)
- **Device**: Apple Silicon (Mac mini M4 Pro 24GB, MPS backend)
- **Approach**: LoRA fine-tuning + ChatTemplate formatting + BERTScore evaluation + Gradio ChatBot

## 🧪 Features

- Streaming-safe dataset sampling
- Custom training loop (memory-efficient)
- Tokenization with ChatTemplate style (`<|user|>\n...\n<|assistant|>`)
- Performance evaluation using [BERTScore](https://github.com/Tiiiger/bert_score)
- Interactive chatbot with [Gradio](https://www.gradio.app/)

## 📁 File Structure

```bash
project/
├── main.py               # Full training + eval + chat script
├── phase5-phi-stack_723.py       # Reproducible Conda environment
├── LICENSE
└── README.md
