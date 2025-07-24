#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5 Option A (Variation)
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Dataset: lvwerra/stack-exchange-paired (sampled, streaming-safe)
Mac mini M4 Pro / Metal(MPS)
LoRA fine-tuning + BERTScore eval + Gradio Chat UI
"""

# ------------------ Imports ------------------
import time
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from bert_score import score
import gradio as gr
from transformers import get_scheduler

# ------------------ Config ------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N_EXAMPLES = 1000
MAX_LENGTH = 256
EPOCHS = 1
BATCH_SIZE = 4
GEN_TOKENS = 128
SEED = 42
LR = 3e-5

torch.manual_seed(SEED)

# ------------------ Load & prepare dataset ------------------
streamed = load_dataset("lvwerra/stack-exchange-paired", split="test", streaming=True)
samples = []
for i, ex in enumerate(streamed):
    if ex.get("question") and ex.get("response_j"):
        samples.append({"question": ex["question"], "response_j": ex["response_j"]})
    if len(samples) >= N_EXAMPLES:
        break

dataset_raw = Dataset.from_list(samples)
print(f"✅ Loaded {len(dataset_raw)} usable examples.")

# ------------------ Tokenizer ------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------ Chat-style formatting ------------------
def to_chat(example):
    q = example["question"].strip()
    a = example["response_j"].strip()
    return {"text": f"<|user|>\n{q}\n<|assistant|>\n{a}"}

dataset_chat = dataset_raw.map(to_chat)

# ------------------ Tokenization ------------------
def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

tokenized = dataset_chat.map(tokenize_fn, batched=True)
tokenized.set_format(type="torch")

# ✨ 解決：不要な str カラムを除去
tokenized = tokenized.remove_columns(["question", "response_j", "text"])


# ------------------ Model + LoRA (MPS) ------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": "mps"},
)

peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# ------------------ Dataloader + Optimizer ------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=len(dataloader) * EPOCHS,
)

# ------------------ Custom training loop ------------------
print("🚀 Starting custom training loop...")
model.train()
t0 = time.time()
for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):
        batch = {k: v.to("mps") for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step() 
        optimizer.zero_grad()
        if step % 50 == 0:
            print(f"Epoch {epoch+1}, Step {step}: Loss = {loss.item():.4f}")
t1 = time.time()
print(f"⏱️ Training Time: {t1 - t0:.2f} sec")

# ------------------ Clear MPS cache ------------------
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# ------------------ Evaluation (BERTScore) ------------------
print("🔍 Running evaluation (generation + BERTScore)...")
eval_prompts = [f"<|user|>\n{ex['question']}\n<|assistant|>\n" for ex in dataset_raw]
refs = [ex["response_j"].strip() for ex in dataset_raw]
preds = []

model.eval()
t0 = time.time()
for prompt in eval_prompts:
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",     # ← テンソルで出力
        truncation=True,         # ← 長すぎるプロンプトを自動でカット
        max_length=1024          # ← 最大長を明示的に制限
).input_ids.to("mps")        # ← MPS（GPU）に送る

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=GEN_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    pred = pred.split("<|assistant|>")[-1].strip()
    preds.append(pred)
t1 = time.time()

P, R, F1 = score(preds, refs, lang="en")
print(f"✔️ BERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
print(f"⏱️ Inference Total Time: {t1 - t0:.2f} sec")
print(f"⏱️ Inference Avg Time/sample: {(t1 - t0)/len(preds):.2f} sec")

# ------------------ Gradio Chat UI ------------------
def chat_fn(user_input, history):
    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to("mps")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=GEN_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True).split("<|assistant|>")[-1]
    return response.strip()

gr.ChatInterface(fn=chat_fn, title="TinyLlama ChatBot (LoRA)", type="messages").launch()