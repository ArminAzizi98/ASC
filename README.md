# Activation-Steered Compression (ASC)

**Activation-Steered Compression (ASC)** is a training-free method that compresses verbose reasoning in Large Language Models (LLMs) at inference time by manipulating internal activations. It achieves substantial reductions in Chain-of-Thought (CoT) length while preserving, or even improving, answer accuracy — enabling faster, more efficient, and cost-effective deployment of reasoning models.

> 📄 This repository accompanies our paper:  
> **Activation Steering for Chain-of-Thought Compression**

## 🚀 Overview

Chain-of-Thought prompting improves reasoning but often leads to:
- Verbose explanations
- Redundant reasoning steps
- Increased token usage and latency

ASC addresses this inefficiency by:
- Extracting a **steering vector** from paired verbose vs. concise rationales
- Injecting it into the model’s residual stream at inference time
- Compressing CoTs without retraining or fine-tuning

## 🧠 Key Features

- ⚙️ **Training-free**: Works on any model without parameter updates
- 💡 **Concise reasoning**: Reduces CoT length by up to 67%
- ⚡ **Efficient inference**: Up to 2.73× speedup in wall-clock time
- 🧪 **Model-agnostic**: Works across 7B, 8B, and 32B parameter models
- 📐 **Theoretical guarantees**: KL-bounded scaling ensures safe intervention

## 📊 Results Summary

| Model | Dataset | Accuracy (Δ) | Token Reduction |
|-------|---------|---------------|-----------------|
| DeepSeek-LLaMA-8B | MATH500 | ↔︎ (no drop) | ↓33.8% |
| DeepSeek-Qwen-7B  | GSM8K   | ↔︎           | ↓67.43% |
| QwQ-32B           | MATH500 | ↑ +0.4%      | ↓50.7% |

## 🛠️ Setup

```bash
git clone https://github.com/your-username/ASC.git
cd ASC
pip install -r requirements.txt
