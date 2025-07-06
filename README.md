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

### Performance Comparison: CoT vs. ASC

| Model                          | Method | MATH500 Acc. (%) | MATH500 Tokens | GSM8K Acc. (%) | GSM8K Tokens |
|-------------------------------|--------|------------------|----------------|----------------|--------------|
| Deepseek-R1-Distill-Qwen-7B   | CoT    | 88.8             | 3984           | 88.6           | 1080         |
|                               | ASC    | **89.0**         | **1543**       | 88.6           | **536**      |
| Deepseek-R1-Distill-LLaMA-8B  | CoT    | 89.2             | 3554           | 89.1           | 2610         |
|                               | ASC    | **89.2**         | **2353**       | **89.3**       | **850**      |
| QwQ-32B                       | CoT    | 93.8             | 4508           | **96.5**       | 1530         |
|                               | ASC    | 94.2             | **2222**       | 96.4           | **830**      |


## 🛠️ Setup

```bash
git clone https://github.com/your-username/ASC.git
cd ASC
pip install -r requirements.txt
