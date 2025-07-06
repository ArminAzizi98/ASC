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
```

## 🧪 Inference Example

The easiest way to use ASC during inference is with the `--steering` flag:

```bash
python -u generate.py \
  --model_name "Qwen/QwQ-32B" \
  --problem '''Define
\[p = \sum_{k = 1}^\infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]
Find a way to write
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]
in terms of $p$ and $q.$''' \
  --steering
```
You may provide any math problem as the `--problem` argument. 



## 🧭 Creating Steering Vectors

To generate a steering vector for a new model or domain, follow these steps:

1. **Generate Concise CoTs using GPT-4o**  
   Requires access to the OpenAI ChatGPT API. This script prompts GPT-4o to produce math-centric, minimal-English rationales.
   ```bash
   python generate_short_cots.py

2. **Generate Verbose CoTs using the Target Model**
   The following script generates standard chain-of-thought (CoT) outputs from your chosen reasoning model.
   ```bash
   python generate_long_cots.py

3. **Extract the Steering Vector**
   Use this script to compute the activation-space vector that maps verbose to concise reasoning, based on the CoT pairs.
   ```bash
   python extract_steering_vector.py

   
## ✅ Supported Models

The following models have been tested and are currently supported by ASC:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `Qwen/QwQ-32B`

> ℹ️ More models will be added soon. Contributions are welcome!

