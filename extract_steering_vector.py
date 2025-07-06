import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy
import json

model_name = "Qwen/QwQ-32B"

model_name_to_layer_index = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 20, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":20, "Qwen/QwQ-32B": 57}
model_name_to_steering_vectors = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "steering_vectors_qwen7b.pt", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "steering_vectors_llama8b.pt", "Qwen/QwQ-32B":"steering_vectors_qwq32b.pt"}
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",          # or bfloat16 / float32 to taste
    device_map="balanced",                 # spreads across the available GPUs
    output_hidden_states=True          # ← crucial: exposes internal activations
).eval()

print(model)

with open('short_cots.json', 'r') as file:
    data = json.load(file)

with open('long_cots.json', 'r') as file:
    data1 = json.load(file)


vectors = []

for i in range(len(data)):

# 2️⃣  Tokenise your prompt -----------------------------------------------------
    P1 = data[i]["problem"] + data[i]['answer']
    inputs1 = tokenizer(P1, return_tensors="pt").to(model.device)
# 3️⃣  Run a single forward pass (no generation) -------------------------------
    with torch.no_grad():
        out = model(**inputs1, use_cache=False)   # <- no generation, no kv-cache

    last_tok1 = inputs1['input_ids'].shape[1] - 1
    act1 = out.hidden_states[int(model_name_to_layer_index[model_name])]         # shape: (batch, seq_len, hidden)
    P2 = data[i]["problem"] + data1[i]['answer']
    inputs2 = tokenizer(P2, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inputs2, use_cache=False)

    last_tok2 = inputs2['input_ids'].shape[1] - 1
    act2 = out.hidden_states[int(model_name_to_layer_index[model_name])]
    vectors.append((act1[0,last_tok1] - act2[0, last_tok2]).float().detach().cpu())
    print(i)
    torch.save(torch.tensor(numpy.array(vectors)), "./vectors/"+model_name_to_steering_vectors[model_name])
