import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "Qwen/QwQ-32B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'balanced', torch_dtype='auto')
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


N_QUESTIONS   = 100
TEMPERATURE   = 0.3
MAX_TOKENS    = 4096
OUTFILE       = "long_cots.json"


def reg_gen(problem, generator, max_total_tokens=8192, temperature=0.7):
    prompt = problem
    full_text = prompt
    out = generator(full_text, max_new_tokens=max_total_tokens, repetition_penalty = 1.2, do_sample=True, temperature=temperature)[0]["generated_text"]
    return out


with open('short_cots.json', 'r') as file:
    data = json.load(file)

count = 0
results = []
for sample in data:
    problem = sample["problem"] + " Let's think step by step.\n"
    input_ids = tokenizer(problem, return_tensors="pt").input_ids.to(model.device)

    answer = reg_gen(problem, generator, MAX_TOKENS, TEMPERATURE)
    results.append({"problem": problem, "answer": answer})

    with open(OUTFILE, "w", encoding="utf‑8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(count)
    count = count + 1

print(f"✅  Saved {count} Q‑A pairs to {OUTFILE}")
