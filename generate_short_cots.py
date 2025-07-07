import os, json, time, random, pathlib
from datasets import load_dataset, concatenate_datasets
import openai
from openai import OpenAI, RateLimitError, APIError

# ---------- configuration ----------
N_QUESTIONS   = 100
MODEL_NAME    = "gpt-4o-mini"
TEMPERATURE   = 0.3
MAX_TOKENS    = 4096
DELAY_SECONDS = (1.0, 2.0)
OUTFILE       = "short_cots.json"
SEED          = 42

# ---- dataset cache dir (make sure it is writable) ----
CACHE_DIR = pathlib.Path.home() / "hf_datasets_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---- OpenAI client ----
client = OpenAI(api_key="your_openai_api_key")

# ---------- load 100 random problems ----------
subjects = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Load the train split for each subject and store them in a list
datasets = [
    load_dataset("EleutherAI/hendrycks_math", name=subject, split="train")
    for subject in subjects
]

# Concatenate all datasets into one
ds = concatenate_datasets(datasets)

problems = ds.shuffle(seed=SEED).select(range(N_QUESTIONS))

SYSTEM_MSG = (
    "You are an expert competition mathematician. "
    "When you give a solution, express it **primarily in formal math notation"
    "** with *minimal* surrounding English. "
    "Return the final answer in a boxed format."
)
USER_TEMPLATE = (
    "Solve the following problem step by step. "
    "**Answer almost entirely in math notation**; keep English words to the bare minimum.\n\n"
    "Problem:\n{problem}"
)

def solve_with_gpt4(problem_text: str) -> str:
    """Call GPT‑4 with exponential‑back‑off retry on transient errors."""
    backoff = 1.0
    while True:
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",   "content": USER_TEMPLATE.format(problem=problem_text)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content.strip()
        except (RateLimitError, APIError) as e:
            # exponential back‑off up to ~64 s
            wait = min(backoff, 64)
            print(f"[warn] {e.__class__.__name__} – retrying in {wait:.1f}s")
            time.sleep(wait + random.random())
            backoff *= 2

results = []
for idx, row in enumerate(problems):
    print(f"🧮  Solving problem {idx+1}/{N_QUESTIONS}")
    answer = solve_with_gpt4(row["problem"])
    results.append({"problem": row["problem"], "answer": answer})
    time.sleep(random.uniform(*DELAY_SECONDS))

with open(OUTFILE, "w", encoding="utf‑8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅  Saved {len(results)} Q‑A pairs to {OUTFILE}")
