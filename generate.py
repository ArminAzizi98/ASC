import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,TextStreamer
import numpy as np
import numpy
import json
from datasets import load_dataset
import argparse


model_name_to_gamma = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 0.27, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 0.46, "Qwen/QwQ-32B": 0.5}
model_name_to_steering_vectors = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "steering_vectors_qwen7b.pt", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "steering_vectors_llama8b.pt", "Qwen/QwQ-32B":"steering_vectors_qwq32b.pt"}
model_name_to_layer_index = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 20, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":20, "Qwen/QwQ-32B": 57}


def main():
    parser = argparse.ArgumentParser(description="Process model, problem, and steering inputs.")

    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of the model to use.'
    )

    parser.add_argument(
        '--problem',
        type=str,
        required=True,
        help='Description of the problem to solve.'
    )

    parser.add_argument(
        '--steering',
        action='store_true',
        help='Enable steering if this flag is set.'
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    torch_dtype="auto",
    device_map="balanced"
    ).eval()
    if args.steering: 
        steering_vec = torch.load("./vectors/"+model_name_to_steering_vectors[args.model_name])
        steering_vec = steering_vec.mean(dim=0)
        steering_vec = steering_vec.to(model.device).to(model.dtype)
        steering_str =model_name_to_gamma[args.model_name]

    def add_steer(_, __, output):
        gamma = steering_str
        output[0][:,-1,:] =output[0][:,-1,:] - gamma * steering_vec.to(output[0][:,-1,:].device)
        return (output[0], *output[1:])

    if args.steering:
        handle = model.model.layers[int(model_name_to_layer_index[args.model_name])].register_forward_hook(add_steer)

    inputs = tokenizer(args.problem, return_tensors="pt").to(model.device)
    print("Steering mode is: ", args.steering)
    print("Problem is: ", args.problem)
    print("*********************************************************")
    with torch.no_grad():
        out = model.generate(
        **inputs,
        max_new_tokens=8192,
        temperature=0.7,
        repetition_penalty = 1.1,
        )
    c = inputs["input_ids"].shape[-1]
    print("Answer is:")
    print(tokenizer.decode(out[0][c:], skip_special_tokens=True))
    print("Token count is: ", len(out[0][c:]))

if __name__ == '__main__':
    main()
