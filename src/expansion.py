import pandas as pd
import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Fix for padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU

pipe = pipeline(
    "text-generation",
    model=model.to("cuda" if device == 0 else "cpu"),
    tokenizer=tokenizer,
    device=device
)

def query_expansion(pipeline, prompt):
    response = pipeline(prompt, max_new_tokens=30)[0]["generated_text"]
    return response

def batch_query_expansion(pipeline, queries, persona):
    expert_prompt = "Rewrite the following question as if asked by a domain expert, using formal and technical language"
    novice_prompt = "Rewrite this query using broader and simpler terms"

    if persona == "expert_user":
      prompt = expert_prompt
    elif persona == "novice_user":
      prompt = novice_prompt
    else:
       raise ValueError("Insert as 'persona' parameter either 'expert_user' or 'novice_user'")

    prompts = [
      f"""{prompt}: '{query}'\nRewritten:"""
      for query in queries
    ]
    
    print(f"All {len(prompts)} prompts created. Starting pipeline processing...")

    responses = pipeline(
        prompts,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=pipeline.tokenizer.pad_token_id,
    )

    print(f"Pipeline processing completed. Got {len(responses)} responses.")

    folder_path = "content"
    os.makedirs(folder_path, exist_ok=True)
    experiment_name = f"{persona}_expansion"
    destination_path = os.path.join(folder_path, experiment_name) + ".txt"
    with open(destination_path, "w") as f:
        for r in responses:
            f.write(r[0]["generated_text"] + "\n")
    print(f"Output saved to: {destination_path}")

    expanded_queries = [r[0]["generated_text"] for r in responses]
    return expanded_queries


import logging
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Example usage
test_df = pd.read_csv("/content/test.csv")
queries = test_df["query_text"].unique()

batch_query_expansion(pipe, queries, "novice_user")
batch_query_expansion(pipe, queries, "expert_user")