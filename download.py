from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import torch

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    model_id = "meta-llama/Llama-2-7b-hf"
    save_directory = "models/llama-2-7b"

    # Loading and saving the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    # Loading and saving the dataset
    dataset = load_dataset("databricks/databricks-dolly-15k")
    dataset.save_to_disk("data/databricks-dolly-15k")

    # Creating the sample file
    with open("data/databricks-dolly-15k.txt", "w") as f:
        f.write("=== databricks-dolly-15k Dataset Samples ===\n\n")
        for i in range(5):
            f.write(f"--- Example {i+1} ---\n")
            f.write(f"Instruction: {dataset['train'][i]['instruction']}\n")
            f.write(f"Response: {dataset['train'][i]['response']}\n\n")

if __name__ == "__main__":
    main()