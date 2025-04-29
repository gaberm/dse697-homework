import os
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

def main():
    base_model_name = "models/llama-2-7b"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=".cache",
    )
    print("Cuda available:", torch.cuda.is_available())
    base_model = base_model.to("cuda" if torch.cuda.is_available() else "cpu")
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Generate a response and decode it to a string
    system_prompt = "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>"
    user_prompts = [
        "What is 2 + 2?", 
        "What is the capital of France?", 
        "Who long is the Great Wall of China?",
        "What is the largest mammal?",
        "What is the speed of light?",
        ]
    responses_pretrained = []
    for user_prompt in tqdm(user_prompts, desc="Generating example responses"):
        input = f"{system_prompt}\n\n{user_prompt} [/INST]"
        inputs = tokenizer(input, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        base_model = base_model.to("cuda")

        generated_ids = base_model.generate(
            inputs["input_ids"],
            max_length=512,  
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        responses_pretrained.append(output)

    # Save the example responses to a file
    with open("data/responses_pretrained.txt", "w") as f:
        for idx, pair in enumerate(zip(user_prompts, responses_pretrained)):
            f.write(f"--- Example {idx+1} ---\n")
            f.write(f"Prompt: {pair[0]}\n")
            f.write(f"Response: {pair[1]}\n\n")

    # Data set
    dataset = load_from_disk("data/databricks-dolly-15k")
    text = []
    for row in tqdm(dataset['train'], desc="Processing dataset", unit="example"):
        text.append(f"{system_prompt}\n\n{row['instruction']}\n[/INST]\n{row['response']}</s>")
    train_data = Dataset.from_dict({"text": text})

    # Training Params
    train_params = SFTConfig(
        output_dir=".cache/checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        #optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=len(train_data) // 2,
        learning_rate=4e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        dataset_text_field="text",
        report_to=[],
    )

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, peft_parameters)
    model.print_trainable_parameters()

    # Trainer with LoRA configuration
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=train_data,
        peft_config=peft_parameters,
        processing_class=tokenizer,
        args=train_params
    )

    # Training
    fine_tuning.train()
    model.save_pretrained("models/llama-2-7b-finetuned")

    # Regenerate the example responses with the fine-tuned model
    responses_finetuned = []
    for user_prompt in tqdm(user_prompts, desc="Generating example responses"):
        input = f"{system_prompt}\n\n{user_prompt} [/INST]"
        inputs = tokenizer(input, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v, in inputs.items()}
        model = model.to("cuda")

        generated_ids = model.generate(
            inputs["input_ids"],
            max_length=128,  
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        responses_finetuned.append(output)

    # Save the fine-tuned example responses to a file
    with open("data/responses_finetuned.txt", "w") as f:
        for idx, pair in enumerate(zip(user_prompts, responses_finetuned)):
            f.write(f"--- Example {idx+1} ---\n")
            f.write(f"Prompt: {pair[0]}\n")
            f.write(f"Response: {pair[1]}\n\n")

if __name__ == "__main__":
    main()