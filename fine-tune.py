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

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    base_model_name = "models/llama-2-7b"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cuda:0",
        cache_dir=".cache",
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1

    # Generate a response and decode it to a string
    example_prompts = [
        "What is 2 + 2?", 
        "What is the capital of France?", 
        "Who long is the Great Wall of China?",
        "What is the largest mammal?",
        "What is the speed of light?",
        ]
    example_responses = []
    for prompt in example_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model = model.to("cuda")

        generated_ids = model.generate(
            inputs["input_ids"],
            max_length=512,  
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        example_responses.append(generated_text)

    # Save the example responses to a file
    with open("data/example_responses.txt", "w") as f:
        for i, response in enumerate(example_responses):
            f.write(f"--- Example {i+1} ---\n")
            f.write(f"Response: {response}\n\n")

    # Data set
    dataset = load_from_disk("data/databricks-dolly-15k")
    text = []
    for i in range(len(dataset)):
        text.append(f"<s>[INST] {dataset['instruction'][i]} [/INST] {dataset['response'][i]} </s>")
    train_data = Dataset.from_dict({"text": text})

    # Training Params
    train_params = SFTConfig(
        output_dir="./results_modified",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        #optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=50,
        learning_rate=4e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        dataset_text_field="text",
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
    example_responses_finetuned = []
    for prompt in example_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
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

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        example_responses_finetuned.append(generated_text)

    # Save the fine-tuned example responses to a file
    with open("data/example_responses_finetuned.txt", "w") as f:
        for i, response in enumerate(example_responses_finetuned):
            f.write(f"--- Example {i+1} ---\n")
            f.write(f"Response: {response}\n\n")

if __name__ == "__main__":
    main()