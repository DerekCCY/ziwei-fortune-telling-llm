import json
from pathlib import Path
import argparse
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from scripts.formatting.sft_format_prompt import format_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--train_path", type=str, default="data/sft_dataset/train.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ziwei-qwen3-4b-ft")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=4096)
    args = parser.parse_args()

    print("[SFT] Loading dataset...")
    raw_dataset = load_dataset("json", data_files=args.train_path, split="train")

    # Convert JSON → prompt text
    def format_row(row):
        return {"text": format_example(row)}

    dataset = raw_dataset.map(format_row)

    print("[SFT] Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("[SFT] Applying QLoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)

    print("[SFT] Preparing dataloader...")
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("[SFT] Training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none"
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    print(f"[SFT] Done! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
