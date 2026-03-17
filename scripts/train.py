import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "models/qwen2.5-7b-base"
DATA_PATH = "data/cleaned/mk_corpus"
OUTPUT_DIR = "models/mk-qwen2.5-7b"

torch.cuda.empty_cache()
wandb.init(project="macedonian-llm", name="qwen2.5-7b-mk-plain")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 128

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    device_map="auto",
    attn_implementation="eager",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)
model = get_peft_model(model, LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()

print("Loading dataset...")
dataset = load_from_disk(DATA_PATH).select(range(5000))
print(f"Training on {len(dataset):,} documents")

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="wandb",
        dataset_text_field="text",
        optim="adafactor",
        dataloader_pin_memory=False,
    ),
)

print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Model saved to {OUTPUT_DIR}")
wandb.finish()