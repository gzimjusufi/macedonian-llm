import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import wandb

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/qwen2.5-7b-base"
DATA_PATH = "data/cleaned/mk_corpus"
OUTPUT_DIR = "models/mk-qwen2.5-7b"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 8  # effective batch size = 16
LR = 2e-4
EPOCHS = 1

# ── W&B ─────────────────────────────────────────────────────────────────────
wandb.init(project="macedonian-llm", name="qwen2.5-7b-mk-v1")

# ── Load tokenizer ───────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Load model in 4-bit ──────────────────────────────────────────────────────
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

# ── LoRA config ──────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = load_from_disk(DATA_PATH)
dataset = dataset.select(range(50000))  # start with 50k for first run
print(f"Training on {len(dataset):,} documents")

# ── Training args ────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=25,
    save_steps=200,
    save_total_limit=2,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="wandb",
    optim="paged_adamw_8bit",
)

# ── Trainer ──────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer,
)

# ── Train ────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ── Save ─────────────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
wandb.finish()