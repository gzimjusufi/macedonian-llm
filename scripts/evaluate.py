import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

BASE_MODEL_PATH = "models/qwen2.5-7b-base"
FINETUNED_MODEL = "GzimJusufi/macedonian-qwen2.5-7b"
TEST_DATA_PATH = "data/cleaned/mk_corpus"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path, adapter_path=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer

def compute_perplexity(model, tokenizer, texts, max_length=128):
    losses = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(DEVICE)
            if inputs["input_ids"].shape[1] < 10:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    perplexity = np.exp(np.mean(losses))
    return perplexity

# Load test set — 200 samples not seen during training
print("Loading test data...")
dataset = load_from_disk(TEST_DATA_PATH)
test_data = dataset.select(range(10000, 10200))
texts = [row["text"][:500] for row in test_data]
print(f"Evaluating on {len(texts)} test samples")

# Evaluate base model
print("\nEvaluating base model...")
base_model, base_tokenizer = load_model(BASE_MODEL_PATH)
base_ppl = compute_perplexity(base_model, base_tokenizer, texts)
print(f"Base model perplexity: {base_ppl:.2f}")
del base_model
torch.cuda.empty_cache()

# Evaluate fine-tuned model
print("\nEvaluating fine-tuned model...")
ft_model, ft_tokenizer = load_model(BASE_MODEL_PATH, adapter_path=FINETUNED_MODEL)
ft_ppl = compute_perplexity(ft_model, ft_tokenizer, texts)
print(f"Fine-tuned model perplexity: {ft_ppl:.2f}")
del ft_model
torch.cuda.empty_cache()

# Results
improvement = ((base_ppl - ft_ppl) / base_ppl) * 100
print(f"\n{'='*50}")
print(f"Base model perplexity:       {base_ppl:.2f}")
print(f"Fine-tuned model perplexity: {ft_ppl:.2f}")
print(f"Improvement:                 {improvement:.1f}%")
print(f"{'='*50}")

results = {
    "base_perplexity": base_ppl,
    "finetuned_perplexity": ft_ppl,
    "improvement_percent": improvement,
    "test_samples": len(texts),
}
with open("results/evaluation.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to results/evaluation.json")