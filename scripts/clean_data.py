import os
import re
from datasets import load_from_disk, Dataset, concatenate_datasets

RAW_DIR = "data/raw"
CLEAN_DIR = "data/cleaned"
os.makedirs(CLEAN_DIR, exist_ok=True)

# ── Cleaning function ────────────────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return None
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove lines with too many special characters (likely garbage)
    ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if ratio < 0.5:
        return None
    # Filter too short or too long
    if len(text) < 50 or len(text) > 10000:
        return None
    return text

# ── 1. Clean Wikipedia ───────────────────────────────────────────────────────
print("Cleaning Wikipedia...")
wiki = load_from_disk(f"{RAW_DIR}/wiki_mk")
wiki_texts = []
for row in wiki:
    cleaned = clean_text(row["text"])
    if cleaned:
        wiki_texts.append({"text": cleaned, "source": "wikipedia"})
print(f"Wikipedia: {len(wiki)} → {len(wiki_texts)} documents")

# ── 2. Clean mC4 ────────────────────────────────────────────────────────────
print("Cleaning mC4...")
mc4 = load_from_disk(f"{RAW_DIR}/mc4_mk")
mc4_texts = []
for row in mc4:
    cleaned = clean_text(row["text"])
    if cleaned:
        mc4_texts.append({"text": cleaned, "source": "mc4"})
print(f"mC4: {len(mc4)} → {len(mc4_texts)} documents")

# ── 3. Clean Helsinki (extract Macedonian side only) ────────────────────────
print("Cleaning Helsinki...")
helsinki = load_from_disk(f"{RAW_DIR}/helsinki_mk")
helsinki_texts = []
for row in helsinki:
    cleaned = clean_text(row["translation"]["mk"])
    if cleaned:
        helsinki_texts.append({"text": cleaned, "source": "helsinki"})
print(f"Helsinki: {len(helsinki)} → {len(helsinki_texts)} documents")

# ── 4. Combine and deduplicate ───────────────────────────────────────────────
print("Combining and deduplicating...")
all_texts = wiki_texts + mc4_texts + helsinki_texts
seen = set()
deduped = []
for row in all_texts:
    key = row["text"][:100]  # first 100 chars as dedup key
    if key not in seen:
        seen.add(key)
        deduped.append(row)

print(f"Combined: {len(all_texts)} → {len(deduped)} after dedup")

# ── 5. Save ──────────────────────────────────────────────────────────────────
dataset = Dataset.from_list(deduped)
dataset.save_to_disk(f"{CLEAN_DIR}/mk_corpus")
print(f"\nDone! Clean corpus saved: {len(dataset):,} documents")
print(f"Saved to: {CLEAN_DIR}/mk_corpus")