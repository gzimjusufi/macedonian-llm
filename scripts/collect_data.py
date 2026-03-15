import os
from datasets import load_dataset

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ── 1. Macedonian Wikipedia ──────────────────────────────────────────────────
print("Downloading Macedonian Wikipedia...")
wiki = load_dataset(
    "wikimedia/wikipedia",
    "20231101.mk",
    split="train"
)
wiki.save_to_disk(f"{RAW_DIR}/wiki_mk")
print(f"Wikipedia done — {len(wiki):,} articles")

# ── 2. mC4 Macedonian (parquet, no streaming) ────────────────────────────────
print("Downloading mC4 Macedonian...")
mc4 = load_dataset(
    "allenai/c4",
    "mk",
    split="train[:50000]",
    streaming=False
)
mc4.save_to_disk(f"{RAW_DIR}/mc4_mk")
print(f"mC4 done — {len(mc4):,} documents")

# ── 3. Helsinki-NLP Macedonian-English pairs ─────────────────────────────────
print("Downloading Helsinki translation pairs...")
helsinki = load_dataset(
    "Helsinki-NLP/opus-100",
    "en-mk",
    split="train"
)
helsinki.save_to_disk(f"{RAW_DIR}/helsinki_mk")
print(f"Helsinki done — {len(helsinki):,} pairs")

print("\nAll datasets downloaded successfully!")
print(f"Saved to: {RAW_DIR}")

# import os
# from datasets import load_dataset

# # Paths
# RAW_DIR = "data/raw"
# os.makedirs(RAW_DIR, exist_ok=True)

# # ── 1. OSCAR Macedonian corpus ──────────────────────────────────────────────
# print("Downloading OSCAR Macedonian corpus...")
# oscar = load_dataset(
#     "oscar-corpus/OSCAR-2301",
#     language="mk",
#     split="train",
#     trust_remote_code=True
# )
# oscar.save_to_disk(f"{RAW_DIR}/oscar_mk")
# print(f"OSCAR done — {len(oscar):,} documents")

# # ── 2. Macedonian Wikipedia ──────────────────────────────────────────────────
# print("Downloading Macedonian Wikipedia...")
# wiki = load_dataset(
#     "wikimedia/wikipedia",
#     "20231101.mk",
#     split="train",
#     trust_remote_code=True
# )
# wiki.save_to_disk(f"{RAW_DIR}/wiki_mk")
# print(f"Wikipedia done — {len(wiki):,} articles")

# # ── 3. Domestic-yak instruction dataset ─────────────────────────────────────
# print("Downloading domestic-yak instruction dataset...")
# yak = load_dataset(
#     "domestic-yak/mk-instruct",
#     split="train",
#     trust_remote_code=True
# )
# yak.save_to_disk(f"{RAW_DIR}/yak_instruct_mk")
# print(f"Domestic-yak done — {len(yak):,} instruction pairs")

# print("\nAll datasets downloaded successfully!")
# print(f"Saved to: {RAW_DIR}")