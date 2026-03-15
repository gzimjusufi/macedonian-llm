from datasets import load_from_disk

ds = load_from_disk('data/cleaned/mk_corpus')
print(f'Total: {len(ds):,} documents')
print(f'Sources: {set(ds["source"])}')
print()
for i in [0, 1000, 50000, 200000]:
    print(f'--- Sample {i} ({ds[i]["source"]}) ---')
    print(ds[i]['text'][:200])
    print()