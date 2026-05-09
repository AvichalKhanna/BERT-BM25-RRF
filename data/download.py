from datasets import load_dataset
import json
import os

print("Downloading MS MARCO...")
dataset = load_dataset("ms_marco", "v1.1", split="train")

# We only need 100K passages for GTX 1650
print("Extracting 100K passages...")
passages = []
for i, item in enumerate(dataset):
    if i >= 50000:
        break
    for passage in item['passages']['passage_text']:
        passages.append({
            "id": len(passages),
            "text": passage,
            "query": item['query']
        })
    if i % 10000 == 0:
        print(f"  Processed {i}/100000 queries...")

# Save to disk
os.makedirs("data", exist_ok=True)
with open("data/passages.json", "w") as f:
    json.dump(passages, f)

print(f"✅ Saved {len(passages)} passages to data/passages.json")