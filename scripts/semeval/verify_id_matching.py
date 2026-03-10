import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
cache_dir = project_root / 'cache'
data_dir = project_root / 'data'

# Load gold file
print("Gold file sample (first 5 from ECNU_hom.gold):")
with open(data_dir / 'ECNU_hom.gold', 'r', encoding='utf-8') as f:
    for i in range(5):
        line = f.readline().strip().split('\t')
        print(f"  ID: {line[0]}, Label: {line[1]}")

# Load phase3 jsonl
print("\nPhase3 JSONL sample (first 5 from phase3_text.homographic.jsonl):")
with open(cache_dir / 'phase3_text.homographic.jsonl', 'r', encoding='utf-8') as f:
    for i in range(5):
        data = json.loads(f.readline())
        print(f"  ID: {data['id']}, Choice: {data['Choice']}")

# Check actual ID overlap
print("\nID Matching Check:")
gold_ids = set()
with open(cache_dir / 'ECNU_hom.gold', 'r', encoding='utf-8') as f:
    for line in f:
        gold_ids.add(line.strip().split('\t')[0])

pred_ids = set()
with open(cache_dir / 'phase3_text.homographic.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        pred_ids.add(data['id'])

common = gold_ids & pred_ids
print(f"Gold file IDs: {len(gold_ids)}")
print(f"Prediction IDs: {len(pred_ids)}")
print(f"Common IDs: {len(common)}")
print(f"Example common IDs: {sorted(list(common))[:5]}")
