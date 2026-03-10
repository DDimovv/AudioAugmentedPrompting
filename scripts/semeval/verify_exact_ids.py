import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
cache_dir = project_root / 'cache'
data_dir = project_root / 'data'

# Load gold labels
gold_labels = {}
with open(data_dir / 'ECNU_hom.gold', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            item_id, label = parts
            gold_labels[item_id] = int(label)

# Load predictions
predictions = {}
with open(cache_dir / 'phase3_text.homographic.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            item_id = data.get('id')
            choice = data.get('Choice')
            if item_id and choice is not None:
                if "is a pun" in choice and "not a pun" not in choice:
                    predictions[item_id] = 1
                elif "is not a pun" in choice:
                    predictions[item_id] = 0
        except json.JSONDecodeError:
            continue

# Find common IDs
common_ids = set(gold_labels.keys()) & set(predictions.keys())

# Show exact matching for sample IDs
print("Verification: Exact ID Matching")
print("=" * 80)
print(f"\nTotal common IDs: {len(common_ids)}")

print("\nSample 1: Checking first 5 common IDs:")
for idx, id in enumerate(sorted(list(common_ids))[:5]):
    gold_label = gold_labels[id]
    pred_label = predictions[id]
    match = "✓" if gold_label == pred_label else "✗"
    print(f"  {match} ID: {id:8} | Gold: {gold_label} | Pred: {pred_label}")

print("\nSample 2: Checking for specific ID 'hom_209':")
if 'hom_209' in gold_labels:
    print(f"  Gold has hom_209: {gold_labels['hom_209']}")
else:
    print(f"  Gold does NOT have hom_209")
    
if 'hom_209' in predictions:
    print(f"  Prediction has hom_209: {predictions['hom_209']}")
else:
    print(f"  Prediction does NOT have hom_209")

if 'hom_209' in common_ids:
    print(f"  ✓ hom_209 is in COMMON IDs - will be evaluated")
else:
    print(f"  ✗ hom_209 is NOT in common IDs - will be skipped")

print("\nValidation: Checking that ID keys match exactly")
print("-" * 80)
# Verify that when we use an ID from common_ids, both dictionaries have that exact key
mismatches = 0
for id in common_ids:
    if id not in gold_labels:
        print(f"ERROR: {id} not in gold_labels!")
        mismatches += 1
    if id not in predictions:
        print(f"ERROR: {id} not in predictions!")
        mismatches += 1

if mismatches == 0:
    print("✓ All common IDs exist in both gold_labels and predictions dictionaries")
    print("✓ ID matching is working correctly")
