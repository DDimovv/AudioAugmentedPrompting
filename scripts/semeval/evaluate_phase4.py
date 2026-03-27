import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def load_gold_labels(gold_file_path):
    """Load gold labels from file in format: id\tlabel."""
    labels = {}
    with open(gold_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            item_id, label = parts
            try:
                labels[item_id] = int(label)
            except ValueError:
                continue
    return labels


def choice_to_label(choice_text):
    """Convert Choice field text to binary label (pun=1, non-pun=0)."""
    if choice_text is None:
        return None

    choice_lower = str(choice_text).strip().lower()
    if not choice_lower:
        return None

    if "not a pun" in choice_lower or "non-pun" in choice_lower:
        return 0
    if "is a pun" in choice_lower or "a pun" in choice_lower:
        return 1
    return None


def load_prediction_subset(jsonl_file_path):
    """Load subset ids and mapped predictions from an evaluation JSONL file."""
    subset_ids = []
    predictions = {}

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = data.get('id')
            if not item_id:
                continue

            subset_ids.append(item_id)
            label = choice_to_label(data.get('Choice'))
            if label is not None:
                predictions[item_id] = label

    # Preserve order and uniqueness for stable evaluation
    unique_subset_ids = list(dict.fromkeys(subset_ids))
    return unique_subset_ids, predictions


def compute_binary_metrics(y_true, y_pred):
    """Compute standard binary classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'correct': sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp),
        'total': len(y_true),
    }


def calculate_subset_metrics(gold_labels, subset_ids, predictions):
    """Calculate covered-only metrics on subset IDs present in a phase3 file."""
    evaluable_subset_ids = [item_id for item_id in subset_ids if item_id in gold_labels]
    if not evaluable_subset_ids:
        return None

    covered_ids = [item_id for item_id in evaluable_subset_ids if item_id in predictions]
    if not covered_ids:
        return None

    y_true = [gold_labels[item_id] for item_id in covered_ids]
    y_pred = [predictions[item_id] for item_id in covered_ids]
    metrics = compute_binary_metrics(y_true, y_pred)

    return {
        'subset_total': len(covered_ids),
        'metrics': metrics,
    }


def print_metrics_block(filename, metrics):
    """Print per-file metrics."""
    print(f"\n{filename}")
    print("-" * 80)
    print(f"  Subset total:       {metrics['subset_total']}")

    strict = metrics['metrics']
    print("  [Metrics]")
    print(f"    Samples:          {strict['total']}")
    print(f"    Correct:          {strict['correct']}")
    print(f"    Accuracy:         {strict['accuracy']:.4f}")
    print(f"    Precision:        {strict['precision']:.4f}")
    print(f"    Recall:           {strict['recall']:.4f}")
    print(f"    F1 Score:         {strict['f1']:.4f}")


def determine_gold_labels(filename, het_gold, hom_gold):
    """Choose the correct gold label set based on filename."""
    if 'heterographic' in filename:
        return het_gold
    if 'homographic' in filename:
        return hom_gold
    return None


def main():
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / 'cache'
    data_dir = repo_root / 'data'

    het_gold = load_gold_labels(data_dir / 'ECNU_het.gold')
    hom_gold = load_gold_labels(data_dir / 'ECNU_hom.gold')

    print("=" * 80)
    print("PHASE 4 EVALUATION RESULTS")
    print("=" * 80)

    phase4_files = sorted(cache_dir.glob('phase4*.jsonl'))
    per_file = []
    skipped_uncategorized = []

    for jsonl_file in phase4_files:
        filename = jsonl_file.name
        gold_labels = determine_gold_labels(filename, het_gold, hom_gold)
        if gold_labels is None:
            skipped_uncategorized.append(filename)
            continue

        subset_ids, predictions = load_prediction_subset(jsonl_file)
        metrics = calculate_subset_metrics(gold_labels, subset_ids, predictions)
        if metrics is None:
            continue

        per_file.append((filename, metrics, subset_ids, gold_labels, predictions))
        print_metrics_block(filename, metrics)

    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)

    if not per_file:
        print("No evaluable phase4 files found.")
        print("=" * 80)
        return

    all_pred = []
    all_true = []

    total_subset = 0

    for _, _, subset_ids, gold_labels, predictions in per_file:
        covered_ids = [item_id for item_id in subset_ids if item_id in gold_labels and item_id in predictions]
        total_subset += len(covered_ids)

        for item_id in covered_ids:
            all_true.append(gold_labels[item_id])
            all_pred.append(predictions[item_id])

    strict = compute_binary_metrics(all_true, all_pred)

    print(f"\nSubset total:          {total_subset}")

    print("\n[Aggregate Metrics]")
    print(f"Samples:               {strict['total']}")
    print(f"Correct:               {strict['correct']}")
    print(f"Accuracy:              {strict['accuracy']:.4f}")
    print(f"Precision:             {strict['precision']:.4f}")
    print(f"Recall:                {strict['recall']:.4f}")
    print(f"F1 Score:              {strict['f1']:.4f}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
