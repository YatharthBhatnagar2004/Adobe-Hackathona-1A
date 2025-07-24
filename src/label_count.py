import os
import json
from collections import Counter

gt_dir = "data/samples/ground_truth_jsons"
label_counts = Counter()
for fname in os.listdir(gt_dir):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(gt_dir, fname), 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data.get('title'):
        label_counts['Title'] += 1
    for heading in data.get('headings', []) + data.get('outline', []):
        level = heading.get('level', 'H3').upper()
        label_counts[level] += 1
print(label_counts)