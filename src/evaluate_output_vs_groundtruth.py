import os
import json
from difflib import SequenceMatcher
from collections import Counter

def normalize(text):
    return text.lower().strip()

def fuzzy_match(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def heading_key(h):
    return (normalize(h['text']), h['level'].upper(), int(h.get('page_number', h.get('page', 0))))

def match_headings(gt_headings, out_headings, threshold=0.85):
    gt_set = set()
    for h in gt_headings:
        gt_set.add((normalize(h['text']), h['level'].upper(), int(h.get('page_number', h.get('page', 0)))))
    out_set = set()
    for h in out_headings:
        out_set.add((normalize(h['text']), h['level'].upper(), int(h.get('page_number', h.get('page', 0)))))
    # Fuzzy match: allow for small text differences
    matched_gt = set()
    matched_out = set()
    for gt in gt_headings:
        for out in out_headings:
            if gt['level'].upper() == out['level'].upper() and int(gt.get('page_number', gt.get('page', 0))) == int(out.get('page_number', out.get('page', 0))):
                if fuzzy_match(gt['text'], out['text']) > threshold:
                    matched_gt.add(heading_key(gt))
                    matched_out.add(heading_key(out))
    tp = len(matched_gt)
    fn = len(gt_set - matched_gt)
    fp = len(out_set - matched_out)
    return tp, fp, fn, gt_set - matched_gt, out_set - matched_out

def evaluate():
    gt_dir = 'data/samples/ground_truth_jsons'
    out_dir = 'output'
    gt_files = {os.path.splitext(f)[0]: os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.json')}
    out_files = {os.path.splitext(f)[0]: os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.json')}
    common = set(gt_files.keys()) & set(out_files.keys())
    total_tp = total_fp = total_fn = 0
    title_correct = 0
    title_total = 0
    print("\nEvaluation Report:\n------------------")
    for name in sorted(common):
        gt = load_json(gt_files[name])
        out = load_json(out_files[name])
        gt_title = gt.get('title', '').strip()
        out_title = out.get('title', '').strip()
        title_total += 1
        if fuzzy_match(gt_title, out_title) > 0.85:
            title_correct += 1
        gt_headings = gt.get('headings') or gt.get('outline') or []
        out_headings = out.get('headings') or []
        tp, fp, fn, missed, extra = match_headings(gt_headings, out_headings)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(f"\nFile: {name}")
        print(f"  Title: {'CORRECT' if fuzzy_match(gt_title, out_title) > 0.85 else 'WRONG'}")
        print(f"  Headings: TP={tp}, FP={fp}, FN={fn}")
        if missed:
            print("    Missed ground truth headings:")
            for h in missed:
                print(f"      {h}")
        if extra:
            print("    Extra output headings:")
            for h in extra:
                print(f"      {h}")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print("\n==============================")
    print(f"Overall Title Accuracy: {title_correct}/{title_total} ({100*title_correct/title_total:.1f}%)")
    print(f"Overall Headings Precision: {precision:.3f}")
    print(f"Overall Headings Recall:    {recall:.3f}")
    print(f"Overall Headings F1:        {f1:.3f}")
    print("==============================\n")

if __name__ == "__main__":
    evaluate() 