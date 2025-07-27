#!/usr/bin/env python3
"""
src/create_dataset.py (FINAL VERSION)
Extracts a rich, line-by-line feature set to create a high-quality, balanced dataset.
Uses multiprocessing for maximum speed.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import multiprocessing
import fitz  # PyMuPDF
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

# Paths
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
INPUT_PDF_DIR    = PROJECT_ROOT / "data" / "samples" / "input_pdfs"
GROUNDTRUTH_DIR  = PROJECT_ROOT / "data" / "samples" / "ground_truth_jsons"
OUTPUT_CSV_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV       = OUTPUT_CSV_DIR / "labeled_data.csv"

# ──────────────────────────────────────────────────────────────
# Feature Extraction and Labeling Functions
# ──────────────────────────────────────────────────────────────

def get_robust_body_size(sizes: list[float], percentile_range=(25, 75)) -> float:
    """Calculates a robust body font size from a list of font sizes on a page."""
    if not sizes:
        return 12.0  # Default fallback
    
    lower_bound = pd.Series(sizes).quantile(percentile_range[0] / 100)
    upper_bound = pd.Series(sizes).quantile(percentile_range[1] / 100)
    
    plausible_sizes = [s for s in sizes if lower_bound <= s <= upper_bound]
    
    if not plausible_sizes:
        return Counter(sizes).most_common(1)[0][0]
        
    return Counter(plausible_sizes).most_common(1)[0][0]

def extract_features(pdf_path: Path) -> pd.DataFrame:
    """Extracts a rich feature set from each line of a PDF document."""
    rows = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return pd.DataFrame()

    for page_num, page in enumerate(doc, 1):
        blocks = sorted([b for b in page.get_text("dict")["blocks"] if b.get("type") == 0 and "lines" in b], key=lambda b: b["bbox"][1])
        if not blocks: continue

        all_lines = []
        for b in blocks:
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                # Clean non-printable characters
                text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                if not text: continue
                
                span_sizes = [s["size"] for s in l["spans"]]
                line_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else 0
                all_lines.append({
                    "text": text,
                    "size": line_size, "bbox": l["bbox"], "spans": l["spans"]
                })

        page_font_sizes = [line["size"] for line in all_lines if line["size"] > 4]
        body_font_size = get_robust_body_size(page_font_sizes)

        for i, line in enumerate(all_lines):
            text = line["text"]
            spans, font_size = line["spans"], line["size"]
            font_name = Counter(s["font"] for s in spans).most_common(1)[0][0] if spans else "N/A"
            is_bold = any(("bold" in s["font"].lower() or s["flags"] & 16) for s in spans)

            space_before = line["bbox"][1] - all_lines[i-1]["bbox"][3] if i > 0 else line["bbox"][1]
            space_after = all_lines[i+1]["bbox"][1] - line["bbox"][3] if i < len(all_lines) - 1 else page.rect.height - line["bbox"][3]
            is_standalone_line = int(space_before > font_size * 0.5 and space_after > font_size * 0.5)

            rows.append({
                "document": pdf_path.name, "text": text,
                "font_size": font_size, "font_name": font_name,
                "is_bold": int(is_bold), "is_standalone_line": is_standalone_line,
                "page_number": page_num, "x_position": line["bbox"][0],
                "y_position": line["bbox"][1], "word_count": len(text.split()),
                "char_count": len(text), "is_all_caps": int(text.isupper() and len(text) > 1),
                "relative_font_size": round(font_size / body_font_size, 3) if body_font_size else 0,
                "space_before_norm": round(space_before / font_size, 3) if font_size else 0,
                "space_after_norm": round(space_after / font_size, 3) if font_size else 0,
                "starts_with_number": int(bool(re.match(r"^\s*(\d+(\.\d+)*)\b", text))),
            })
    return pd.DataFrame(rows)

def label_rows(df: pd.DataFrame, gt: list) -> pd.DataFrame:
    """Assigns labels to rows based on fuzzy matching against ground truth."""
    df["label"] = "Body"
    by_page = defaultdict(list)
    for h in gt: by_page[h["page"]].append(h)
    
    for idx, row in df.iterrows():
        best_score, best_label = 0, "Body"
        page_gt = by_page.get(row["page_number"], [])
        if not page_gt: continue
        
        for h in page_gt:
            score = fuzz.ratio(row["text"], h["text"])
            if score > 95 and score > best_score:
                best_label, best_score = h["level"], score
        df.at[idx, "label"] = best_label
    return df

def process_single_pdf(pdf_path: Path) -> pd.DataFrame | None:
    """Extracts features and labels for a single PDF file."""
    df = extract_features(pdf_path)
    if df.empty: return None
    
    gt_file = GROUNDTRUTH_DIR / f"{pdf_path.stem}.json"
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            outline = json.load(f).get("outline", [])
        df = label_rows(df, outline)
    else:
        df["label"] = "Body"
    return df

def balance_dataset(df: pd.DataFrame, ratio: int = 1, random_state: int = 42) -> pd.DataFrame:
    """Reduces the number of 'Body' samples to balance the dataset."""
    non_body_df = df[df["label"] != "Body"]
    body_df = df[df["label"] == "Body"]

    n_non_body = len(non_body_df)
    n_body_samples = min(len(body_df), n_non_body * ratio)

    if n_body_samples < len(body_df):
        print(f"[INFO] Balancing dataset: reducing 'Body' samples from {len(body_df)} to {n_body_samples} (approx {ratio}:1 ratio).")
        body_df_sampled = body_df.sample(n=n_body_samples, random_state=random_state)
        balanced_df = pd.concat([non_body_df, body_df_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        return balanced_df
    else:
        print("[INFO] Dataset is already balanced or has few 'Body' samples. No downsampling needed.")
        return df

def main():
    """Main function to process PDFs in parallel and create a final, balanced dataset."""
    pdf_files = sorted(list(INPUT_PDF_DIR.glob("*.pdf")))
    if not pdf_files:
        print("[ERROR] No PDFs found. Aborting.")
        return
        
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"🚀 Found {len(pdf_files)} PDFs. Starting processing with {num_processes} parallel workers...")
    
    all_dfs = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(process_single_pdf, pdf_files)
        for df in tqdm(results, total=len(pdf_files), desc="Creating Dataset"):
            if df is not None and not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        print("\n[ERROR] No data was extracted. Aborting.")
        return

    print("\n[INFO] Parallel processing complete. Concatenating results...")
    full_dataset = pd.concat(all_dfs, ignore_index=True).dropna(subset=["text"])
    
    print("\nOriginal label distribution:")
    print(full_dataset['label'].value_counts(normalize=True).round(3))
    
    balanced_dataset = balance_dataset(full_dataset, ratio=1)
    
    print("\nNew balanced label distribution:")
    print(balanced_dataset['label'].value_counts(normalize=True).round(3))

    final_dataset = balanced_dataset.drop(columns=["document"], errors="ignore")
    final_dataset.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Dataset created: {len(final_dataset)} rows saved to {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()