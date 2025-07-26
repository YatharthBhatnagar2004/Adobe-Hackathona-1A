#!/usr/bin/env python3
"""
src/create_dataset.py (Best and Final Version)
Extracts a rich, line-by-line feature set to create the highest quality dataset.
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
PROJECT_ROOT     = Path(__file__).parent.parent
INPUT_PDF_DIR    = PROJECT_ROOT / "data" / "samples" / "input_pdfs"
GROUNDTRUTH_DIR  = PROJECT_ROOT / "data" / "samples" / "ground_truth_jsons"
OUTPUT_CSV_DIR   = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV       = OUTPUT_CSV_DIR / "labeled_data.csv"

# ──────────────────────────────────────────────────────────────
# Feature Extraction Function (Line-by-Line)
# ──────────────────────────────────────────────────────────────
def extract_features(pdf_path: Path) -> pd.DataFrame:
    rows = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return pd.DataFrame()

    for page_num, page in enumerate(doc, 1):
        # Extract text blocks and sort them vertically
        blocks = sorted([b for b in page.get_text("dict")["blocks"] if b["type"] == 0 and "lines" in b], key=lambda b: b["bbox"][1])

        if not blocks:
            continue

        # Get all line texts with their sizes and positions for context
        all_lines = []
        for b in blocks:
            for l in b["lines"]:
                # Get the most common font size and style for the line
                span_sizes = [s["size"] for s in l["spans"]]
                line_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else 0
                all_lines.append({
                    "text": "".join(s["text"] for s in l["spans"]).strip(),
                    "size": line_size,
                    "bbox": l["bbox"],
                    "spans": l["spans"]
                })

        # Calculate body font size for the page
        sizes = [line["size"] for line in all_lines if line["size"] > 4]
        body_font_size = Counter(sizes).most_common(1)[0][0] if sizes else 12.0

        for i, line in enumerate(all_lines):
            text = line["text"]
            if not text:
                continue

            spans = line["spans"]
            font_size = line["size"]
            font_name = Counter(s["font"] for s in spans).most_common(1)[0][0] if spans else "N/A"
            is_bold = any(("bold" in s["font"].lower() or s["flags"] & 16) for s in spans)

            # Contextual spacing
            space_before = line["bbox"][1] - all_lines[i-1]["bbox"][3] if i > 0 else line["bbox"][1]
            space_after = all_lines[i+1]["bbox"][1] - line["bbox"][3] if i < len(all_lines) - 1 else page.rect.height - line["bbox"][3]
            is_standalone_line = int(space_before > font_size * 0.5 and space_after > font_size * 0.5)

            chars = len(text)
            words = len(text.split())

            rows.append({
                "document": pdf_path.name,
                "text": text,
                "font_size": font_size,
                "font_name": font_name,
                "is_bold": int(is_bold),
                "is_standalone_line": is_standalone_line,
                "page_number": page_num,
                "x_position": line["bbox"][0],
                "y_position": line["bbox"][1],
                "word_count": words,
                "char_count": chars,
                "is_all_caps": int(text.isupper() and chars > 1),
                "relative_font_size": round(font_size / body_font_size, 3) if body_font_size else 0,
                "space_before_norm": round(space_before / font_size, 3) if font_size else 0,
                "space_after_norm": round(space_after / font_size, 3) if font_size else 0,
                "starts_with_number": int(bool(re.match(r"^\d", text))),
            })
    return pd.DataFrame(rows)

def label_rows(df: pd.DataFrame, gt: list) -> pd.DataFrame:
    df["label"] = "Body"
    by_page = defaultdict(list)
    for h in gt: by_page[h["page"]].append(h)
    for idx, row in df.iterrows():
        best_score, best_label = 0, "Body"
        for h in by_page.get(row["page_number"], []):
            score = fuzz.ratio(row["text"], h["text"])
            if score > 95 and score > best_score:
                best_label, best_score = h["level"], score
        df.at[idx, "label"] = best_label
    return df

def process_single_pdf(pdf_path: Path) -> pd.DataFrame | None:
    df = extract_features(pdf_path)
    if df.empty: return None
    gt_file = GROUNDTRUTH_DIR / f"{pdf_path.stem}.json"
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            outline = json.load(f).get("outline", [])
        df = label_rows(df, outline)
    else: df["label"] = "Body"
    return df

def main():
    pdf_files = list(INPUT_PDF_DIR.glob("*.pdf"))
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
    full_dataset = full_dataset.drop(columns=["document"], errors="ignore")
    full_dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Dataset created: {len(full_dataset)} rows saved to {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
