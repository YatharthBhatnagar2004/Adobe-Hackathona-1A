#!/usr/bin/env python3
"""
src/main.py (Best and Final Version)
Runs high-performance parallel inference with improved post-processing
logic to ensure high accuracy by eliminating invalid headings.
"""

import json
import re
from pathlib import Path
from collections import Counter
import fitz  # PyMuPDF
import joblib
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR    = PROJECT_ROOT / "input"
OUTPUT_DIR   = PROJECT_ROOT / "output"
MODEL_DIR    = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load models once
PIPELINE      = joblib.load(MODEL_DIR / "ensemble_pipeline.pkl")
LABEL_ENCODER = joblib.load(MODEL_DIR / "label_encoder.pkl")

# Config
CONF_THRESHOLDS = {
    "Title": 0.40, "H1": 0.40, "H2": 0.40, "H3": 0.50, "Body": 0.30
}
FALSE_POS_PATTERNS = [
    r"table\s+of\s+contents?", r"contents?$", r"list\s+of\s+figures", r"list\s+of\s+tables",
    r"revision\s+history", r"references?$", r"bibliography", r"acknowledgments?",
    r"appendix\s+[A-Z0-9]", r"glossary"
]
MAX_HEADING_WORDS = 25 # Any "heading" with more words than this is demoted

def extract_features(pdf_path: Path) -> pd.DataFrame:
    rows = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return pd.DataFrame()

    for page_num, page in enumerate(doc, 1):
        blocks = sorted([b for b in page.get_text("dict")["blocks"] if b["type"] == 0 and "lines" in b], key=lambda b: b["bbox"][1])
        if not blocks: continue
        all_lines = []
        for b in blocks:
            for l in b["lines"]:
                span_sizes = [s["size"] for s in l["spans"]]
                line_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else 0
                all_lines.append({
                    "text": "".join(s["text"] for s in l["spans"]).strip(),
                    "size": line_size, "bbox": l["bbox"], "spans": l["spans"]
                })
        sizes = [line["size"] for line in all_lines if line["size"] > 4]
        body_font_size = Counter(sizes).most_common(1)[0][0] if sizes else 12.0
        for i, line in enumerate(all_lines):
            text = line["text"]
            if not text: continue
            spans = line["spans"]
            font_size = line["size"]
            font_name = Counter(s["font"] for s in spans).most_common(1)[0][0] if spans else "N/A"
            is_bold = any(("bold" in s["font"].lower() or s["flags"] & 16) for s in spans)
            space_before = line["bbox"][1] - all_lines[i-1]["bbox"][3] if i > 0 else line["bbox"][1]
            space_after = all_lines[i+1]["bbox"][1] - line["bbox"][3] if i < len(all_lines) - 1 else page.rect.height - line["bbox"][3]
            is_standalone_line = int(space_before > font_size * 0.5 and space_after > font_size * 0.5)
            chars = len(text)
            words = len(text.split())
            rows.append({
                "text": text, "font_size": font_size, "font_name": font_name,
                "is_bold": int(is_bold), "is_standalone_line": is_standalone_line,
                "page_number": page_num, "x_position": line["bbox"][0], "y_position": line["bbox"][1],
                "word_count": words, "char_count": chars, "is_all_caps": int(text.isupper() and chars > 1),
                "relative_font_size": round(font_size / body_font_size, 3) if body_font_size else 0,
                "space_before_norm": round(space_before / font_size, 3) if font_size else 0,
                "space_after_norm": round(space_after / font_size, 3) if font_size else 0,
                "starts_with_number": int(bool(re.match(r"^\d", text))),
            })
    return pd.DataFrame(rows)

def post_process(df: pd.DataFrame) -> pd.DataFrame:
    proba = PIPELINE.predict_proba(df)
    df["level"] = LABEL_ENCODER.inverse_transform(np.argmax(proba, axis=1))
    df["confidence"] = np.max(proba, axis=1)

    for lvl, th in CONF_THRESHOLDS.items():
        demote_mask = (df["level"] == lvl) & (df["confidence"] < th)
        df.loc[demote_mask, "level"] = "Body"

    for pat in FALSE_POS_PATTERNS:
        mask = df["text"].str.lower().str.contains(pat, regex=True, na=False)
        df.loc[mask, "level"] = "Body"

    long_heading_mask = (df["level"] != "Body") & (df["word_count"] > MAX_HEADING_WORDS)
    df.loc[long_heading_mask, "level"] = "Body"

    # New rule: Demote short, numeric-only headings (likely page numbers)
    numeric_heading_mask = (df["level"] != "Body") & (df["text"].str.isdigit()) & (df["text"].str.len() < 4)
    df.loc[numeric_heading_mask, "level"] = "Body"

    # New rule: Clean up concatenated text in headings
    for idx, row in df.loc[df['level'] != 'Body'].iterrows():
        if '.' in row['text']:
            df.at[idx, 'text'] = row['text'].split('.')[0].strip()

    highest_seen_level = 0
    for i, row in df.iterrows():
        if row["level"].startswith("H"):
            current_level = int(row["level"][1:])
            if current_level > highest_seen_level + 1:
                df.at[i, "level"] = f"H{highest_seen_level + 1}"
                highest_seen_level += 1
            else:
                highest_seen_level = current_level
        elif row['level'] == "Title":
            highest_seen_level = 0

    is_dupe = (df["text"].str.lower() == df["text"].str.lower().shift()) & \
              (df["page_number"] == df["page_number"].shift()) & (df["level"] != "Body")
    return df[~is_dupe]

def process_single_pdf(pdf_path: Path) -> tuple[Path, dict] | None:
    try:
        df = extract_features(pdf_path)
        if df.empty: return None
        df_processed = post_process(df)
        title_series = df_processed.loc[df_processed["level"] == "Title", "text"]
        title = title_series.iloc[0] if not title_series.empty else "Untitled Document"
        headings = df_processed[df_processed["level"].str.startswith("H")]
        outline = headings.rename(
            columns={"level": "level", "text": "text", "page_number": "page"}
        )[["level", "text", "page"]].to_dict(orient="records")
        out_data = {"title": title, "outline": outline}
        out_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
        return (out_path, out_data)
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return None

def main():
    pdf_files = sorted(list(INPUT_DIR.glob("*.pdf")))
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {INPUT_DIR.resolve()}. Aborting.")
        return
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"🚀 Found {len(pdf_files)} PDFs. Starting inference with {num_processes} parallel workers...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(process_single_pdf, pdf_files)
        for result in tqdm(results_iterator, total=len(pdf_files), desc="Running Inference"):
            if result:
                out_path, out_data = result
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out_data, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Inference complete. Output files are in {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
