#!/usr/bin/env python3
"""
src/main.py (FINAL VERSION)
Runs high-performance parallel inference with improved post-processing
logic to ensure high accuracy by eliminating invalid headings.
"""

import json
import re
import multiprocessing
from pathlib import Path
from collections import Counter
from typing import Any

import fitz  # PyMuPDF
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load models once
PIPELINE = joblib.load(MODEL_DIR / "ensemble_pipeline.pkl")
LABEL_ENCODER = joblib.load(MODEL_DIR / "label_encoder.pkl")

# Config
CONF_THRESHOLDS = {
    "Title": 0.50, "H1": 0.50, "H2": 0.50, "H3": 0.60, "Body": 0.30
}
FALSE_POS_PATTERNS = [
    r"^(table\s+of\s+)?contents?$", r"list\s+of\s+(figures|tables)",
    r"figure\s+\d", r"box\s+\d", r"table\s+\d",
    r"revision\s+history", r"references?$", r"bibliography", r"acknowledgments?",
    r"appendix\s+[A-Z0-9]", r"glossary", r"annex\s+[A-Z0-9]"
]
MAX_HEADING_WORDS = 25

def get_robust_body_size(sizes: list[float], percentile_range=(25, 75)) -> float:
    """Calculates a robust body font size from a list of font sizes on a page."""
    if not sizes:
        return 12.0
    
    lower_bound = pd.Series(sizes).quantile(percentile_range[0] / 100)
    upper_bound = pd.Series(sizes).quantile(percentile_range[1] / 100)
    
    plausible_sizes = [s for s in sizes if lower_bound <= s <= upper_bound]
    
    if not plausible_sizes:
        return Counter(sizes).most_common(1)[0][0]
        
    return Counter(plausible_sizes).most_common(1)[0][0]

def extract_features(pdf_path: Path) -> pd.DataFrame:
    """Extracts features from a PDF document page by page for robustness."""
    rows = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return pd.DataFrame()

    for page_num, page in enumerate(doc, 1):
        blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 0 and "lines" in b]
        if not blocks: continue
            
        all_lines = []
        for b in blocks:
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                if not text: continue

                span_sizes = [s["size"] for s in l["spans"]]
                line_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else 0
                
                all_lines.append({
                    "text": text, "size": line_size,
                    "bbox": l["bbox"], "spans": l["spans"]
                })
        
        all_lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
        
        page_font_sizes = [line["size"] for line in all_lines if line["size"] > 4]
        body_font_size = get_robust_body_size(page_font_sizes)

        for i, line in enumerate(all_lines):
            text, font_size, spans = line["text"], line["size"], line["spans"]
            font_name = Counter(s["font"] for s in spans).most_common(1)[0][0] if spans else "N/A"
            is_bold = any(("bold" in s["font"].lower() or s["flags"] & 16) for s in spans)
            
            space_before = line["bbox"][1] - all_lines[i-1]["bbox"][3] if i > 0 else page.rect.y0
            space_after = all_lines[i+1]["bbox"][1] - line["bbox"][3] if i < len(all_lines) - 1 else page.rect.height - line["bbox"][3]
            is_standalone_line = int(space_before > font_size * 0.5 and space_after > font_size * 0.5)

            rows.append({
                "text": text, "font_size": font_size, "font_name": font_name,
                "is_bold": int(is_bold), "is_standalone_line": is_standalone_line,
                "page_number": page_num, "x_position": line["bbox"][0], "y_position": line["bbox"][1],
                "word_count": len(text.split()), "char_count": len(text),
                "is_all_caps": int(text.isupper() and len(text) > 1),
                "relative_font_size": round(font_size / body_font_size, 3) if body_font_size else 0,
                "space_before_norm": round(space_before / font_size, 3) if font_size else 0,
                "space_after_norm": round(space_after / font_size, 3) if font_size else 0,
                "starts_with_number": int(bool(re.match(r"^\s*(\d+(\.\d+)*)\b", text))),
            })
            
    return pd.DataFrame(rows)

def post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Applies a series of rules to clean and structure the model's predictions."""
    if "confidence" not in df.columns:
        df["level"] = "Body"
        return df

    for lvl, th in CONF_THRESHOLDS.items():
        demote_mask = (df["level"] == lvl) & (df["confidence"] < th)
        df.loc[demote_mask, "level"] = "Body"

    for pat in FALSE_POS_PATTERNS:
        mask = df["text"].str.lower().str.contains(pat, regex=True, na=False)
        df.loc[mask, "level"] = "Body"

    long_heading_mask = (df["level"] != "Body") & (df["word_count"] > MAX_HEADING_WORDS)
    df.loc[long_heading_mask, "level"] = "Body"

    numeric_heading_mask = (df["level"] != "Body") & (df["text"].str.isdigit()) & (df["text"].str.len() < 4)
    df.loc[numeric_heading_mask, "level"] = "Body"

    highest_seen_level = 0
    for i, row in df.iterrows():
        if row["level"].startswith("H"):
            current_level = int(row["level"][1:])
            if current_level > highest_seen_level + 1:
                df.at[i, "level"] = f"H{min(current_level, highest_seen_level + 1)}"
                highest_seen_level = min(current_level, highest_seen_level + 1)
            else:
                highest_seen_level = current_level
        elif row['level'] == "Title":
            highest_seen_level = 0

    is_dupe = (df["text"].str.lower() == df["text"].str.lower().shift()) & \
              (df["page_number"] == df["page_number"].shift()) & (df["level"] != "Body")
    
    return df[~is_dupe].copy()

def find_title_heuristic(df: pd.DataFrame) -> str:
    """A fallback method to find the title based on the largest font size on the first page."""
    first_page_df = df[df['page_number'] == 1]
    if first_page_df.empty:
        return "Untitled Document"

    non_numeric_df = first_page_df[~first_page_df['text'].str.isdigit()]
    if non_numeric_df.empty:
        return "Untitled Document"
        
    title_row = non_numeric_df.loc[non_numeric_df['font_size'].idxmax()]
    return title_row['text']

def process_single_pdf(pdf_path: Path) -> None:
    """Processes a single PDF: extracts features, predicts, post-processes, and saves output."""
    try:
        df = extract_features(pdf_path)
        if df.empty: return

        proba = PIPELINE.predict_proba(df)
        df["level"] = LABEL_ENCODER.inverse_transform(np.argmax(proba, axis=1))
        df["confidence"] = np.max(proba, axis=1)

        df_processed = post_process(df)
        
        title_series = df_processed.loc[df_processed["level"] == "Title", "text"]
        title = title_series.iloc[0] if not title_series.empty else find_title_heuristic(df)

        headings = df_processed[df_processed["level"].str.startswith("H")]
        outline = headings.rename(
            columns={"text": "text", "level": "level", "page_number": "page"}
        )[["text", "level", "page"]].to_dict(orient="records")
        
        out_data = {"title": title, "outline": outline}
        out_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")

def main():
    """Main function to find PDFs and process them in parallel."""
    pdf_files = sorted(list(INPUT_DIR.glob("*.pdf")))
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {INPUT_DIR.resolve()}. Aborting.")
        return

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"🚀 Found {len(pdf_files)} PDFs. Starting inference with {num_processes} parallel workers...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_single_pdf, pdf_files), total=len(pdf_files), desc="Running Inference"))

    print(f"\n✅ Inference complete. Output files are in {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()