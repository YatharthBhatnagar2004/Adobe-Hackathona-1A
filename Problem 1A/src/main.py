from sklearn.pipeline import Pipeline 
import json
import re
import multiprocessing
from pathlib import Path
from collections import Counter
import fitz
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

PIPELINE = joblib.load(MODEL_DIR / "ensemble_pipeline.pkl")
LABEL_ENCODER = joblib.load(MODEL_DIR / "label_encoder.pkl")

# Config
CONF_THRESHOLDS = {"Title": 0.6, "H1": 0.6, "H2": 0.65, "H3": 0.7, "Body": 0.3}
FALSE_POS_PATTERNS = [r"^(table\s+of\s+)?contents?$", r"list\s+of\s+(figures|tables)", r"figure\s+\d", r"box\s+\d", r"table\s+\d",
                      r"revision\s+history", r"references?$", r"bibliography", r"acknowledgments?", r"appendix\s+[A-Z0-9]", r"glossary", r"annex\s+[A-Z0-9]"]
MAX_HEADING_WORDS = 25
PDF_PROCESS_TIMEOUT = 120

def get_robust_body_size(sizes: list[float], percentile_range=(25, 75)) -> float:
    if not sizes: return 12.0
    lower, upper = pd.Series(sizes).quantile([p/100 for p in percentile_range])
    plausible = [s for s in sizes if lower <= s <= upper]
    return Counter(plausible or sizes).most_common(1)[0][0]

def extract_features(pdf_path: Path) -> pd.DataFrame:
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
                text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                if not text: continue
                span_sizes = [s["size"] for s in l["spans"]]
                line_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else 0
                all_lines.append({"text": text, "size": line_size, "bbox": l["bbox"], "spans": l["spans"]})

        page_font_sizes = [line["size"] for line in all_lines if line["size"] > 4]
        body_font_size = get_robust_body_size(page_font_sizes)

        for i, line in enumerate(all_lines):
            text, font_size, spans = line["text"], line["size"], line["spans"]
            font_name = Counter(s["font"] for s in spans).most_common(1)[0][0] if spans else "N/A"
            is_bold = any(("bold" in s["font"].lower() or s["flags"] & 16) for s in spans)
            
            prev_line = all_lines[i-1] if i > 0 else {"bbox": (0,0,0,0), "size": font_size}
            space_before = line["bbox"][1] - prev_line["bbox"][3]
            font_size_diff_prev = font_size - prev_line["size"]
            is_prev_line_blank = int(space_before > (prev_line["size"] * 1.5))

            next_line = all_lines[i+1] if i < len(all_lines) - 1 else {"bbox": (0, page.rect.height, 0, page.rect.height), "size": font_size}
            space_after = next_line["bbox"][1] - line["bbox"][3]
            font_size_diff_next = font_size - next_line["size"]
            is_next_line_blank = int(space_after > (font_size * 1.5))

            rows.append({
                "text": text, "font_size": font_size, "font_name": font_name, "is_bold": int(is_bold), "page_number": page_num,
                "x_position": line["bbox"][0], "y_position": line["bbox"][1], "word_count": len(text.split()), "char_count": len(text),
                "is_all_caps": int(text.isupper() and len(text) > 1),
                "relative_font_size": round(font_size / body_font_size, 3) if body_font_size else 0,
                "starts_with_number": int(bool(re.match(r"^\s*(\d+(\.\d+)*)\b", text))),
                "font_size_diff_prev": font_size_diff_prev, "is_prev_line_blank": is_prev_line_blank,
                "font_size_diff_next": font_size_diff_next, "is_next_line_blank": is_next_line_blank,
            })
    return pd.DataFrame(rows)

def post_process(df: pd.DataFrame) -> pd.DataFrame:
    if "confidence" not in df.columns: df["level"] = "Body"; return df
    for lvl, th in CONF_THRESHOLDS.items():
        df.loc[(df["level"] == lvl) & (df["confidence"] < th), "level"] = "Body"
    for pat in FALSE_POS_PATTERNS:
        df.loc[df["text"].str.contains(pat, regex=True, na=False, case=False), "level"] = "Body"
    df.loc[(df["level"] != "Body") & (df["word_count"] > MAX_HEADING_WORDS), "level"] = "Body"
    df.loc[(df["level"] != "Body") & (df["text"].str.isdigit()) & (df["text"].str.len() < 4), "level"] = "Body"
    highest_seen_level = 0
    for i, row in df.iterrows():
        if row["level"].startswith("H"):
            current_level = int(row["level"][1:])
            if current_level > highest_seen_level + 1:
                df.at[i, "level"] = f"H{min(current_level, highest_seen_level + 1)}"
            highest_seen_level = int(df.at[i, "level"][1:])
        elif row['level'] == "Title": highest_seen_level = 0
    is_dupe = (df["text"].str.lower() == df["text"].str.lower().shift()) & \
              (df["page_number"] == df["page_number"].shift()) & (df["level"] != "Body")
    return df[~is_dupe].copy()

def find_title_heuristic(df: pd.DataFrame) -> str:
    first_page_df = df[df['page_number'] == 1]
    if first_page_df.empty: return "Untitled Document"
    non_numeric_df = first_page_df[~first_page_df['text'].str.isdigit()]
    if non_numeric_df.empty: return "Untitled Document"
    return non_numeric_df.loc[non_numeric_df['font_size'].idxmax()]['text']

def process_single_pdf(pdf_path: Path) -> None:
    try:
        df = extract_features(pdf_path)
        if df.empty: return

        clf = PIPELINE.named_steps['clf']
        for est_wrapper in clf.estimators_:
            est = est_wrapper
            if isinstance(est_wrapper, Pipeline):
                 est = est_wrapper.steps[-1][1]

            if hasattr(est, 'set_params'):
                if 'device' in est.get_params():
                    est.set_params(device='cpu')

        proba = PIPELINE.predict_proba(df)
        df["level"] = LABEL_ENCODER.inverse_transform(np.argmax(proba, axis=1))
        df["confidence"] = np.max(proba, axis=1)
        df_processed = post_process(df)
        
        title = df_processed.loc[df_processed["level"] == "Title", "text"].iloc[0] if not df_processed.loc[df_processed["level"] == "Title"].empty else find_title_heuristic(df)
        headings = df_processed[df_processed["level"].str.startswith("H")]
        outline = headings.rename(columns={"text": "text", "level": "level", "page_number": "page"})[["text", "level", "page"]].to_dict(orient="records")
        
        out_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f: json.dump({"title": title, "outline": outline}, f, indent=4, ensure_ascii=False)
    except Exception as e: print(f"Error processing {pdf_path.name}: {e}")

def main():
    pdf_files = sorted(list(INPUT_DIR.glob("*.pdf")))
    if not pdf_files: print(f"[ERROR] No PDFs found in {INPUT_DIR.resolve()}. Aborting."); return

    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Found {len(pdf_files)} PDFs. Starting CPU inference with {num_processes} workers...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = [pool.apply_async(process_single_pdf, (pdf_path,)) for pdf_path in pdf_files]
        with tqdm(total=len(pdf_files), desc="Running Inference") as pbar:
            for res in results:
                try: res.get(timeout=PDF_PROCESS_TIMEOUT)
                except multiprocessing.TimeoutError: pbar.write(f"Task timed out. Skipping.")
                except Exception as e: pbar.write(f"Worker process failed: {e}")
                pbar.update(1)

    print(f"\nInference complete. Output files are in {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()