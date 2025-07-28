import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import multiprocessing
import fitz  
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
                "document": pdf_path.name, "text": text, "font_size": font_size, "font_name": font_name,
                "is_bold": int(is_bold), "page_number": page_num, "x_position": line["bbox"][0],
                "y_position": line["bbox"][1], "word_count": len(text.split()), "char_count": len(text),
                "is_all_caps": int(text.isupper() and len(text) > 1),
                "relative_font_size": round(font_size / body_font_size, 3) if body_font_size else 0,
                "starts_with_number": int(bool(re.match(r"^\s*(\d+(\.\d+)*)\b", text))),
                "font_size_diff_prev": font_size_diff_prev, "is_prev_line_blank": is_prev_line_blank,
                "font_size_diff_next": font_size_diff_next, "is_next_line_blank": is_next_line_blank,
            })
    return pd.DataFrame(rows)

def label_rows(df: pd.DataFrame, gt: list) -> pd.DataFrame:
    df["label"] = "Body"
    by_page = defaultdict(list); [by_page[h["page"]].append(h) for h in gt]
    
    for idx, row in df.iterrows():
        page_gt = by_page.get(row["page_number"], []);
        if not page_gt: continue
        matches = [(h['level'], fuzz.ratio(row["text"], h["text"])) for h in page_gt]
        if not matches: continue
        best_match = max(matches, key=lambda item: item[1])
        if best_match[1] > 95: df.at[idx, "label"] = best_match[0]
    return df

def process_single_pdf(pdf_path: Path) -> pd.DataFrame | None:
    df = extract_features(pdf_path);
    if df.empty: return None
    gt_file = GROUNDTRUTH_DIR / f"{pdf_path.stem}.json"
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f: outline = json.load(f).get("outline", [])
        df = label_rows(df, outline)
    else: df["label"] = "Body"
    return df

def balance_dataset(df: pd.DataFrame, ratio: int = 1, random_state: int = 42) -> pd.DataFrame:
    non_body_df, body_df = df[df["label"] != "Body"], df[df["label"] == "Body"]
    n_body_samples = min(len(body_df), len(non_body_df) * ratio)
    if n_body_samples < len(body_df):
        print(f"[INFO] Balancing: reducing 'Body' samples from {len(body_df)} to {n_body_samples} (~{ratio}:1).")
        body_df_sampled = body_df.sample(n=n_body_samples, random_state=random_state)
        return pd.concat([non_body_df, body_df_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

def main():
    pdf_files = sorted(list(INPUT_PDF_DIR.glob("*.pdf")))
    if not pdf_files: print("[ERROR] No PDFs found. Aborting."); return
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Found {len(pdf_files)} PDFs. Starting processing with {num_processes} workers...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        all_dfs = list(tqdm(pool.imap_unordered(process_single_pdf, pdf_files), total=len(pdf_files), desc="Creating Dataset"))
    
    full_dataset = pd.concat([df for df in all_dfs if df is not None], ignore_index=True).dropna(subset=["text"])
    if full_dataset.empty: print("\n[ERROR] No data was extracted. Aborting."); return

    print("\nOriginal label distribution:"); print(full_dataset['label'].value_counts(normalize=True).round(3))
    balanced_dataset = balance_dataset(full_dataset, ratio=1)
    print("\nNew balanced label distribution:"); print(balanced_dataset['label'].value_counts(normalize=True).round(3))

    balanced_dataset.drop(columns=["document"], errors="ignore").to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset created: {len(balanced_dataset)} rows saved to {OUTPUT_CSV.resolve()}")

if __name__ == "__main__": multiprocessing.freeze_support(); main()