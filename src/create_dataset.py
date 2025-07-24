import pandas as pd
import json
import os
import fitz  # PyMuPDF
import re
from collections import Counter
from thefuzz import fuzz
from PIL import Image
import pytesseract
import io
import multiprocessing
import joblib

TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
SAMPLES_DIR = "data/samples"
PDF_DIR = os.path.join(SAMPLES_DIR, "input_pdfs")
JSON_DIR = os.path.join(SAMPLES_DIR, "ground_truth_jsons")
PROCESSED_DATA_PATH = "data/processed/labeled_data.csv"
CLASSIFIER_PATH = "models/classifier_v2.joblib"

if os.path.exists(TESSERACT_CMD_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

def _ocr_worker(image_bytes, return_dict):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        return_dict['ocr_data'] = ocr_data
    except Exception as e:
        return_dict['error'] = str(e)

def extract_text_with_ocr(page: fitz.Page, ocr_timeout=10) -> list:
    ocr_lines = []
    try:
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=_ocr_worker, args=(img_bytes, return_dict))
        p.start()
        p.join(ocr_timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            print(f"  -> OCR timed out on page {page.number + 1}")
            return ocr_lines
        if 'error' in return_dict:
            print(f"  -> OCR failed on page {page.number + 1}: {return_dict['error']}")
            return ocr_lines
        ocr_data = return_dict.get('ocr_data', None)
        if ocr_data is not None and not ocr_data.empty:
            for _, group in ocr_data.groupby(['block_num', 'par_num', 'line_num']):
                line_text = ' '.join(group['text'].astype(str))
                x0, y0 = group['left'].min(), group['top'].min()
                x1, y1 = (group['left'] + group['width']).max(), (group['top'] + group['height']).max()
                is_all_caps = line_text.isupper() and len(line_text) > 1
                ocr_lines.append({
                    'text': line_text, 'font_size': 12, 'font_name': 'OCR',
                    'is_bold': False, 'bbox': (x0, y0, x1, y1),
                    'page_number': page.number + 1, 'y_position': y0,
                    'is_all_caps': is_all_caps, 'x_position': x0
                })
    except Exception as e:
        print(f"  -> OCR failed on page {page.number + 1}: {e}")
    return ocr_lines

def extract_features_from_pdf(pdf_path: str) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    lines_data = []
    for page_num, page in enumerate(doc):
        text_page = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)
        page_lines = []
        for block in text_page["blocks"]:
            if block['type'] == 0:
                for line in block['lines']:
                    line_text = " ".join(span['text'] for span in line['spans']).strip()
                    if not line_text: continue
                    font_sizes = [span['size'] for span in line['spans']]
                    font_names = [span['font'] for span in line['spans']]
                    most_common_font = Counter(font_names).most_common(1)[0][0] if font_names else 'N/A'
                    is_bold = any(style in most_common_font.lower() for style in ['bold', 'black', 'heavy'])
                    is_all_caps = line_text.isupper() and len(line_text) > 1
                    x_position = round(line['bbox'][0])
                    y_position = line['bbox'][1]
                    page_lines.append({
                        'text': line_text, 'font_size': max(font_sizes) if font_sizes else 0,
                        'font_name': most_common_font, 'is_bold': is_bold,
                        'bbox': line['bbox'], 'page_number': page_num + 1,
                        'y_position': y_position,
                        'is_all_caps': is_all_caps, 'x_position': x_position
                    })
        if len("".join([line['text'] for line in page_lines])) < 100:
            print(f"  -> Page {page_num + 1} has little text. Attempting OCR...")
            ocr_page_lines = extract_text_with_ocr(page)
            lines_data.extend(ocr_page_lines if ocr_page_lines else page_lines)
        else:
            lines_data.extend(page_lines)
    if not lines_data:
        return pd.DataFrame()
    df = pd.DataFrame(lines_data)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    if not df.empty:
        df['is_centered'] = abs(df['x_position'] - df['x_position'].median()) < 20
        df['is_title_case'] = df['text'].apply(lambda x: x.istitle())
        df['has_numbers'] = df['text'].str.contains(r'\d')
        df['has_symbols'] = df['text'].str.contains(r'[•\-\*]')
    return df

def load_ground_truth_from_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        "title": data.get("title", ""),
        "headings": data.get("outline") or data.get("headings") or []
    }

def normalize_text(text: str) -> str:
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    text = re.sub(r'\b(manual|chapter|section|page|p)\b', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def ngram_overlap(a: str, b: str, n: int = 3) -> float:
    a_ngrams = set([a[i:i+n] for i in range(len(a)-n+1)])
    b_ngrams = set([b[i:i+n] for i in range(len(b)-n+1)])
    return len(a_ngrams & b_ngrams) / max(len(a_ngrams), len(b_ngrams)) if a_ngrams and b_ngrams else 0.0

def multiline_candidates(df, max_lines=3):
    for idx in range(len(df)):
        yield (idx,), df.iloc[idx]['text']
        if idx < len(df) - 1:
            yield (idx, idx+1), df.iloc[idx]['text'] + ' ' + df.iloc[idx+1]['text']
        if idx < len(df) - 2:
            yield (idx, idx+1, idx+2), df.iloc[idx]['text'] + ' ' + df.iloc[idx+1]['text'] + ' ' + df.iloc[idx+2]['text']

def label_dataset_with_ground_truth(features_df: pd.DataFrame, ground_truth: dict) -> pd.DataFrame:
    features_df['label'] = 'Body'
    match_algo = fuzz.token_set_ratio
    all_gt_headings = ground_truth.get("headings", [])

    # Define features list at the top so it's always available
    features = [
        'font_size', 'is_bold', 'x_position', 'y_position', 'word_count', 'is_all_caps',
        'is_centered', 'is_title_case', 'has_numbers', 'has_symbols'
    ]

    # Title
    if ground_truth.get("title"):
        title_text = normalize_text(ground_truth["title"])
        title_df = features_df[features_df['page_number'].isin([1, 2, 3])].copy()
        X = features_df[features].copy()
        title_df['match_ratio'] = title_df['text'].apply(lambda x: match_algo(normalize_text(x), title_text))
        if not title_df.empty and title_df['match_ratio'].max() > 60:
            best_match = title_df.loc[title_df['match_ratio'].idxmax()]
            features_df.loc[best_match.name, 'label'] = 'Title'

    # Headings H1–H7
    for heading in all_gt_headings:
        heading_text = heading.get("text", "").strip()
        level = heading.get("level", "H3").upper()
        if not (level.startswith("H") and level[1:].isdigit() and 1 <= int(level[1:]) <= 7):
            level = "H3"
        try:
            page_num = int(heading.get("page", heading.get("page_number")))
        except:
            continue
        heading_text_norm = normalize_text(heading_text)
        search_df = features_df[features_df['page_number'].between(page_num - 3, page_num + 3)]
        candidates = search_df[(search_df['font_size'] >= search_df['font_size'].quantile(0.80)) |
                               (search_df['is_bold']) |
                               (search_df['is_all_caps']) |
                               (search_df['x_position'] < search_df['x_position'].quantile(0.20))]
        best_score = 0
        best_idx = None
        for idx_tuple, candidate_text in multiline_candidates(candidates):
            cand_norm = normalize_text(candidate_text)
            fuzzy_score = match_algo(cand_norm, heading_text_norm)
            ngram_score = ngram_overlap(cand_norm, heading_text_norm)
            score = 0.7 * fuzzy_score / 100 + 0.3 * ngram_score
            if score > best_score:
                best_score = score
                best_idx = idx_tuple
        if best_score > 0.6:
            for i in best_idx:
                features_df.loc[candidates.index[i], 'label'] = level

    # Classifier
    try:
        clf = joblib.load(CLASSIFIER_PATH)
        for col in features:
            if col not in features_df.columns:
                features_df[col] = 0
        X = features_df[features]
        for col in X.select_dtypes(include=bool).columns:
            X[col] = X[col].astype(int).to_numpy(copy=True)
        preds = clf.predict(X)
        for idx, pred in enumerate(preds):
            if features_df.iloc[idx]['label'] == 'Body' and (pred.startswith("H") or pred == "Title"):
                features_df.iloc[idx, features_df.columns.get_loc('label')] = pred
    except Exception as e:
        print(f"  -> Classifier integration failed: {e}")

    return features_df

def create_labeled_dataset():
    pdf_files = {os.path.splitext(f)[0]: f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')}
    json_files = {os.path.splitext(f)[0]: f for f in os.listdir(JSON_DIR) if f.lower().endswith('.json')}
    base_names = sorted(list(set(pdf_files.keys()) & set(json_files.keys())))
    print(f"Found {len(base_names)} matching PDF/JSON pairs to process.")
    all_labeled_dfs = []

    for name in base_names:
        print(f"\nProcessing {name}...")
        pdf_path = os.path.join(PDF_DIR, pdf_files[name])
        json_path = os.path.join(JSON_DIR, json_files[name])
        try:
            features_df = extract_features_from_pdf(pdf_path)
            if features_df.empty:
                print(f"  -> Warning: No text extracted. Skipping.")
                continue
            # Skip empty JSON files
            if os.path.getsize(json_path) == 0:
                print(f"  -> ERROR: JSON file {json_files[name]} is empty. Skipping.")
                continue
            try:
                gt = load_ground_truth_from_json(json_path)
            except json.JSONDecodeError as e:
                print(f"  -> ERROR: JSON file {json_files[name]} is invalid: {e}. Skipping.")
                continue
            labeled_df = label_dataset_with_ground_truth(features_df, gt)
            labeled_df['filename'] = pdf_files[name]
            all_labeled_dfs.append(labeled_df)
        except Exception as e:
            print(f"  -> ERROR processing {name}: {e}")
            continue

    if not all_labeled_dfs:
        print("No data was processed. Exiting.")
        return

    final_dataset = pd.concat(all_labeled_dfs, ignore_index=True)
    output_columns = [
        'filename', 'page_number', 'label', 'text', 'font_size', 'font_name', 'is_bold', 'word_count',
        'is_all_caps', 'x_position', 'y_position', 'is_centered', 'is_title_case', 'has_numbers', 'has_symbols'
    ]
    final_dataset = final_dataset[[col for col in output_columns if col in final_dataset.columns]]

    # 🧹 Data Cleanup
    allowed_labels = {'Title'} | {f'H{i}' for i in range(1, 8)} | {'Body'}
    initial_count = len(final_dataset)
    final_dataset = final_dataset[
        final_dataset['label'].isin(allowed_labels) &
        final_dataset['text'].notna() &
        final_dataset['text'].str.strip().ne("") &
        final_dataset['page_number'].notna()
    ].copy()
    final_dataset['page_number'] = final_dataset['page_number'].astype(int)
    dropped = initial_count - len(final_dataset)
    if dropped > 0:
        print(f"  -> Dropped {dropped} invalid or malformed rows.")

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    final_dataset.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\n✅ Dataset created at '{PROCESSED_DATA_PATH}' with {len(final_dataset)} rows.")
    print("\nLabel distribution:")
    print(final_dataset['label'].value_counts())

if __name__ == "__main__":
    create_labeled_dataset()
