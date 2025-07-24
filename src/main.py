import pandas as pd
import fitz  # PyMuPDF
import joblib
import os
import json
from collections import Counter

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
MODEL_PATH = "/app/models/classifier_v2.joblib"

def extract_features_from_pdf(pdf_path: str) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    all_lines = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    line_text = " ".join([span['text'] for span in line['spans']]).strip()
                    if not line_text: continue
                    font_sizes = [s['size'] for s in line['spans']]
                    font_names = [s['font'] for s in line['spans']]
                    all_lines.append({
                        'text': line_text,
                        'font_size': max(font_sizes),
                        'is_bold': int('bold' in font_names[0].lower()),
                        'bbox': line['bbox'],
                        'page_number': page.number + 1
                    })

    df = pd.DataFrame(all_lines)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['is_all_caps'] = df['text'].apply(lambda x: x.isupper())
    df['x_position'] = df['bbox'].apply(lambda b: round(b[0]))
    df['y_position'] = df['bbox'].apply(lambda b: round(b[1]))

    # Additional features
    df['is_centered'] = df['x_position'].apply(lambda x: abs(x - 300) < 50)
    df['is_title_case'] = df['text'].apply(lambda t: t.istitle())
    df['has_numbers'] = df['text'].str.contains(r'\d').astype(int)
    df['has_symbols'] = df['text'].str.contains(r'[•#@*$%+=><|_/\\\[\]\(\):;,.-]').astype(int)

    return df

def process_all_pdfs():
    model = joblib.load(MODEL_PATH)
    features = [
        'font_size', 'is_bold', 'x_position', 'y_position', 'word_count', 'is_all_caps',
        'is_centered', 'is_title_case', 'has_numbers', 'has_symbols'
    ]

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pdf_path = os.path.join(INPUT_DIR, pdf_file)
        df = extract_features_from_pdf(pdf_path)
        if df.empty: continue

        preds = model.predict(df[features])
        df['level'] = preds

        # Title detection
        title_row = df[df['level'] == 'Title']
        if not title_row.empty:
            title = title_row.iloc[0]['text']
        else:
            top3 = df[df['page_number'] <= 3]
            top3 = top3.sort_values(by=['font_size', 'is_bold'], ascending=[False, False])
            title = top3.iloc[0]['text'] if not top3.empty else "No Title Found"

        # Heading detection
        headings = df[df['level'].str.startswith('H')][['text', 'level', 'page_number']]
        headings = headings.drop_duplicates(subset=['text', 'level', 'page_number'])

        output = {
            "title": title,
            "headings": headings.to_dict(orient="records")
        }

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, pdf_file.replace(".pdf", ".json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    process_all_pdfs()
