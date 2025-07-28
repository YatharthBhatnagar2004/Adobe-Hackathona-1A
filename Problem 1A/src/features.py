# src/features.py

import fitz  # PyMuPDF
import pandas as pd
import os

def extract_features_from_pdf(pdf_path: str) -> pd.DataFrame:
    """
    Extracts text blocks and their features from a given PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A pandas DataFrame where each row represents a text block
        and columns represent its features.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No such file: '{pdf_path}'")

    doc = fitz.open(pdf_path)
    blocks_data = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:  # 0 indicates a text block
                for line in block['lines']:
                    for span in line['spans']:
                        blocks_data.append({
                            'text': span['text'].strip(),
                            'font_size': round(span['size']),
                            'font_name': span['font'],
                            'is_bold': 'bold' in span['font'].lower(),
                            'x_position': round(span['bbox'][0]),
                            'page_number': page_num + 1,
                        })

    if not blocks_data:
        return pd.DataFrame()

    df = pd.DataFrame(blocks_data)
    df = df[df['text'] != ''] # Remove empty text blocks
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))

    return df

def get_pdf_page_count(pdf_path: str) -> int:
    """
    Returns the number of pages in the given PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No such file: '{pdf_path}'")
    doc = fitz.open(pdf_path)
    return doc.page_count