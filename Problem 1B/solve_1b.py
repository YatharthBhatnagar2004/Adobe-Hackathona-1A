#!/usr/bin/env python3
"""
FINAL ACCURACY-FOCUSED Solution for Adobe India Hackathon - Challenge 1B

This version combines a robust direct-parsing method using 1A models with an
advanced Retriever-Re-ranker architecture and fully dynamic query generation.

To run this code, you need to install the following libraries:
pip install PyMuPDF sentence-transformers==2.2.2 scikit-learn==1.4.2 numpy pandas torch transformers
"""
import json
import re
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import joblib
import warnings
from collections import Counter

warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator.*")


# --- CONFIGURATION (WITH UPGRADED MODELS) ---
RETRIEVER_MODEL = 'all-mpnet-base-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

TOP_N_SECTIONS = 5
CANDIDATES_TO_RERANK = 25

# --- PATHS ---
INPUT_DIR = Path("./input_1b")
OUTPUT_DIR = Path("./output")
MODELS_1A_DIR = Path("models")
MODELS_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVER_MODEL_PATH = MODELS_DIR / RETRIEVER_MODEL
RERANKER_MODEL_PATH = MODELS_DIR / RERANKER_MODEL.replace("/", "_")


# --- 1. ROBUST DOCUMENT PARSING (Full implementation) ---
# (Functions 'get_robust_body_size', 'post_process', 'extract_features', 
# and 'parse_documents_with_ml_model' are omitted for brevity but should be included here
# from the previous version.)
CONF_THRESHOLDS = {"Title": 0.6, "H1": 0.6, "H2": 0.65, "H3": 0.7, "Body": 0.3}
FALSE_POS_PATTERNS = [r"^(table\s+of\s+)?contents?$", r"list\s+of\s+(figures|tables)", r"figure\s+\d", r"box\s+\d", r"table\s+\d", r"revision\s+history", r"references?$", r"bibliography", r"acknowledgments?", r"appendix\s+[A-Z0-9]", r"glossary", r"annex\s+[A-Z0-9]"]
MAX_HEADING_WORDS = 25

def get_robust_body_size(sizes: list[float], percentile_range=(25, 75)) -> float:
    if not sizes: return 12.0
    lower, upper = pd.Series(sizes).quantile([p/100 for p in percentile_range])
    plausible = [s for s in sizes if lower <= s <= upper]
    return Counter(plausible or sizes).most_common(1)[0][0]

def post_process(df: pd.DataFrame) -> pd.DataFrame:
    if "confidence" not in df.columns:
        df["level"] = "Body"
        return df
    for lvl, th in CONF_THRESHOLDS.items():
        df.loc[(df["level"] == lvl) & (df["confidence"] < th), "level"] = "Body"
    for pat in FALSE_POS_PATTERNS:
        df.loc[df["text"].str.contains(pat, regex=True, na=False, case=False), "level"] = "Body"
    df.loc[(df["level"] != "Body") & (df["word_count"] > MAX_HEADING_WORDS), "level"] = "Body"
    df.loc[(df["level"] != "Body") & (df["text"].str.isdigit()) & (df["text"].str.len() < 4), "level"] = "Body"
    df.loc[(df["level"] != "Body") & (df["text"].str.endswith('.')), "level"] = "Body"
    
    highest_seen_level = 0
    for i, row in df.iterrows():
        if row["level"].startswith("H"):
            current_level = int(row["level"][1:])
            if current_level > highest_seen_level + 1:
                df.at[i, "level"] = f"H{min(current_level, highest_seen_level + 1)}"
            highest_seen_level = int(df.at[i, "level"][1:])
        elif row['level'] == "Title":
            highest_seen_level = 0
    is_dupe = (df["text"].str.lower() == df["text"].str.lower().shift()) & \
              (df["page_number"] == df["page_number"].shift()) & (df["level"] != "Body")
    return df[~is_dupe].copy()

def extract_features(pdf_path: Path) -> pd.DataFrame:
    rows = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return pd.DataFrame()
        
    for page_num, page in enumerate(doc, 1):
        blocks = sorted([b for b in page.get_text("dict")["blocks"] if b.get("type") == 0 and "lines" in b], key=lambda b: b["bbox"][1])
        if not blocks: continue
            
        all_lines_on_page = []
        for b in blocks:
            for l in b["lines"]:
                text = "".join(s["text"] for s in l["spans"]).strip()
                text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
                if not text: continue
                span_sizes = [s["size"] for s in l["spans"]]
                line_size = Counter(span_sizes).most_common(1)[0][0] if span_sizes else 0
                all_lines_on_page.append({"text": text, "size": line_size, "bbox": l["bbox"], "spans": l["spans"]})

        if not all_lines_on_page: continue
            
        page_font_sizes = [line["size"] for line in all_lines_on_page if line["size"] > 4]
        body_font_size = get_robust_body_size(page_font_sizes)

        for i, line in enumerate(all_lines_on_page):
            text, font_size, spans = line["text"], line["size"], line["spans"]
            font_name = Counter(s["font"] for s in spans).most_common(1)[0][0] if spans else "N/A"
            is_bold = any(("bold" in s["font"].lower() or s["flags"] & 16) for s in spans)

            prev_line = all_lines_on_page[i - 1] if i > 0 else {"bbox": (0, 0, 0, 0), "size": font_size}
            font_size_diff_prev = font_size - prev_line["size"]
            is_prev_line_blank = int((line["bbox"][1] - prev_line["bbox"][3]) > (prev_line["size"] * 1.5))

            next_line = all_lines_on_page[i + 1] if i < len(all_lines_on_page) - 1 else {"bbox": (0, page.rect.height, 0, page.rect.height), "size": font_size}
            font_size_diff_next = font_size - next_line["size"]
            space_after = next_line["bbox"][1] - line["bbox"][3]
            is_next_line_blank = int(space_after > (font_size * 1.5))

            rows.append({
                "text": text, "font_size": font_size, "font_name": font_name, "is_bold": int(is_bold), "page_number": page_num,
                "word_count": len(text.split()),
                "relative_font_size": round(font_size / body_font_size, 3) if body_font_size else 0,
                "char_count": len(text), "is_all_caps": int(text.isupper() and len(text) > 1),
                "starts_with_number": int(bool(re.match(r"^\s*(\d+(\.\d+)*)\b", text))),
                "font_size_diff_prev": font_size_diff_prev, "is_prev_line_blank": is_prev_line_blank,
                "font_size_diff_next": font_size_diff_next, "is_next_line_blank": is_next_line_blank,
                "x_position": line["bbox"][0], "y_position": line["bbox"][1]
            })
    return pd.DataFrame(rows)

def parse_documents_with_ml_model(pdf_path: Path, pipeline, label_encoder) -> list:
    features_df = extract_features(pdf_path)
    if features_df.empty: return []

    required_cols = pipeline.feature_names_in_
    for col in required_cols:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[required_cols]

    proba = pipeline.predict_proba(features_df)
    features_df["level"] = label_encoder.inverse_transform(np.argmax(proba, axis=1))
    features_df["confidence"] = np.max(proba, axis=1)
    processed_df = post_process(features_df)

    chunks, current_heading, current_text, current_page = [], "Introduction", "", 1
    for _, row in processed_df.iterrows():
        is_heading = (row['level'].startswith('H') or row['level'] == 'Title')
        is_list_item = row['text'].strip().startswith(('•', '*', '-', '●'))
        
        if is_heading and not is_list_item:
            if current_heading and current_text.strip():
                chunks.append({"document": pdf_path.name, "page_number": current_page, "section_title": current_heading, "content": current_text.strip()})
            current_heading, current_page, current_text = row['text'], row['page_number'], ""
        else:
            current_text += " " + row['text']
            
    if current_heading and current_text.strip():
        chunks.append({"document": pdf_path.name, "page_number": current_page, "section_title": current_heading, "content": current_text.strip()})
    return chunks

# --- 2. ENHANCED QUERY GENERATION ---
def generate_enhanced_query(persona: str, jbtb: str) -> str:
    """
    Generates a descriptive, context-rich query for any persona and job-to-be-done.
    This version is generic, robust, and adds complexity by including context, intent, and open-ended prompts.
    """
    base_query = f"You are a {persona}. Your goal is to: '{jbtb}'. "
    base_query += (
        "Provide detailed, relevant, and actionable information. "
        "Include practical tips, important considerations, and potential challenges. "
        "Highlight both opportunities and risks. "
        "If applicable, suggest resources, best practices, and step-by-step guidance. "
        "Avoid generic or unrelated content. "
        "If the task is ambiguous, clarify assumptions and offer multiple perspectives. "
        "Be concise but thorough. "
    )
    return base_query

# --- 3. ADVANCED RELEVANCE SCORING (Retriever + Re-ranker) ---
def find_top_sections(sections_df: pd.DataFrame, query: str) -> pd.DataFrame:
    if sections_df.empty:
        return pd.DataFrame()

    print("Stage 1: Retrieving initial candidates...")
    retriever = SentenceTransformer(str(RETRIEVER_MODEL_PATH))
    section_embeddings = retriever.encode(sections_df['section_title'].tolist(), show_progress_bar=True)
    query_embedding = retriever.encode(query)
    
    sections_df['retriever_score'] = torch.nn.functional.cosine_similarity(
        torch.tensor(section_embeddings), torch.tensor(query_embedding).unsqueeze(0), dim=1
    ).tolist()
    
    top_candidates = sections_df.sort_values(by='retriever_score', ascending=False).head(CANDIDATES_TO_RERANK)

    print("Stage 2: Re-ranking candidates...")
    reranker = CrossEncoder(str(RERANKER_MODEL_PATH))
    rerank_pairs = [[query, row['section_title'] + ". " + row['content']] for _, row in top_candidates.iterrows()]

    if not rerank_pairs: return pd.DataFrame()
        
    rerank_scores = reranker.predict(rerank_pairs, show_progress_bar=True)
    top_candidates['relevance_score'] = rerank_scores
    
    return top_candidates.sort_values(by='relevance_score', ascending=False).head(TOP_N_SECTIONS)


# --- 4. GENERALIZED SUBSECTION ANALYSIS (Extractive) ---
def generate_refined_subsections(top_sections_df: pd.DataFrame, query: str) -> list[dict]:
    print("Stage 3: Generating refined subsections...")
    if top_sections_df.empty:
        return []

    retriever = SentenceTransformer(str(RETRIEVER_MODEL_PATH))
    query_embedding = retriever.encode(query)
    analysis_results = []

    for _, row in top_sections_df.iterrows():
        sentences = [s.strip() for s in row['content'].split('.') if len(s.strip().split()) > 5]
        if not sentences: continue
            
        sentence_embeddings = retriever.encode(sentences, show_progress_bar=False)
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(sentence_embeddings), torch.tensor(query_embedding).unsqueeze(0), dim=1
        )
        
        top_indices = torch.topk(similarities, k=min(3, len(sentences))).indices
        refined_text = " ".join([sentences[i] for i in sorted(top_indices.tolist())])
        
        analysis_results.append({
            "document": row['document'],
            "refined_text": refined_text.strip(),
            "page_number": int(row['page_number'])
        })
    return analysis_results


# --- 5. MAIN ORCHESTRATOR ---
def main(input_json_path: Path):
    print(f"Processing input file: {input_json_path.name}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona = input_data['persona']['role']
    jbtb = input_data['job_to_be_done']['task']
    
    # Use the new enhanced query for all stages
    enhanced_query = generate_enhanced_query(persona, jbtb)
    print(f"Using Enhanced Query: '{enhanced_query}'")
    
    pipeline = joblib.load(MODELS_1A_DIR / "ensemble_pipeline.pkl")
    label_encoder = joblib.load(MODELS_1A_DIR / "label_encoder.pkl")

    all_sections = []
    pdf_files = [doc['filename'] for doc in input_data['documents']]
    
    # NEW: Recursively search for PDFs in input_1b and its subdirectories
    all_pdf_paths = list(INPUT_DIR.rglob('*.pdf'))
    all_pdf_names = {p.name: p for p in all_pdf_paths}
    
    for pdf_file in pdf_files:
        # Try to find the PDF by name in any subdirectory
        pdf_path = all_pdf_names.get(pdf_file)
        if pdf_path and pdf_path.exists():
            all_sections.extend(parse_documents_with_ml_model(pdf_path, pipeline, label_encoder))
        else:
            print(f"⚠ Warning: Missing PDF for {pdf_file}. Skipping.")
    
    sections_df = pd.DataFrame(all_sections)
    if sections_df.empty:
        print("❌ Error: No sections could be parsed from the documents. Aborting.")
        return

    top_sections_df = find_top_sections(sections_df, enhanced_query)

    extracted_sections_output = [{
        "document": row['document'],
        "section_title": row['section_title'],
        "importance_rank": i,
        "page_number": int(row['page_number'])
    } for i, (_, row) in enumerate(top_sections_df.iterrows(), 1)]

    subsection_analysis_output = generate_refined_subsections(top_sections_df, enhanced_query)

    final_output = {
        "metadata": { "input_documents": pdf_files, "persona": persona, "job_to_be_done": jbtb, "processing_timestamp": datetime.now().isoformat() },
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_analysis_output
    }

    output_filename = "challenge1b_output.json"
    output_path = OUTPUT_DIR / output_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Successfully processed and saved final output to: {output_path}")


if __name__ == "__main__":
    input_file = Path("input_1b/challenge1b_input.json")
    if not input_file.exists():
        print(f"❌ Error: Input JSON file not found at {input_file}")
    else:
        main(input_file)