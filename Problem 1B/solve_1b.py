import json
import re
from pathlib import Path
from datetime import datetime
import fitz  
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import joblib
from collections import Counter
import warnings
import spacy 

warnings.filterwarnings("ignore")

RETRIEVER_MODEL = 'all-mpnet-base-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
TOP_N_SECTIONS = 5
CANDIDATES_TO_RERANK = 30 

INPUT_DIR = Path("./input")
OUTPUT_DIR = Path("./output")
MODELS_DIR = Path("./models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")

RETRIEVER_MODEL_PATH = MODELS_DIR / RETRIEVER_MODEL.replace('/', '_')
RERANKER_MODEL_PATH = MODELS_DIR / RERANKER_MODEL.replace('/', '_')

def get_robust_body_size(sizes):
    if not sizes: return 12.0
    q1, q3 = pd.Series(sizes).quantile([0.25, 0.75])
    core_sizes = [s for s in sizes if q1 <= s <= q3]
    return Counter(core_sizes or sizes).most_common(1)[0][0]

def post_process(df):
    FALSE_POS_PATTERNS = [r"table of contents", r"list of figures", r"references", r"index", r"glossary"]
    MAX_WORDS_HEADING = 30
    MIN_WORDS_HEADING = 2
    
    GENERIC_HEADINGS = {"introduction", "conclusion", "abstract", "foreword", "summary", "appendix", "background", "methodology"}

    if "confidence" not in df.columns:
        df["level"] = "Body"
        return df

    CONF_THRESH = {"Title": 0.6, "H1": 0.6, "H2": 0.65, "H3": 0.7, "Body": 0.3}
    for lvl, th in CONF_THRESH.items():
        df.loc[(df.level == lvl) & (df.confidence < th), "level"] = "Body"

    for pat in FALSE_POS_PATTERNS:
        df.loc[df.text.str.contains(pat, case=False, na=False), "level"] = "Body"
    
    df.loc[(df.level != "Body") & (df.word_count > MAX_WORDS_HEADING), "level"] = "Body"
    df.loc[(df.level != "Body") & (df.word_count < MIN_WORDS_HEADING) & (~df.text.str.contains(r'^\d+')), "level"] = "Body"
    
    df.loc[df['text'].str.lower().isin(GENERIC_HEADINGS), 'level'] = 'Body'
    
    df.loc[(df.level != "Body") & (df.text.str.isdigit()), "level"] = "Body"
    df.loc[(df.level != "Body") & (df.text.str.endswith(".")), "level"] = "Body"
    
    return df

def extract_features(pdf_path):
    rows = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return pd.DataFrame()

    for page_num, page in enumerate(doc, 1):
        blocks = [b for b in page.get_text("dict").get("blocks", []) if b.get("type") == 0 and "lines" in b]
        lines = []
        for b in blocks:
            for l in b.get("lines", []):
                spans = l.get("spans", [])
                if not spans: continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text: continue
                
                font_sizes = [s["size"] for s in spans]
                font_names = [s["font"] for s in spans]
                size = Counter(font_sizes).most_common(1)[0][0]
                font = Counter(font_names).most_common(1)[0][0]
                
                lines.append({"text": text, "size": size, "font": font, "bbox": l["bbox"], "spans": spans})

        if not lines: continue
        body_size = get_robust_body_size([l["size"] for l in lines])

        for line in lines:
            text = line["text"]
            spans = line["spans"]
            is_bold = any(("bold" in s["font"].lower() or s.get("flags", 0) & 16) for s in spans)
            
            rows.append({
                "text": text, "font_size": line["size"], "font_name": line["font"], "is_bold": int(is_bold),
                "page_number": page_num, "word_count": len(text.split()), "char_count": len(text),
                "relative_font_size": round(line["size"] / body_size, 3) if body_size > 0 else 0,
                "is_all_caps": int(text.isupper() and len(text) > 1),
                "starts_with_number": int(bool(re.match(r"^\d+", text))),
                "x_position": line["bbox"][0], "y_position": line["bbox"][1]
            })
    return pd.DataFrame(rows)

def parse_documents_with_ml_model(pdf_path, pipeline, label_encoder):
    df = extract_features(pdf_path)
    if df.empty: return []

    for col in pipeline.feature_names_in_:
        if col not in df.columns: df[col] = 0
    df_features = df[pipeline.feature_names_in_]

    proba = pipeline.predict_proba(df_features)
    df["level"] = label_encoder.inverse_transform(np.argmax(proba, axis=1))
    df["confidence"] = np.max(proba, axis=1)
    df = post_process(df)

    sections = []
    current_heading, current_text, current_page = "", "", 1
    
    for _, row in df.iterrows():
        is_heading = row.level in ("Title", "H1", "H2", "H3")
        is_recipe_part = row.text.lower().strip().startswith(('ingredients', 'instructions', 'method'))

        if is_heading and not is_recipe_part:
            if current_heading and current_text.strip():
                sections.append({
                    "document": pdf_path.name, "section_title": current_heading,
                    "page_number": current_page, "content": current_text.strip()
                })
            current_heading, current_text, current_page = row.text, "", row.page_number
        else:
            if not current_heading:
                 current_heading, current_page = row.text, row.page_number
            else:
                 current_text += " " + row.text

    if current_heading and current_text.strip():
        sections.append({
            "document": pdf_path.name, "section_title": current_heading,
            "page_number": current_page, "content": current_text.strip()
        })
    return sections

def extract_semantic_keywords(task_description):
    doc = NLP(task_description.lower())
    keywords = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'ADJ', 'PROPN', 'VERB']
    ]
    return list(set(keywords))

def generate_semantic_query(persona, jbtb):
    keywords = extract_semantic_keywords(jbtb)
    keyword_str = ", ".join(keywords[:5])
    
    query = f"As a {persona}, I need to {jbtb}. "
    query += f"Focus on sections that address: {keyword_str}. "
    query += "Provide practical, actionable information relevant to this specific task."
    
    return query

def semantic_filter_sections(sections_df, requirements):
    if sections_df.empty: return sections_df
    
    retriever = SentenceTransformer(str(RETRIEVER_MODEL_PATH))
    requirements_text = " ".join(requirements[:10])
    
    section_texts = sections_df['section_title'] + ". " + sections_df['content'].str[:200]
    section_embeddings = retriever.encode(section_texts.tolist(), show_progress_bar=False)
    requirement_embedding = retriever.encode(requirements_text)
    
    semantic_scores = torch.nn.functional.cosine_similarity(
        torch.tensor(section_embeddings), torch.tensor(requirement_embedding).unsqueeze(0), dim=1
    ).tolist()
    
    sections_df['semantic_score'] = semantic_scores
    return sections_df

def enforce_document_diversity(top_df, max_per_doc=2):
    final = []
    seen_docs = Counter()
    for _, row in top_df.iterrows():
        if seen_docs[row['document']] < max_per_doc:
            final.append(row)
            seen_docs[row['document']] += 1
        if len(final) == TOP_N_SECTIONS:
            break
    return pd.DataFrame(final)

def find_top_sections(sections_df, query):
    if sections_df.empty:
        print("Warning: No sections available for ranking.")
        return pd.DataFrame()

    retriever = SentenceTransformer(str(RETRIEVER_MODEL_PATH))
    sections_df['combined'] = sections_df.apply(lambda r: f"Title: {r['section_title']}. Content: {r['content'][:500]}", axis=1)
    
    section_embeddings = retriever.encode(sections_df['combined'].tolist(), show_progress_bar=False, normalize_embeddings=True)
    query_embedding = retriever.encode(query, normalize_embeddings=True)
    
    sections_df['retriever_score'] = torch.nn.functional.cosine_similarity(
        torch.tensor(section_embeddings), torch.tensor(query_embedding).unsqueeze(0), dim=1
    ).tolist()

    top_candidates = sections_df.sort_values(by='retriever_score', ascending=False).head(CANDIDATES_TO_RERANK)
    if top_candidates.empty:
        return pd.DataFrame()

    reranker = CrossEncoder(str(RERANKER_MODEL_PATH))
    rerank_pairs = [[query, row['combined']] for _, row in top_candidates.iterrows()]
    rerank_scores = reranker.predict(rerank_pairs, show_progress_bar=False)
    top_candidates['relevance_score'] = rerank_scores
    
    top_candidates['final_score'] = 0.8 * top_candidates['relevance_score'] + 0.2 * top_candidates['retriever_score']

    top_final = top_candidates.sort_values(by='final_score', ascending=False)
    return enforce_document_diversity(top_final)

def generate_refined_subsections(top_sections_df):
    if top_sections_df.empty: return []
    return [
        {"document": row.document, "refined_text": row.content, "page_number": int(row.page_number)}
        for _, row in top_sections_df.iterrows()
    ]

def main(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    persona = data.get('persona', {}).get('role', 'User')
    jbtb = data.get('job_to_be_done', {}).get('task', '')
    if not jbtb:
        print("Error: 'job_to_be_done' task is missing from input JSON.")
        return

    query = generate_semantic_query(persona, jbtb)
    print(f"Generated Query: {query}")
    
    requirements = extract_semantic_keywords(jbtb)

    try:
        pipeline = joblib.load(MODELS_DIR / "ensemble_pipeline.pkl")
        label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    except FileNotFoundError:
        print(f"Error: ML models not found in {MODELS_DIR}. Please ensure they are present.")
        return

    all_sections = []
    for doc in data.get('documents', []):
        path = INPUT_DIR / "pdfs" / doc['filename']
        if path.exists():
            all_sections.extend(parse_documents_with_ml_model(path, pipeline, label_encoder))
        else:
            print(f"Warning: Document not found at {path}")

    if not all_sections:
        print("Error: No sections could be extracted from any documents.")
        return

    df = pd.DataFrame(all_sections)
    
    df = semantic_filter_sections(df, requirements)
    
    top_sections = find_top_sections(df, query)

    extracted_sections_output = [
        {"document": row.document, "section_title": row.section_title, "importance_rank": i + 1, "page_number": int(row.page_number)}
        for i, (_, row) in enumerate(top_sections.head(TOP_N_SECTIONS).iterrows())
    ]

    subsection_output = generate_refined_subsections(top_sections)

    output = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in data.get('documents', [])],
            "persona": persona, "job_to_be_done": jbtb,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections_output,
        "subsection_analysis": subsection_output
    }

    output_file = OUTPUT_DIR / "challenge1b_output_final.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"✅ Final generalized output saved to {output_file}")

if __name__ == "__main__":
    input_path = INPUT_DIR / "challenge1b_input.json"
    if input_path.exists():
        main(input_path)
    else:
        print(f"❌ Input JSON not found at {input_path.resolve()}. Please place it in the {INPUT_DIR} directory.")
