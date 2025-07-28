from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

# --- Configuration ---
RETRIEVER_MODEL = 'all-mpnet-base-v2'
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
MODELS_DIR = Path("./models")

# --- Download Logic ---
if __name__ == "__main__":
    print(f"Downloading models to '{MODELS_DIR.resolve()}'...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download and save the retriever model
    print(f"Downloading retriever: {RETRIEVER_MODEL}...")
    retriever = SentenceTransformer(RETRIEVER_MODEL)
    retriever.save(str(MODELS_DIR / RETRIEVER_MODEL))

    # Download and save the re-ranker model
    print(f"Downloading re-ranker: {RERANKER_MODEL}...")
    reranker = CrossEncoder(RERANKER_MODEL)
    # Replace '/' for valid folder name
    reranker.save(str(MODELS_DIR / RERANKER_MODEL.replace("/", "_")))

    print("\nAll models have been successfully downloaded and saved locally.")
