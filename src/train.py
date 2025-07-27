#!/usr/bin/env python3
"""
src/train.py (FINAL VERSION - HIGH ACCURACY)
Trains a high-performance ensemble model using a balanced feature set.
"""

import warnings
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# Centralized Configuration
CONFIG = {
    "TEST_SIZE": 0.2, "RANDOM_STATE": 42,
    # MODIFIED: Increased max_features to create a larger, more powerful model
    "TFIDF_PARAMS": {
        "max_features": 200000,
        "min_df": 2, "max_df": 0.95, "ngram_range": (1, 2)
    },
    # MODIFIED: Slightly increased model complexity to leverage more features
    "XGB1_PARAMS": {
        "objective": "multi:softprob", "tree_method": "hist",
        "n_estimators": 800, "max_depth": 12,
        "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 1,
        "eval_metric": "mlogloss", "use_label_encoder": False, "n_jobs": 1,
        "verbosity": 0, "early_stopping_rounds": 75
    },
    "XGB2_PARAMS": {
        "objective": "multi:softprob", "tree_method": "hist",
        "n_estimators": 800, "max_depth": 12,
        "learning_rate": 0.07,
        "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 5, "gamma": 2,
        "eval_metric": "mlogloss", "use_label_encoder": False, "n_jobs": 1,
        "verbosity": 0, "early_stopping_rounds": 75
    }
}
# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "labeled_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def check_gpu_support() -> str:
    """Checks if XGBoost can use a GPU and returns the appropriate device string."""
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        print("✅ CUDA-enabled GPU detected. Using 'cuda'.")
        return "cuda"
    except (ImportError, cupy.cuda.runtime.CUDARuntimeError):
        print("⚠️ CUDA-enabled GPU not found. Falling back to 'cpu'.")
        return "cpu"

def main():
    """Main function to load data, train the ensemble model, and save artifacts."""
    print(f"Loading data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    df["text"] = df["text"].fillna("")
    df["font_name"] = df["font_name"].fillna("__MISSING__")

    y_raw = df.pop("label")
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    print("Label distribution:\n", pd.Series(y_raw).value_counts(normalize=True).round(3), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=CONFIG["TEST_SIZE"], stratify=y, random_state=CONFIG["RANDOM_STATE"]
    )
    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

    NUM_COLS = [
        'font_size', 'is_bold', 'is_standalone_line', 'x_position', 'y_position',
        'word_count', 'char_count', 'is_all_caps', 'relative_font_size',
        'space_before_norm', 'space_after_norm', 'starts_with_number'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), ["font_name"]),
            ("txt", TfidfVectorizer(**CONFIG["TFIDF_PARAMS"]), "text"),
        ],
        remainder="drop", verbose_feature_names_out=False, n_jobs=-1
    )

    print("\nPreprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    device = check_gpu_support()
    xgb1_params = {**CONFIG["XGB1_PARAMS"], "device": device}
    xgb2_params = {**CONFIG["XGB2_PARAMS"], "device": device}
    
    xgb1 = XGBClassifier(**xgb1_params)
    xgb2 = XGBClassifier(**xgb2_params)
    estimators = [xgb1, xgb2]
    
    fit_params = {
        "X": X_train_processed, "y": y_train,
        "sample_weight": compute_sample_weight(class_weight="balanced", y=y_train),
        "eval_set": [(X_test_processed, y_test)],
    }

    print(f"\n🚀 Training {len(estimators)} models sequentially on device: '{device}'...")
    trained_estimators = []
    for i, estimator in enumerate(estimators, 1):
        print(f"--- Training Model {i}/{len(estimators)} ---")
        estimator.fit(**fit_params, verbose=False)
        trained_estimators.append(estimator)
    print("Training complete.")

    print("Assembling final pipeline...")
    ensemble = VotingClassifier(
        estimators=[("xgb1", trained_estimators[0]), ("xgb2", trained_estimators[1])], voting="soft"
    )
    
    ensemble.le_ = le
    ensemble.estimators_ = trained_estimators
    ensemble.classes_ = le.classes_
    
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", ensemble)])

    print("\n--- Evaluation ---")
    y_pred_str = pipeline.predict(X_test)
    y_pred_encoded = le.transform(y_pred_str)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred_encoded, target_names=le.classes_, zero_division=0))

    print("Generating confusion matrix...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        le.inverse_transform(y_test), y_pred_str, ax=ax,
        labels=le.classes_, xticks_rotation="vertical",
        normalize='true', values_format=".1%"
    )
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    cm_path = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"📊 Confusion matrix saved to {cm_path.resolve()}")

    joblib.dump(pipeline, MODEL_DIR / "ensemble_pipeline.pkl")
    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    size_mb = (MODEL_DIR / "ensemble_pipeline.pkl").stat().st_size / (1024*1024)
    print(f"\n✅ Saved model (~{size_mb:.1f} MB) and artifacts to {MODEL_DIR.resolve()}")

if __name__ == "__main__":
    main()