#!/usr/bin/env python3
"""
src/train.py (Best and Final Version)
Trains a high-performance ensemble model using an enhanced feature set
and provides a full evaluation report to verify accuracy.
"""

import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import compute_sample_weight
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# Centralized Configuration
CONFIG = {
    "TEST_SIZE": 0.2, "RANDOM_STATE": 42,
    "TFIDF_PARAMS": {
        "max_features": 50000, "min_df": 2, "max_df": 0.95, "ngram_range": (1, 2)
    },
    "XGB1_PARAMS": {
        "objective": "multi:softprob", "tree_method": "gpu_hist",
        "n_estimators": 800, "max_depth": 16, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 1,
        "eval_metric": "mlogloss", "use_label_encoder": False, "n_jobs": 1,
        "verbosity": 0, "early_stopping_rounds": 75
    },
    "XGB2_PARAMS": {
        "objective": "multi:softprob", "tree_method": "gpu_hist",
        "n_estimators": 600, "max_depth": 12, "learning_rate": 0.07,
        "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 5, "gamma": 2,
        "eval_metric": "mlogloss", "use_label_encoder": False, "n_jobs": 1,
        "verbosity": 0, "early_stopping_rounds": 75
    }
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_CSV     = PROJECT_ROOT / "data" / "processed" / "labeled_data.csv"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"Loading data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    df["text"] = df["text"].fillna("")
    df["font_name"] = df["font_name"].fillna("__MISSING__")

    y_raw = df.pop("label")
    le = LabelEncoder().fit(y_raw)
    y  = le.transform(y_raw)
    print("Label distribution:\n", pd.Series(y_raw).value_counts(normalize=True).round(3), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=CONFIG["TEST_SIZE"], stratify=y, random_state=CONFIG["RANDOM_STATE"]
    )
    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

    # IMPROVEMENT: Add all relevant numeric features for training
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

    xgb1 = XGBClassifier(**CONFIG["XGB1_PARAMS"])
    xgb2 = XGBClassifier(**CONFIG["XGB2_PARAMS"])
    estimators = [xgb1, xgb2]
    fit_params = {
        "X": X_train_processed, "y": y_train,
        "sample_weight": compute_sample_weight(class_weight="balanced", y=y_train),
        "eval_set": [(X_test_processed, y_test)], "verbose": False
    }

    print(f"🚀 Training {len(estimators)} models in parallel...")
    tasks = [joblib.delayed(estimator.fit)(**fit_params) for estimator in estimators]
    trained_estimators = joblib.Parallel(n_jobs=len(estimators))(
        tqdm(tasks, desc="Training Ensemble Models", total=len(tasks))
    )
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
    y_pred = le.transform(y_pred_str)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("Generating confusion matrix...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, labels=le.transform(le.classes_),
        display_labels=le.classes_, xticks_rotation="vertical",
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
