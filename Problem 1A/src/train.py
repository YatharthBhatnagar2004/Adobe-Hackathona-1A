import warnings
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = {
    "TEST_SIZE": 0.2, "RANDOM_STATE": 42,
    "TFIDF_PARAMS": {"max_features": 30000, "min_df": 2, "max_df": 0.95, "ngram_range": (1, 2)},
    "XGB1_PARAMS": {
        "objective": "multi:softprob", "tree_method": "hist", "n_estimators": 1500, "max_depth": 12,
        "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "mlogloss", "early_stopping_rounds": 50
    },
    "XGB2_PARAMS": {
        "objective": "multi:softprob", "tree_method": "hist", "n_estimators": 1000, "max_depth": 6,
        "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "mlogloss", "early_stopping_rounds": 50
    },
    "CATBOOST_PARAMS": {
        "iterations": 1200,  # n_estimators
        "learning_rate": 0.02,
        "depth": 10,
        "eval_metric": "MultiClass",
        "auto_class_weights": "Balanced",  
        "early_stopping_rounds": 50,
        "verbose": 0  
    },
    "LGBM_PARAMS": {
        "objective": "multiclass", "metric": "multi_logloss", "n_estimators": 1500,
        "learning_rate": 0.02, "num_leaves": 31, "max_depth": -1,
        "n_jobs": -1, "verbose": -1, "class_weight": "balanced"
    }
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "labeled_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def check_gpu_support() -> str:
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        print("CUDA-enabled GPU detected.")
        return "cuda"
    except (ImportError, cupy.cuda.runtime.CUDARuntimeError):
        print("CUDA-enabled GPU not found. Falling back to 'cpu'.")
        return "cpu"


def main():
    print(f"Loading data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    df[["text", "font_name"]] = df[["text", "font_name"]].fillna("")

    le = LabelEncoder().fit(df["label"])
    y = le.transform(df["label"])
    X = df.drop("label", axis=1)

    print("Label distribution:\n", pd.Series(le.inverse_transform(
        y)).value_counts(normalize=True).round(3), "\n")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["TEST_SIZE"], stratify=y, random_state=CONFIG["RANDOM_STATE"])
    print(
        f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

    NUM_COLS = ['font_size', 'is_bold', 'x_position', 'y_position', 'word_count', 'char_count', 'is_all_caps',
                'relative_font_size', 'starts_with_number', 'font_size_diff_prev', 'is_prev_line_blank',
                'font_size_diff_next', 'is_next_line_blank']

    preprocessor = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore",
         sparse_output=False), ["font_name"]),
        ("txt", TfidfVectorizer(**CONFIG["TFIDF_PARAMS"]), "text"),
    ], remainder="drop", n_jobs=-1)

    print("\nPreprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    device = check_gpu_support()
    xgb1 = XGBClassifier(
        **CONFIG["XGB1_PARAMS"], device=device, use_label_encoder=False, verbosity=0)
    xgb2 = XGBClassifier(
        **CONFIG["XGB2_PARAMS"], device=device, use_label_encoder=False, verbosity=0)
    lgbm = lgb.LGBMClassifier(**CONFIG["LGBM_PARAMS"])
    catboost = CatBoostClassifier(**CONFIG["CATBOOST_PARAMS"])
    print("\nTraining base models sequentially...")

    print("--- Training Model 1/2 (XGBoost) ---")
    xgb1.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)])
    print("--- Training Model 2/4 (XGBoost2) ---")
    xgb2.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)])

    print("--- Training Model 3/4 (LightGBM) ---")
    lgbm.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)],
             callbacks=[lgb.early_stopping(50, verbose=False)])
    print("Base model training complete.")
    print("--- Training Model 4/4 (CatBoost) ---")
    catboost.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)])

    base_estimators = [("xgb1", xgb1), ("xgb2", xgb2), ("lgbm", lgbm), ("catboost", catboost)]    
    meta_model = LogisticRegression(
        class_weight='balanced', solver='liblinear', max_iter=1000)

    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv='prefit',  
        stack_method='predict_proba',
        passthrough=True
    )

    print("\nTraining Stacking meta-model...")
    ensemble.fit(X_train_processed, y_train)
    print("Training complete.")

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", ensemble)])

    print("\n--- Evaluation ---")
    y_pred_encoded = pipeline.predict(X_test)
    y_pred_str = le.inverse_transform(y_pred_encoded)
    y_test_str = le.inverse_transform(y_test)

    print("Classification Report:\n", classification_report(
        y_test_str, y_pred_str, zero_division=0))

    print("Generating confusion matrix...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_test_str, y_pred_str, ax=ax, labels=le.classes_, xticks_rotation="vertical", normalize='true', values_format=".1%")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    cm_path = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path.resolve()}")

    joblib.dump(pipeline, MODEL_DIR / "ensemble_pipeline.pkl")
    joblib.dump(le, MODEL_DIR / "label_encoder.pkl")
    size_mb = (MODEL_DIR / "ensemble_pipeline.pkl").stat().st_size / (1024*1024)
    print(
        f"\nSaved model (~{size_mb:.1f} MB) and artifacts to {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
