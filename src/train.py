import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

PROCESSED_DATA_PATH = "data/processed/labeled_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "classifier_v2.joblib")

def train_model():
    print("Loading full labeled data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    df.dropna(subset=['label'], inplace=True)
    print(f"Training on {len(df)} samples...")

    features = [
        'font_size', 'is_bold', 'x_position', 'y_position', 'word_count', 'is_all_caps',
        'is_centered', 'is_title_case', 'has_numbers', 'has_symbols'
    ]
    for col in features:
        if col not in df.columns:
            df[col] = 0

    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    print("\n--- Evaluation ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
