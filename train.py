"""Train a Titanic survival classifier and save it with joblib.

Run:  python train.py
Output: model.joblib  (a scikit-learn Pipeline that expects the columns in FEATURES order)
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# IMPORTANT: the app must build its input row in exactly this column order.
FEATURES = ["pclass", "sex", "age", "fare"]
TARGET = "survived"

DATA_PATH = Path(__file__).parent / "data" / "titanic.csv"
MODEL_PATH = Path(__file__).parent / "model.joblib"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURES[:1] + ["sex", "age", "fare", TARGET]].copy()
    # Encode sex numerically: female = 1, male = 0 (same encoding used in app.py)
    df["sex"] = (df["sex"] == "female").astype(int)
    return df


def main() -> None:
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),  # some ages are missing
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, target_names=["died", "survived"]))

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.name}")
    print(f"Feature order: {FEATURES}")


if __name__ == "__main__":
    main()
