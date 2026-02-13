import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = os.path.join("data", "features", "training_features.csv")
MODEL_PATH = os.path.join("models")
os.makedirs(MODEL_PATH, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["failure_within_30_days", "asset_id"])
    y = df["failure_within_30_days"]
    return X, y


def build_preprocessor(X):
    categorical_features = ["asset_type"]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )
    return preprocessor


def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n📊 {name} Evaluation")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    joblib.dump(model, f"{MODEL_PATH}/{name.lower().replace(' ', '_')}.joblib")
    print(f"✅ Model saved: models/{name.lower().replace(' ', '_')}.joblib")


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    # Logistic Regression (Baseline)
    lr_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    train_and_evaluate(
        lr_pipeline, X_train, X_test, y_train, y_test, "Logistic Regression"
    )

    # Random Forest (Production Model)
    rf_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ))
    ])

    train_and_evaluate(
        rf_pipeline, X_train, X_test, y_train, y_test, "Random Forest"
    )


if __name__ == "__main__":
    main()
