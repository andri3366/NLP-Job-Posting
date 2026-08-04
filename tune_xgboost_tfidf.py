import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.feature_extractors import HybridTFIDFExtractor, TFIDFTextExtractor
from src.preprocess import clean_text


RANDOM_STATE = 42
MODEL_DIR = Path(__file__).resolve().parent / "model"
DATA_PATH = Path(__file__).resolve().parent / "data" / "fake_job_postings_cleaned.csv"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "telecommuting",
        "has_company_logo",
        "has_questions",
        "employment_type",
        "required_experience",
        "required_education",
        "country",
        "fraudulent",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    fill_text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    fill_cat_cols = ["employment_type", "required_experience", "required_education", "country"]
    fill_binary_cols = ["telecommuting", "has_company_logo", "has_questions"]

    df[fill_text_cols] = df[fill_text_cols].fillna("")
    df[fill_cat_cols] = df[fill_cat_cols].fillna("Missing")
    df[fill_binary_cols] = df[fill_binary_cols].fillna(0)

    df["combined_text"] = (
        df["title"] + " "
        + df["company_profile"] + " "
        + df["description"] + " "
        + df["requirements"] + " "
        + df["benefits"]
    )

    df["combined_text"] = df["combined_text"].apply(clean_text)
    return df


def build_param_distributions() -> dict:
    return {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.3],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }


def build_base_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


def tune_model(X, y, label: str) -> RandomizedSearchCV:
    n_iter = int(os.getenv("XGB_TUNE_ITER", "20"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    scale_pos_weight = (negatives / positives) if positives else 1.0

    search = RandomizedSearchCV(
        estimator=build_base_model(scale_pos_weight),
        param_distributions=build_param_distributions(),
        n_iter=n_iter,
        scoring="f1",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True,
    )

    print(f"\nTuning {label} model with {n_iter} random search iterations...")
    search.fit(X, y)

    print(f"Best {label} CV F1: {search.best_score_:.4f}")
    print(f"Best {label} params: {search.best_params_}")

    train_predictions = search.best_estimator_.predict(X)
    train_f1 = f1_score(y, train_predictions, zero_division=0)
    print(f"Best {label} train F1 (for sanity check): {train_f1:.4f}")

    return search


def save_full_model_artifacts(search: RandomizedSearchCV, extractor: HybridTFIDFExtractor) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(search.best_estimator_, MODEL_DIR / "tune_best_model.pkl")
    joblib.dump(extractor.vectorizer, MODEL_DIR / "tune_best_extractor_vectorizer.pkl")
    joblib.dump(extractor.cat_columns, MODEL_DIR / "tune_cat_features.pkl")
    joblib.dump({"extractor": "tfidf"}, MODEL_DIR / "tune_feature_info.pkl")

    return {
        "best_cv_f1": float(search.best_score_),
        "best_params": search.best_params_,
    }


def save_text_model_artifacts(search: RandomizedSearchCV, extractor: TFIDFTextExtractor) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(search.best_estimator_, MODEL_DIR / "tune_text_best_model.pkl")
    joblib.dump(extractor, MODEL_DIR / "tune_text_best_extractor.pkl")

    return {
        "best_cv_f1": float(search.best_score_),
        "best_params": search.best_params_,
    }


def main() -> None:
    df = load_dataset()
    y = df["fraudulent"]

    print("Preparing full model TF-IDF + structured features...")
    full_extractor = HybridTFIDFExtractor()
    X_full = full_extractor.fit_transform(df)

    print("Preparing text-only TF-IDF features...")
    text_extractor = TFIDFTextExtractor(max_features=5000)
    X_text = text_extractor.fit_transform(df["combined_text"])

    full_search = tune_model(X_full, y, "full")
    text_search = tune_model(X_text, y, "text")

    full_summary = save_full_model_artifacts(full_search, full_extractor)
    text_summary = save_text_model_artifacts(text_search, text_extractor)

    results = {
        "full_model": full_summary,
        "text_model": text_summary,
    }

    with open(MODEL_DIR / "xgb_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved only the best artifacts for each model to the model directory:")
    print("- tune_best_model.pkl")
    print("- tune_best_extractor_vectorizer.pkl")
    print("- tune_cat_features.pkl")
    print("- tune_feature_info.pkl")
    print("- tune_text_best_model.pkl")
    print("- tune_text_best_extractor.pkl")
    print("- tune_xgb_tuning_results.json")


if __name__ == "__main__":
    main()
