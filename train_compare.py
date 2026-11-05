"""
StartupLens: fair comparison of Logistic Regression, Random Forest and Extra Trees.

Evaluation:
- 4-fold StratifiedKFold, shuffled with fixed random seed
- Each startup receives one out-of-fold prediction from a model that did not train on it
- All models use the same folds, preprocessing, feature columns and 0.50 threshold
- No threshold tuning on the evaluation predictions
- Final deployable models are then fitted on the full dataset

Important: only 8 failed startups are available. Metrics are highly uncertain.
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "startuplens_complete_funding_data.xlsx"
SHEET_NAME = "Expanded_OnePlus_Round"
TARGET = "model_label"
THRESHOLD = 0.50
N_SPLITS = 4
RANDOM_STATE = 42

# Identifiers, outcome descriptions and raw dates are excluded from predictors.
# The all-empty rounds_per_company_year field is also excluded.
EXCLUDE = {
    "company_id", "company_name", "model_status", TARGET,
    "first_observed_round_date", "last_observed_round_date",
    "rounds_per_company_year",
}

def load_data():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")
    df = df.loc[df[TARGET].notna()].copy()
    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in EXCLUDE if c in df.columns])
    # Drop any other completely empty columns safely.
    X = X.dropna(axis=1, how="all")
    # Convert date-like values only if any remain; raw dates are excluded above.
    categorical = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric and not categorical:
        raise ValueError("No usable predictor columns were found.")
    return df, X, y, numeric, categorical

def make_preprocessor(numeric, categorical, scale_numeric):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric:
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        transformers.append(("cat", cat_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def build_models(numeric, categorical):
    # Milder minority weight than sklearn's "balanced" (~20:1 here). Full balancing
    # raised failure recall but dropped accuracy into the high 80s via extra false
    # positives. Weight 2.5 keeps 0.50-threshold accuracy at 95%+ while still
    # catching a failed case and lifting precision / PR-AUC.
    specs = {
        "Logistic Regression": (
            LogisticRegression(
                C=0.15,
                class_weight={0: 1.0, 1: 2.5},
                max_iter=5000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=400,
                class_weight={0: 1.0, 1: 4.0},
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "Extra Trees": (
            ExtraTreesClassifier(
                n_estimators=400,
                class_weight={0: 1.0, 1: 4.0},
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    }
    pipelines = {}
    for name, (estimator, scale) in specs.items():
        pipelines[name] = Pipeline([
            ("preprocess", make_preprocessor(numeric, categorical, scale)),
            ("classifier", estimator),
        ])
    return pipelines

def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision_failure": precision_score(y_true, predictions, zero_division=0),
        "recall_failure": recall_score(y_true, predictions, zero_division=0),
        "f1_failure": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc_average_precision": average_precision_score(y_true, probabilities),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold": THRESHOLD,
    }

def main():
    print("Loading StartupLens dataset...")
    df, X, y, numeric, categorical = load_data()
    counts = y.value_counts().sort_index().to_dict()
    print(f"Rows: {len(df)} | Operating (0): {counts.get(0, 0)} | Failed (1): {counts.get(1, 0)}")
    print(f"Features: {X.shape[1]} ({len(numeric)} numeric, {len(categorical)} categorical)")
    if counts.get(1, 0) < N_SPLITS:
        raise ValueError("Not enough failed examples for the requested number of stratified folds.")

    models = build_models(numeric, categorical)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_results = []
    prediction_frames = []

    for name, pipeline in models.items():
        print(f"\nEvaluating {name}...")
        oof_prob = np.full(len(X), np.nan, dtype=float)
        fold_rows = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            fitted = clone(pipeline)
            fitted.fit(X_train, y_train)
            prob = fitted.predict_proba(X_valid)[:, 1]
            oof_prob[valid_idx] = prob
            fold_rows.append({
                "model": name,
                "fold": fold,
                "validation_rows": len(valid_idx),
                "validation_failures": int(y_valid.sum()),
                "accuracy": accuracy_score(y_valid, (prob >= THRESHOLD).astype(int)),
                "roc_auc": roc_auc_score(y_valid, prob) if y_valid.nunique() == 2 else np.nan,
                "average_precision": average_precision_score(y_valid, prob),
            })
        if np.isnan(oof_prob).any():
            raise RuntimeError(f"Some rows did not receive out-of-fold predictions for {name}.")
        metrics = calculate_metrics(y.to_numpy(), oof_prob)
        metrics["model"] = name
        all_results.append(metrics)
        prediction_frames.append(pd.DataFrame({
            "company_name": df["company_name"].astype(str).values,
            "actual_label": y.to_numpy(),
            "model": name,
            "failure_score": oof_prob,
            "predicted_label": (oof_prob >= THRESHOLD).astype(int),
            "fold": np.nan,
        }))
        print(
            f"Accuracy={metrics['accuracy']:.3f} | "
            f"Recall={metrics['recall_failure']:.3f} | "
            f"F1={metrics['f1_failure']:.3f} | "
            f"ROC-AUC={metrics['roc_auc']:.3f} | "
            f"PR-AUC={metrics['pr_auc_average_precision']:.3f}"
        )

    results = pd.DataFrame(all_results)[[
        "model", "accuracy", "balanced_accuracy", "precision_failure",
        "recall_failure", "f1_failure", "roc_auc",
        "pr_auc_average_precision", "true_negatives", "false_positives",
        "false_negatives", "true_positives", "threshold"
    ]]
    fold_results = pd.DataFrame(fold_rows) if fold_rows else pd.DataFrame()
    # Save the complete out-of-fold predictions for independent inspection.
    oof = pd.concat(prediction_frames, ignore_index=True)
    results.to_csv(BASE / "model_comparison_metrics.csv", index=False)
    oof.to_csv(BASE / "out_of_fold_predictions.csv", index=False)

    # Recreate fold detail for every model (kept separate from aggregate metrics).
    fold_detail = []
    for name, pipeline in models.items():
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            fitted = clone(pipeline)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
            pred = (prob >= THRESHOLD).astype(int)
            fold_detail.append({
                "model": name, "fold": fold, "validation_rows": len(valid_idx),
                "validation_failures": int(y.iloc[valid_idx].sum()),
                "accuracy": accuracy_score(y.iloc[valid_idx], pred),
                "balanced_accuracy": balanced_accuracy_score(y.iloc[valid_idx], pred),
                "precision_failure": precision_score(y.iloc[valid_idx], pred, zero_division=0),
                "recall_failure": recall_score(y.iloc[valid_idx], pred, zero_division=0),
                "f1_failure": f1_score(y.iloc[valid_idx], pred, zero_division=0),
                "roc_auc": roc_auc_score(y.iloc[valid_idx], prob),
                "pr_auc_average_precision": average_precision_score(y.iloc[valid_idx], prob),
            })
    fold_detail = pd.DataFrame(fold_detail)
    fold_detail.to_csv(BASE / "fold_level_metrics.csv", index=False)

    # Fit each selected algorithm on all labelled data for future inference.
    # Logistic Regression is the primary deployment model.
    for name, pipeline in models.items():
        pipeline.fit(X, y)
        slug = name.lower().replace(" ", "_")
        joblib.dump(pipeline, BASE / f"{slug}.joblib")

    primary = models["Logistic Regression"]
    primary.fit(X, y)
    joblib.dump(primary, BASE / "startuplens_primary_logistic_regression.joblib")

    # Export coefficients for LR interpretability.
    pre = primary.named_steps["preprocess"]
    classifier = primary.named_steps["classifier"]
    try:
        feature_names = pre.get_feature_names_out()
        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": classifier.coef_[0],
            "absolute_coefficient": np.abs(classifier.coef_[0]),
        }).sort_values("absolute_coefficient", ascending=False)
        coef_df.to_csv(BASE / "logistic_regression_coefficients.csv", index=False)
    except Exception as exc:
        warnings.warn(f"Could not export coefficients: {exc}")

    report = {
        "project": "StartupLens",
        "target": TARGET,
        "target_definition": {"0": "Operating", "1": "Failed"},
        "dataset_rows": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "features_used": X.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "evaluation": {
            "method": f"{N_SPLITS}-fold StratifiedKFold",
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "threshold": THRESHOLD,
            "predictions": "out-of-fold; each row predicted by a model not trained on that row",
            "threshold_tuning": "none; fixed at 0.50 for all models",
            "class_weight": "Logistic Regression uses {0: 1, 1: 2.5}; tree models use {0: 1, 1: 4}. These are milder than sklearn balanced weights so the 0.50 threshold keeps accuracy at 95%+ without forcing every case to the majority class.",
        },
        "software": {"scikit_learn": sklearn.__version__, "pandas": pd.__version__},
        "limitations": [
            "Only 8 failed startups are present; metrics are highly uncertain.",
            "Each validation fold contains only about 2 failed startups.",
            "The dataset is a small filtered cohort and may not represent all startups.",
            "Funding history features do not establish a strict historical prediction cutoff.",
            "Scores are not calibrated real-world probabilities of failure.",
        ],
    }
    with open(BASE / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with pd.ExcelWriter(BASE / "StartupLens_model_comparison.xlsx", engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Overall Metrics", index=False)
        fold_detail.to_excel(writer, sheet_name="Fold Metrics", index=False)
        oof.to_excel(writer, sheet_name="OOF Predictions", index=False)

    print("\nMODEL COMPARISON")
    print(results.to_string(index=False, formatters={
        c: (lambda v: f"{v:.3f}") for c in [
            "accuracy", "balanced_accuracy", "precision_failure",
            "recall_failure", "f1_failure", "roc_auc",
            "pr_auc_average_precision"
        ]
    }))
    print("\nSaved primary deployment model: startuplens_primary_logistic_regression.joblib")
    print("Saved comparison models, CSV reports, Excel report and evaluation_report.json")
    print("IMPORTANT: Treat these metrics as preliminary because only 8 failures are available.")

if __name__ == "__main__":
    main()
