"""Score new startups using the saved primary Logistic Regression pipeline."""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL = BASE / "startuplens_primary_logistic_regression.joblib"
THRESHOLD = 0.50

def main():
    parser = argparse.ArgumentParser(description="Generate StartupLens failure-risk scores.")
    parser.add_argument("--input", default="new_startups_template.csv", help="Input CSV with startup feature columns.")
    parser.add_argument("--output", default="startup_predictions.csv", help="Output CSV path.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Decision threshold from 0 to 1.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    model = joblib.load(MODEL)
    incoming = pd.read_csv(Path(args.input))
    feature_names = list(model.named_steps["preprocess"].feature_names_in_)
    X = incoming.reindex(columns=feature_names)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)
    result = incoming.copy()
    result["failure_score_uncalibrated"] = probabilities
    result["risk_classification"] = np.where(predictions == 1, "Higher risk flag", "Lower risk flag")
    result["decision_threshold"] = args.threshold
    result.to_csv(args.output, index=False)
    print(f"Scored {len(result)} startup(s).")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Saved: {Path(args.output).resolve()}")
    print("Scores are model outputs, not calibrated probabilities or guarantees of future failure.")

if __name__ == "__main__":
    main()
