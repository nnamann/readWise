"""Evaluate all recommendation variants using the corrected v7 protocol."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import OUTPUT_DIR
from src.data_loader import load_curated_dataset
from src.evaluation import EvaluationConfig, evaluate_models


def main() -> None:
    books, ratings = load_curated_dataset()
    results, details = evaluate_models(books, ratings, EvaluationConfig())
    details["model_revision"] = (
        "v7 corrected feature-fit scope; original content plus previous and revised hybrid; no holdout tuning"
    )
    details["historical_comparison_note"] = (
        "V5 and v6 are preserved. V7 keeps the same seed and holdout fraction, but fits TF-IDF vocabulary/IDF "
        "only on training-rated catalogue records and excludes held-out editions of works already rated in "
        "training (duplicate held-out editions count once). Therefore v7 scores are not directly comparable "
        "to v5/v6; model rows within v7 share the corrected protocol."
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "recommendation_evaluation_v7_results.csv", index=False)
    (OUTPUT_DIR / "recommendation_evaluation_v7_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print(json.dumps(details, indent=2))
    print(f"Saved corrected evaluation to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
