import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import OUTPUT_DIR
from src.data_loader import load_curated_dataset
from src.evaluation import EvaluationConfig, evaluate_models

books, ratings = load_curated_dataset()
results, details = evaluate_models(books, ratings, EvaluationConfig())
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_DIR / "recommendation_evaluation_v5_results.csv", index=False)
(OUTPUT_DIR / "recommendation_evaluation_v5_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
print(results.to_string(index=False))
print(json.dumps(details, indent=2))
print(f"Saved expanded-catalogue results to {OUTPUT_DIR}")
