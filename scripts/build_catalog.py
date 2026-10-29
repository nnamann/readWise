"""Validate the source dataset + sourced metadata-only additions and write a curation report."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, OUTPUT_DIR, RAW_DIR
from src.data_loader import _find_file, _normalize_columns, _read_table, load_curated_dataset, load_dataset


def _author_count(books: pd.DataFrame) -> int:
    authors = books.author.fillna("").astype(str).str.strip()
    invalid = {"", "nan", "none", "null", "unknown", "unknown author", "n/a", "na",
               "not available", "not applicable (na )"}
    valid = ~authors.str.casefold().isin(invalid)
    return int(authors[valid].str.casefold().nunique())


def build_report(raw_dir: Path = RAW_DIR, additions_path: Path | None = None,
                 evaluation_dir: Path = OUTPUT_DIR) -> dict:
    original_books, ratings = load_dataset(raw_dir)
    raw_books_table = _normalize_columns(_read_table(_find_file(("BX-Books.csv", "Books.csv"), raw_dir)))
    raw_id_column = "isbn" if "isbn" in raw_books_table else "book-id"
    raw_ids = raw_books_table[raw_id_column].astype(str).str.strip()
    raw_duplicate_ids = int(raw_ids.duplicated(keep="first").sum())
    title_key = original_books.title.astype(str).str.casefold().str.replace(r"[^\w]+", " ", regex=True).str.strip()
    author_key = original_books.author.astype(str).str.casefold().str.replace(r"[^\w]+", " ", regex=True).str.strip()
    duplicate_key_counts = (title_key + "::" + author_key).value_counts()
    duplicate_groups = duplicate_key_counts[duplicate_key_counts.gt(1)]
    curated_books, curated_ratings = load_curated_dataset(raw_dir, additions_path)
    additions_path = additions_path or (DATA_DIR / "curated" / "additional_books.csv")
    additions = pd.read_csv(additions_path, dtype=str, keep_default_na=False)
    explicit = ratings.loc[ratings.rating.gt(0)]
    zero = ratings.loc[ratings.rating.eq(0)]
    matched = explicit.book_id.astype(str).isin(set(original_books.book_id.astype(str)))
    matched_books = explicit.loc[matched, "book_id"].nunique()
    metadata_columns = ["title", "author", "publisher", "year"]

    def missing_counts(frame: pd.DataFrame) -> dict:
        result = {}
        for field in metadata_columns:
            values = frame[field].fillna("").astype(str).str.strip()
            invalid = {"", "nan", "none", "null", "unknown", "unknown author", "n/a", "na",
                       "not available", "not applicable (na )"}
            missing = values.str.casefold().isin(invalid)
            if field == "year":
                missing |= values.eq("0")
            result[field] = int(missing.sum())
        return result

    eval_results_path = evaluation_dir / "recommendation_evaluation_v5_results.csv"
    eval_details_path = evaluation_dir / "recommendation_evaluation_v5_details.json"
    historical_results_path = evaluation_dir / "evaluation_results.csv"
    reconciliation_path = evaluation_dir / "rating_reconciliation_report.json"
    evaluation_results = (pd.read_csv(eval_results_path).to_dict(orient="records")
                          if eval_results_path.is_file() else None)
    evaluation_details = (json.loads(eval_details_path.read_text(encoding="utf-8"))
                          if eval_details_path.is_file() else None)
    historical_results = (pd.read_csv(historical_results_path).to_dict(orient="records")
                          if historical_results_path.is_file() else None)
    reconciliation = (json.loads(reconciliation_path.read_text(encoding="utf-8"))
                      if reconciliation_path.is_file() else None)
    report = {
        "scope": "Full Book-Crossing source catalogue retained; verified metadata-only additions appended.",
        "count_interpretation": "The local BX-Books.csv has 270,491 parsed rows, of which one duplicate ISBN row is removed by the existing loader, leaving 270,490 unique ISBN edition records. The 22,451 count does not appear in the current workspace files; it may describe a separate filtered snapshot, but that cannot be confirmed locally.",
        "source": {
            "books_rows_after_loader": len(original_books),
            "raw_book_rows_read": len(raw_books_table),
            "distinct_book_ids": int(original_books.book_id.nunique()),
            "duplicate_book_id_rows_in_raw_file": raw_duplicate_ids,
            "normalized_title_author_duplicate_groups": int(len(duplicate_groups)),
            "records_in_normalized_title_author_duplicate_groups": int(duplicate_groups.sum()),
            "ratings_rows_after_loader": len(ratings),
            "explicit_rating_rows": int(len(explicit)),
            "zero_valued_interactions": int(len(zero)),
            "explicit_ratings_matching_catalogue": int(matched.sum()),
            "explicit_ratings_without_catalogue_record": int((~matched).sum()),
            "books_with_explicit_ratings": int(matched_books),
            "books_without_explicit_ratings": int(original_books.book_id.nunique() - matched_books),
            "distinct_authors": _author_count(original_books),
            "missing_metadata": missing_counts(original_books),
            "years_outside_1000_to_2026": int((pd.to_numeric(original_books.year, errors="coerce").lt(1000) |
                                                pd.to_numeric(original_books.year, errors="coerce").gt(2026)).sum()),
            "duplicate_book_id_rows_removed_by_loader": raw_duplicate_ids,
        },
        "curated": {
            "books_rows": len(curated_books),
            "distinct_book_ids": int(curated_books.book_id.nunique()),
            "verified_metadata_only_additions": len(additions),
            "retained_source_books": len(original_books),
            "source_records_removed": 0,
            "distinct_authors": _author_count(curated_books),
            "missing_metadata": missing_counts(curated_books),
            "years_outside_1000_to_2026": int((pd.to_numeric(curated_books.year, errors="coerce").lt(1000) |
                                                pd.to_numeric(curated_books.year, errors="coerce").gt(2026)).sum()),
            "books_without_explicit_rating_evidence": int(curated_books.book_id.nunique() - matched_books),
            "ratings_rows": len(curated_ratings),
            "rating_data_identical_to_source": bool(ratings.equals(curated_ratings)),
            "rating_data_note": "No ratings were added for metadata-only records. The normalized source frame includes checksum-safe identifier repairs and deterministic duplicate user-book consolidation; raw source rows remain unchanged and are audited separately.",
            "subject_coverage": "Not reported: the source catalogue has no consistently populated, verified subject field; additions were not assigned inferred genres.",
        },
        "duplicate_policy": {
            "isbn": "ISBN/book_id is preserved as a string; duplicate IDs in the supplement and collisions with source IDs raise an error.",
            "title_author": "Normalized exact title + author collisions raise an error; title-only matches are not merged.",
            "source_duplicates": "Existing source records are retained as supplied after the loader's existing first-occurrence ISBN dedupe; edition-level title-author duplicates are not globally removed.",
        },
        "additions": additions[["book_id", "title", "author", "source_name", "source_url"]].to_dict(orient="records"),
        "evaluation": {
            "historical_outputs_preserved": True,
            "new_metrics_computed": evaluation_results is not None and evaluation_details is not None,
            "method": f"Same deterministic per-user holdout configuration and ratings; all {curated_books.book_id.nunique():,} catalogue records are candidates. Holdout pairs are excluded from training.",
            "results_file": eval_results_path.name if evaluation_results is not None else None,
            "details_file": eval_details_path.name if evaluation_details is not None else None,
            "results": evaluation_results,
            "details": evaluation_details,
            "historical_results": historical_results,
            "historical_metrics_comparable": False,
            "interpretation": "Historical evaluation_results.csv and evaluation_details.json remain unchanged. Evaluation v5 uses checksum-reconciled ratings, excludes unmatched items from user eligibility and relevant targets, applies conservative seen-edition control, adds a train-only popularity baseline, and reports more ranking metrics. Historical numbers use an earlier protocol and are not directly comparable.",
        },
        "rating_reconciliation": reconciliation,
    }
    return report


def main() -> None:
    report = build_report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "catalogue_curation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    additions = report["additions"]
    lines = ["# Catalogue curation report", "", report["scope"], "",
             report["count_interpretation"], "",
             "## Counts", "",
             f"- Source books: {report['source']['books_rows_after_loader']:,} rows / {report['source']['distinct_book_ids']:,} distinct identifiers.",
             f"- Curated books: {report['curated']['books_rows']:,} rows / {report['curated']['distinct_book_ids']:,} distinct identifiers.",
             f"- Verified metadata-only additions: {report['curated']['verified_metadata_only_additions']:,}.",
             f"- Source books with explicit ratings: {report['source']['books_with_explicit_ratings']:,}; without explicit ratings: {report['source']['books_without_explicit_ratings']:,}.",
             f"- Ratings unchanged by metadata additions: {report['curated']['rating_data_identical_to_source']} ({report['curated']['ratings_rows']:,} normalized rows).",
             f"- Distinct authors: {report['source']['distinct_authors']:,} to {report['curated']['distinct_authors']:,}.",
             f"- Source missing or placeholder metadata: {report['source']['missing_metadata']}.",
             f"- Source rows with repeated normalized title-author keys: {report['source']['records_in_normalized_title_author_duplicate_groups']:,} across {report['source']['normalized_title_author_duplicate_groups']:,} groups; retained as possible distinct editions.",
             f"- Explicit ratings without a matching source catalogue record: {report['source']['explicit_ratings_without_catalogue_record']:,}.",
             "", "## Added records", ""]
    lines.extend(f"- **{row['title']}** — {row['author']} ([Open Library record]({row['source_url']}))." for row in additions)
    if report["evaluation"]["results"]:
        lines += ["", "## Model comparison on the same test split", "",
                  "Macro averages over readers with at least one relevant held-out book; popularity uses training ratings only.", "",
                  "| Model | P@5 | P@10 | R@5 | R@10 | NDCG@10 | MAP@10 | Coverage | Diversity |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        lines.extend(
            f"| {row['model']} | {row['precision_at_5']:.4f} | {row['precision_at_10']:.4f} | {row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | {row['ndcg_at_10']:.4f} | {row['map_at_10']:.4f} | {row['catalogue_coverage']:.4%} | {row['intra_list_diversity']:.4f} |"
            for row in report["evaluation"]["results"]
        )
    if report["rating_reconciliation"]:
        rec = report["rating_reconciliation"]
        lines += ["", "## Rating reconciliation", "",
                  f"- Raw parsed rating rows audited: {rec['raw_rating_rows']:,}.",
                  f"- Identifier status across parsed source rows: {rec['identifier_status_counts']} (includes zero-valued interactions, before duplicate consolidation).",
                  f"- Rows available to the explicit-rating model: {rec['model_status_counts'].get('included_explicit_model', 0):,}; zero-valued interactions: {rec['model_status_counts'].get('zero_implicit_not_explicit_rating', 0):,}; duplicate rows superseded after safe repair: {rec['model_status_counts'].get('superseded_duplicate_user_book', 0):,}.",
                  f"- Explicit rows recovered by safe ISBN format repair and available to the model: {rec['recovered_explicit_rows_used_by_model']:,}.",
                  f"- Explicit rows still unresolved after repair and duplicate handling: {rec['explicit_unresolved_rows_after_format_repairs_and_dedup']:,} ({rec['explicit_unresolved_unique_ids']:,} unique IDs).",
                  f"- Unresolved explicit rows with valid but absent ISBNs: {rec['explicit_unresolved_valid_isbn_rows']:,}; invalid/non-ISBN IDs: {rec['explicit_unresolved_invalid_or_non_isbn_rows']:,}."]
    lines += ["", "## Data and evaluation notes", "",
              "All source books were retained: the dataset has no reliable subject labels with which to make a balanced 10,000–15,000 selection, and repeated title-author keys may represent distinct ISBN editions. The app collapses clear physical-format variants within recommendation lists. Raw source files and historical model evaluation files remain intact. The row-level reconciliation is derived and auditable; source IDs are retained alongside any resolved ISBN. New catalogue records have no ratings and appear with zero observed rating evidence. No categories or ratings were inferred. Earlier metrics are not directly comparable because v5 restricts users and relevant targets to catalogue-matched books and adds rating-ID reconciliation, edition control, and ranking metrics.",
              "", report["curated"]["subject_coverage"], ""]
    (OUTPUT_DIR / "catalogue_curation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote catalogue report for {report['curated']['distinct_book_ids']:,} books ({len(additions)} additions).")


if __name__ == "__main__":
    main()
