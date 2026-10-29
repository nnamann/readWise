# Dataset

The raw Book-Crossing files are stored in `data/raw/` locally and are excluded from Git. They were retrieved on 2026-10-26 from the freeCodeCamp-hosted archive: https://cdn.freecodecamp.org/project-data/books/book-crossings.zip. The archive provided `BX-Books.csv`, `BX-Book-Ratings.csv`, and `BX-Users.csv`; schemas and sample records were checked. Book-Crossing documentation describes the data as freely available for research use when acknowledged; another published description specifies non-commercial use. Cite the dataset paper and confirm terms with the data provider before redistributing or using commercially. The source credit is Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, and Georg Lausen, “Improving Recommendation Lists Through Topic Diversification,” WWW ’05, DOI: https://doi.org/10.1145/1060745.1060754. ISBNs are kept as strings. Explicit ratings 1–10 are used for collaborative filtering; zero denotes an implicit interaction and is excluded from explicit-rating recommendations and evaluation. The loader skips malformed CSV rows; the validated local count is therefore recorded by `scripts/prepare_data.py` rather than assumed from headline source counts.

If the files are absent, place those three files in this directory or `data/raw/`. The loader also accepts Kaggle-style `Books.csv` and `Ratings.csv` names. No ratings are synthesized.
# Curated catalogue additions

`curated/additional_books.csv` contains verified bibliographic metadata for books
missing from the Book-Crossing source. These are catalogue-only records: no
ratings, user histories, inferred genres, or cover links are added. ISBN values
are kept as text. Each record includes the Open Library edition record used as
its source.

The app loads the complete normalized raw catalogue and appends this file.
Original files in `data/raw/` remain untouched. Regenerate the reproducible
summary with:

```powershell
.venv\Scripts\python.exe scripts\build_catalog.py
```

This writes `outputs/catalogue_curation_report.json` and
`outputs/catalogue_curation_report.md`. To add a record, verify an edition's
title, author, identifier, publication details, and source URL; leave unknown
fields blank. The loader rejects duplicate identifiers and exact normalized
title-author collisions with the source or supplement. It deliberately keeps
different editions and does not classify subjects from title guesses.

Run `.venv\Scripts\python.exe scripts\reconcile_ratings.py` to create the
row-level ID and model-inclusion audit in `outputs/rating_reconciliation_rows.csv`
and its summary JSON. Only checksum-valid formatting differences with a unique
catalogue match are repaired; unresolved ratings remain in the audit.

Run `.venv\Scripts\python.exe scripts\evaluate_models.py` to evaluate the
expanded candidate catalogue with a seeded per-user holdout and train-only
popularity baseline. Results go to
`outputs/recommendation_evaluation_v5_results.csv` and
`outputs/recommendation_evaluation_v5_details.json`. The pre-curation evaluation
files remain unchanged; their old protocol is not directly comparable to v2.
Rebuild the catalogue report after these commands to include reconciliation and
evaluation metrics.
