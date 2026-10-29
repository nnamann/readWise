# Architecture

```text
Book-Crossing CSV → data_loader → preprocessing
                         ↓               ↓
                    Streamlit UI ← TF-IDF content model
                         ↑       ← item-based rating model
                         └────── ← weighted hybrid
                                  evaluation → outputs/
```

`data_loader` detects CSV delimiters, reads Latin-1, validates required fields, retains string IDs, drops missing titles and duplicate ISBN/user-book rows, and bounds ratings to 0–10. Duplicate catalogue ISBNs keep the first source record; duplicate user/book ratings keep the last source occurrence. `preprocess` defines the explicit-rating subset (rating > 0). The content model vectorizes title, author and publisher and calculates requested cosine similarities, excluding seed and caller-specified seen items before top-K. The collaborative model stores a sparse item-user matrix, uses user-mean-centered explicit values, and counts co-rating overlap before scoring; target-reader ratings are centered around that reader's own mean. The hybrid normalizes available candidate sources, honors the metadata weight, and keeps per-item explanations. Evaluation withholds every sampled user's interactions before fitting and excludes training-seen books before truncating recommendation lists. Streamlit caches deterministic local data and model construction, while per-user selections stay in session state.

## Request flow

1. Discover presents titles sorted by observed rating count; Browse searches and pages through catalogue records by title, author, publisher or year, with source covers and rating summaries.
2. Readers open a book record or save it to the session shelf. For you collects up to five searchable seeds and an optional validated dataset reader ID.
3. The selected approach ranks candidates, excludes seeds and applicable known ratings, and gives a reason based on the evidence used. Results are flagged for refresh when inputs change.
4. Results reads metrics produced by the offline evaluation script and distinguishes sampled eligible readers from the readers included in the macro-average.

## Security and reliability review

The app accepts bounded integer Top-K in model APIs, validates hybrid weights and uses IDs only to look up in-memory records. It does not accept uploads, fetch user-provided URLs, execute archives, or load serialized models. The downloaded ZIP members were checked for path traversal and unexpected executable extensions before extraction. Raw dataset files are ignored by Git. The CSV parser skips malformed rows and validates key schema; inspect source data quality before relying on it. This review is limited to this educational local application and is not a production security audit.
