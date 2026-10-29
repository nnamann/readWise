# ReadWise

**ReadWise: A Hybrid Book Recommendation System Using Content-Based and Collaborative Filtering** is a local Streamlit project for AIM3143, Recommender Systems, Manipal University Jaipur (B.Tech CSE AIML). It demonstrates TF-IDF content profiles, item-based collaborative filtering, a weighted hybrid, cold-start support, and offline ranking evaluation.

## Problem and objectives

Readers face a large catalogue and sparse feedback. ReadWise compares recommendations derived from book metadata and genuine user ratings. It is an educational prototype designed to make the model decisions explainable in a viva.

## Features and architecture

- Separate Discover, Explore, My Shelf, For You, How It Works, and Research sections with compact responsive navigation.
- Start from a real featured title, search titles or authors, browse popular and quieter-reader shelves, and open author-focused collections.
- Explore the catalogue in a cover grid, filter by title/author/publisher/year, sort by observed rating evidence, paginate, and open book details.
- Save and remove books on a session-only My Shelf, then use up to five picks to seed a reading list. Optionally validate a Book-Crossing reader ID and choose plain-language catalogue, reader-rating, or blended recommendations.
- View observed ratings and rating counts; recommendations explain the evidence and exclude favorites or previously rated books.
- Compare measured Precision@10 and Recall@10 from a per-user held-out experiment.
- `app.py` is the Streamlit presentation; `src/` contains data handling, models, and evaluation; `scripts/` provides validation and evaluation entry points.

## Dataset and attribution

The application uses the real Book-Crossing dataset archive hosted at https://cdn.freecodecamp.org/project-data/books/book-crossings.zip, downloaded and validated on 2026-10-26. The archive included `BX-Books.csv`, `BX-Book-Ratings.csv`, and `BX-Users.csv`. BX files are quoted, semicolon-delimited and Latin-1 compatible. The loader preserves ISBNs as strings and treats zero as an implicit interaction rather than an explicit zero-star rating. The raw archive and files are stored in `data/raw/` and excluded from Git. The data is described as available for acknowledged research use, with published references specifying non-commercial use; verify terms before redistribution or commercial use. Cite Ziegler et al., “Improving Recommendation Lists Through Topic Diversification,” WWW ’05, DOI: 10.1145/1060745.1060754. See [data/README.md](data/README.md) for acquisition details.

## Windows PowerShell setup

Use Python 3.11 or newer. From the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/prepare_data.py
streamlit run app.py
```

If PowerShell blocks environment activation, use `\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `\.venv\Scripts\python.exe -m streamlit run app.py`.

Open the local address printed by Streamlit (normally `http://localhost:8501`). Keep the terminal open while using the app; press Ctrl+C there to stop it. The virtual environment is created separately on each laptop and should not be copied.

Reader feedback is stored separately from Book-Crossing data in a private local SQLite file at `~/Documents/Codex/ReadWise/feedback.db` by default. Set `READWISE_FEEDBACK_DB_PATH` to choose another local location. On first launch, existing feedback in `data/feedback.db` is copied into this private store.

### Sharing with a teammate

Send the project folder (or a ZIP) with the source files, `requirements.txt`, `scripts/`, `src/`, `tests/`, and `data/raw/`. The two required files `BX-Books.csv` and `BX-Book-Ratings.csv` are large and excluded from Git; `scripts/prepare_data.py` validates local files but does not download them. If you do not share the dataset files, your teammate must obtain the Book-Crossing archive from the source documented in [data/README.md](data/README.md), extract it, and put the CSV files in `data/raw/` before following the setup steps. Check the dataset's research/non-commercial terms before redistribution. Do not send `.venv/` or Python cache folders.

## Algorithms

**Content-based:** TF-IDF profiles combine title, author, and publisher; cosine similarity compares selected books with catalogue entries. No genre or synopsis is assumed. **Collaborative:** explicit ratings are user-mean-centered and compared with item-based cosine similarity. Candidate scores aggregate deviations from the target reader's own mean, weighted by item similarity; similarities require rating overlap. **Hybrid:** candidate scores are min-max normalized within each model's candidates and blended as `alpha × content + (1 − alpha) × collaborative` (default alpha 0.5). The slider represents catalogue-detail influence. When one model has no evidence, the available model is used and identified as a fallback. Recommendation cards group exact repeated title/author records for display; Browse still exposes editions.

## Evaluation

Run `python scripts/evaluate_models.py` to regenerate `outputs/evaluation_results.csv` and `outputs/evaluation_details.json`. This corrected run sampled 100 readers, held out 446 explicit interactions in total, and included 69 readers with 269 relevant held-out books (rating ≥8) in the macro-average. Metrics: content-based Precision@10/Recall@10 = 0.017391/0.035933; collaborative = 0.001449/0.001610; hybrid = 0.005797/0.020612. The protocol holds out 20% per sampled reader (keeping at least two for training); all held-out interactions stay out of training, including for sampled readers with no relevant held-out book. Models exclude each reader's training-rated books before top-K truncation; held-out books remain candidates. Precision@K divides by K, so short lists count missing places as misses. The earlier saved run is not directly comparable: its training set retained held-out interactions from sampled users omitted from the metric cohort. These scores show a limitation of the current signals, not a production benchmark.

## Limitations

Book-Crossing is old, globally sparse, and ratings are not a random sample of readers. Title/author/publisher similarity is a narrow content signal. Collaborative evidence can be unavailable for new users or low-overlap books. The evaluation sample is intentionally bounded and should not be interpreted as a production benchmark. Cover image URLs come from source metadata and may no longer resolve; a local styled fallback appears when a cover fails. The app makes no external service necessary for recommendations.

## Troubleshooting

- Missing dataset: ensure `data/raw/BX-Books.csv` and `data/raw/BX-Book-Ratings.csv` exist. See `data/README.md`.
- Empty collaborative results: use a valid existing user with at least two catalogue-matched explicit ratings, or choose catalogue-detail discovery.
- Empty content results: select a book with title/author metadata.
- Rebuild metrics after data or evaluation changes with `python scripts/evaluate_models.py`.
- Start tests with `pytest`.

## Project report and team

The detailed methodology, architecture, report outline, and viva prompts are in `docs/`. Team contributions: Student 1 — [name and contribution]; Student 2 — [name and contribution]. Replace these placeholders with actual contributions.
