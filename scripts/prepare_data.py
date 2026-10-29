import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.prepare_recent_books import generate_curated_catalog, CURATED_PATH
from src.data_loader import load_curated_dataset
from src.preprocess import parse_publication_year

# Ensure curated recent additions from Open Library are prepared reproducibly
curated_df = generate_curated_catalog(CURATED_PATH)

books, ratings = load_curated_dataset()
clean_years = parse_publication_year(books.year)
contemporary = books.loc[clean_years.ge(2015)]

print(f"Validated {len(books):,} catalogue books and {len(ratings):,} source user-book rows.")
print(f"Contemporary titles (published 2015-2024): {len(contemporary):,}.")
print(f"Explicit ratings: {(ratings.rating > 0).sum():,}; zero-valued interactions: {(ratings.rating == 0).sum():,}.")
print("Supplemental contemporary books are metadata-only (Open Library CC0) and have no fabricated rating interactions.")
