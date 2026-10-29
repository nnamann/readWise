# Methodology and report outline

## Abstract

ReadWise compares content-based, item-based collaborative, and hybrid book recommendation on Book-Crossing. It uses TF-IDF metadata profiles and explicit user ratings, supports cold-start discovery, and evaluates top-10 rankings with a per-user held-out split. Insert the actual measured result table only after running the evaluation script.

## Suggested report chapters

1. **Introduction:** Recommendation problem, motivation, scope, and relevance to readers.
2. **Objectives:** Implement explainable content and collaborative recommenders, combine their evidence, handle cold start, and measure ranking quality.
3. **Problem statement:** A reader needs relevant books from a large catalogue with sparse feedback; title/author metadata and other readers' ratings provide complementary signals.
4. **System architecture:** Describe the data loader, feature/profile construction, item similarity, hybrid ranking, UI, and evaluation flow (see `architecture.md`).
5. **Dataset:** Report source archive URL, retrieval date, source terms, table schemas, record counts from `python scripts/prepare_data.py`, and handling of missing values, duplicates, ISBNs, and zero ratings.
6. **Methodology:** Explain TF-IDF and cosine similarity; sparse user-item ratings; item-based centered cosine; overlap shrinkage; and normalized weighted hybrid. State that no genres/descriptions are presumed.
7. **Implementation:** Summarize the Python modules, Streamlit interactions, cache boundaries, empty states, and fallback covers.
8. **Experimental setup:** Seed 3143; sample up to 100 users with at least five explicit ratings; hold out 20% per sampled user while retaining two for training; relevance threshold 8/10. All held-out interactions from sampled users are excluded from training, even when that user's test set contains no relevant book. Exclude training-seen books before top-K truncation. Explain the sampled users and metric cohort from `outputs/evaluation_details.json`.
9. **Actual results:** Corrected run: 100 eligible readers sampled; 69 readers included in the macro-average; 446 explicit interactions held out; 269 relevant held-out books. Content-based Precision@10/Recall@10 = 0.017391/0.035933; collaborative = 0.001449/0.001610; hybrid = 0.005797/0.020612. The earlier saved run is not directly comparable because held-out interactions from sampled users omitted from the metric cohort remained in its training rows.
10. **Discussion:** Compare observed metrics, sparse-evidence behavior, and differences among methods, including cases where one method has no candidate evidence.
11. **Limitations:** Old and sparse data; narrow metadata; bounded sample; arbitrary but stated relevance threshold; no claims of broad generalization.
12. **Future scope:** Genre/enriched descriptions, implicit ranking, confidence-aware hybrid weights, larger sampled evaluation, and user studies.
13. **Conclusion:** State only what was implemented and measured.
14. **References:** Cai-Nicolas Ziegler, Sean M. McNee, Joseph A. Konstan, and Georg Lausen. “Improving Recommendation Lists Through Topic Diversification.” Proceedings of the 14th International World Wide Web Conference (WWW ’05), 2005. DOI: https://doi.org/10.1145/1060745.1060754. Dataset source: https://mooc.grouplens.org/datasets/book-crossing/ (accessed 2026-10-26). Add scikit-learn documentation and course materials as required by the instructor.

## Metric definitions

For each user, Precision@K = relevant recommendations among the first K ÷ K. Recall@K = relevant recommendations among the first K ÷ that user's relevant held-out books. Metrics are macro-averaged across the same users with at least one relevant held-out rating. Recommendation lists can contain fewer than K results; missing slots count as not relevant. Sampled users with no relevant held-out items are excluded from both averages, but all their held-out interactions remain out of training. This protocol evaluates top-N ranking, not rating prediction.
