# Viva questions and concise answer prompts

1. **What is content-based filtering?** It ranks items by similarity to a user's chosen or liked item profile. Here TF-IDF represents title, author, and publisher text.
2. **Why TF-IDF?** It downweights common terms and gives more weight to descriptive terms; this is useful for sparse metadata text.
3. **What does cosine similarity measure?** The angle between feature vectors; values near one indicate similar term profiles.
4. **What is the user-item matrix?** Rows are users, columns are books, and observed explicit ratings are its entries; missing ratings are unknown, not zero.
5. **How does collaborative filtering work here?** It finds books with similar rating patterns, then weights a user's rating deviations by those item similarities.
6. **Why mean-center ratings?** Users use rating scales differently; subtracting the user's average emphasizes relative preference.
7. **Why require rating overlap?** Similarity from one shared rater is unreliable; the minimum-overlap rule reduces such unsupported matches.
8. **How does the hybrid work?** It normalizes available content and collaborative candidate scores and combines them with a tunable weight, default 0.5.
9. **What is cold start?** A new user has no rating history. Favorite-book content profiles still provide recommendations; collaborative evidence is unavailable.
10. **What is sparsity?** Most users rate only a tiny share of books, so most matrix cells are unknown.
11. **How are zero ratings handled?** BX zero indicates an implicit interaction, not an explicit zero-star rating; it is excluded from explicit rating models and relevance evaluation.
12. **How do you avoid evaluation leakage?** Split each user's observed explicit ratings first, then construct collaborative profiles from training interactions only and exclude training-rated books from candidates.
13. **What does Precision@10 show?** The fraction of ten ranked slots that contain relevant held-out books.
14. **What does Recall@10 show?** The fraction of that user's relevant held-out books recovered in the first ten.
15. **What are the limitations?** Sparse and old data, limited metadata, small bounded evaluation sample, and dependence on rating overlap.
16. **Does hybrid always win?** No. The saved experiment determines which method performs better under the selected split and relevance threshold.
