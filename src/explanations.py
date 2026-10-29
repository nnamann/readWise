def recommendation_explanation(method: str) -> str:
    return {
        "Content-based": "Metadata similarity based on title, author, and publisher.",
        "Collaborative": "Recommendations supported by rating patterns on books with overlapping reader evidence.",
        "Hybrid": "A weighted combination of catalogue-detail and collaborative rating evidence.",
    }.get(method, "Recommendation evidence is shown with each result.")
