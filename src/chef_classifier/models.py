from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf_svc_pipeline(
    **tfidf_kwargs: Any,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(**tfidf_kwargs),
            ),
            (
                "classifier",
                LinearSVC(),
            ),
        ]
    )
