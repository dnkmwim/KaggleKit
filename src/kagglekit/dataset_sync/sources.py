from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Source:
    kind: str
    value: str

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.value}"


SOURCES = tuple(
    Source("tag", value)
    for value in (
        "artificial intelligence",
        "classification",
        "regression",
        "clustering",
        "time series analysis",
        "nlp",
        "computer vision",
        "recommender systems",
        "outlier analysis",
    )
) + tuple(
    Source("query", value)
    for value in ("data science", "machine learning", "forecasting", "anomaly detection")
)

SOURCE_BY_KEY = {source.key: source for source in SOURCES}


def select_sources(key: str | None) -> tuple[Source, ...]:
    if key is None:
        return SOURCES
    try:
        return (SOURCE_BY_KEY[key],)
    except KeyError as exc:
        allowed = ", ".join(SOURCE_BY_KEY)
        raise ValueError(f"Unknown source {key!r}. Allowed: {allowed}") from exc

