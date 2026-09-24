import re
import unicodedata

from .models import DatasetMetadata


NEGATIVE_TERMS = (
    "interview questions",
    "interview question",
    "cheat sheets",
    "cheat sheet",
    "data science books",
    "data science book",
    "certification",
    "salaries",
    "salary",
    "careers",
    "career",
    "roadmaps",
    "roadmap",
    "handbooks",
    "handbook",
    "courses",
    "course",
    "resume",
    "jobs",
    "job",
)


def normalize_filter_text(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", value or "").lower()
    text = re.sub(r"[-‐‑‒–—]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def data_science_negative(metadata: DatasetMetadata) -> str | None:
    text = normalize_filter_text(f"{metadata.title} {metadata.subtitle or ''}")
    padded = f" {text} "
    for term in NEGATIVE_TERMS:
        if f" {term} " in padded:
            return term
    return None


def tag_page_is_valid(items: list[DatasetMetadata], expected_ref: str) -> bool:
    if not items:
        return True
    expected = normalize_filter_text(expected_ref)
    return any(normalize_filter_text(tag) == expected for item in items for tag in item.tags)

