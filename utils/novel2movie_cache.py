"""Resume-cache helpers for novel2movie retrieval."""

from __future__ import annotations

import os
from collections.abc import Iterator


def iter_score_chunk_files(chunks_dir: str) -> Iterator[tuple[str, float]]:
    """Yield (path, score) for chunk_*-score_*.txt files in a resume cache dir."""
    for chunk_fname in os.listdir(chunks_dir):
        if "-score_" not in chunk_fname or not chunk_fname.endswith(".txt"):
            continue
        try:
            score = float(chunk_fname.split("-score_")[1].split(".txt")[0])
        except ValueError:
            continue
        yield os.path.join(chunks_dir, chunk_fname), score
