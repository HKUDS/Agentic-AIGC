"""Empty novel_text must not ZeroDivisionError on compression summary."""

from __future__ import annotations

from utils.text import compression_ratio_label


def test_empty_novel_ratio_is_na() -> None:
    assert compression_ratio_label("", "") == "📌 Compression Ratio: n/a (empty novel text)"


def test_nonempty_novel_ratio_formats() -> None:
    assert compression_ratio_label("abcd", "ab") == "📌 Compression Ratio: 50.00%"


