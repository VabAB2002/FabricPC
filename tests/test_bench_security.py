"""Names and links that come from result files must not escape the results
folder or run code in report.html (found in the 11/18 review)."""

import pytest

from fabricpc.bench.definitions.registry import COMPARISONS
from fabricpc.bench.results.report import _inline_html
from fabricpc.bench.stats.compare import compare_family


def test_a_metric_name_cannot_point_outside_the_results_folder(tmp_path):
    with pytest.raises(ValueError, match="plain name"):
        compare_family(tmp_path, COMPARISONS["mnist-mlp"], metric="/../../ESCAPED")
    assert not (tmp_path.parent / "ESCAPED.json").exists()


@pytest.mark.parametrize("bad", [None, 3])
def test_a_metric_that_is_not_text_is_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        compare_family(tmp_path, COMPARISONS["mnist-mlp"], metric=bad)


def test_report_links_are_only_http_or_relative():
    assert 'href="https://example.org"' in _inline_html("[a](https://example.org)")
    assert 'href="mnist-mlp/summary.json"' in _inline_html(
        "[s](mnist-mlp/summary.json)"
    )
    for bad in ("javascript:alert(1)", "JavaScript:x", "java\tscript:x", "data:x"):
        assert "<a" not in _inline_html(f"[x]({bad})"), bad
