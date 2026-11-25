"""Tests for SFR-034: Report Test Accuracy for Classification."""

import pytest

from fabricpc.bench.definitions.registry import (
    CLASSIFICATION_DATASETS,
    ROWS,
    is_classification_row,
)


def test_classification_rows_report_accuracy_as_primary_metric():
    """Verify every registered classification benchmark row uses 'accuracy'
    as its primary metric (SFR-034)."""
    # Collect all registered rows that represent classification benchmarks.
    classification_rows = [row for row in ROWS.values() if is_classification_row(row)]
    assert classification_rows, "Expected at least one classification benchmark row"

    # Confirm each classification row belongs to a known classification dataset
    # and reports test accuracy as its headline metric.
    for row in classification_rows:
        assert row.dataset in CLASSIFICATION_DATASETS
        assert row.metric == "accuracy", f"{row.id} primary metric is {row.metric!r}"