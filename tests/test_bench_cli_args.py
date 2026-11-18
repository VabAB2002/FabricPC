"""SFR-004: bad number flags are rejected before anything runs."""

import pytest

from fabricpc.bench.__main__ import main


def rejected(argv, capsys):
    """Run the CLI, expect argparse's exit code 2, and return what it printed."""
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_zero_or_negative_trials_is_rejected_by_name(value, capsys):
    err = rejected(["mnist-mlp-spc", "--trials", value], capsys)
    assert "--trials" in err
    assert "nothing was run" in err


def test_rejected_run_does_not_delete_old_results(tmp_path, capsys):
    old = tmp_path / "mnist-mlp-spc" / "trial0.json"
    old.parent.mkdir()
    old.write_text("{}")
    rejected(["mnist-mlp-spc", "--trials", "0", "--out", str(tmp_path)], capsys)
    assert old.exists()


def test_every_bad_flag_is_named_in_one_message(capsys):
    err = rejected(
        ["mnist-mlp-spc", "--trials", "0", "--epochs", "-2", "--timed", "0"], capsys
    )
    for flag in ("--trials", "--epochs", "--timed"):
        assert flag in err


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--epochs", "nan"),
        ("--epochs", "inf"),
        ("--epochs", "0"),
        ("--warmup", "-1"),
        ("--trial", "-1"),
        ("--curve-batches", "-1"),
        ("--floor", "1.5"),
        ("--floor", "nan"),
        ("--price-per-hour", "0"),
    ],
)
def test_other_bad_values_are_rejected(flag, value, capsys):
    assert flag in rejected(["mnist-mlp-spc", flag, value], capsys)


def test_edge_values_that_work_today_are_still_allowed(capsys):
    code = main(
        [
            "mnist-mlp-spc",
            "--dry-run",
            "--trials",
            "1",
            "--epochs",
            "0.5",
            "--warmup",
            "0",
            "--curve-batches",
            "0",
            "--trial",
            "0",
            "--floor",
            "0",
        ]
    )
    assert code == 0


# --- a family id (several rows) -----------------------------------------------


def test_dry_run_of_a_family_describes_its_rows_and_trains_nothing(
    tmp_path, capsys, monkeypatch
):
    import json

    import fabricpc.bench.__main__ as cli

    trained = []
    monkeypatch.setattr(cli, "_run_one_row", lambda *a: trained.append(a) or 0)
    assert main(["mnist-mlp", "--dry-run", "--out", str(tmp_path)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["id"] for r in rows] == list(cli.COMPARISONS["mnist-mlp"].rows)
    assert trained == []


def test_trial_with_a_family_is_refused(tmp_path, capsys, monkeypatch):
    import fabricpc.bench.__main__ as cli

    trained = []
    monkeypatch.setattr(cli, "_run_one_row", lambda *a: trained.append(a) or 0)
    assert main(["mnist-mlp", "--trial", "0", "--out", str(tmp_path)]) == 2
    assert "Nothing was run" in capsys.readouterr().err
    assert trained == []
