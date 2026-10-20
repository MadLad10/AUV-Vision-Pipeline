from auv_pipeline.benchmark import comparison_rows, to_markdown
from auv_pipeline.cli import build_parser, main


def test_every_stage_has_help(capsys):
    for stage in ("presets", "check", "augment", "train", "prune", "export", "benchmark", "report", "run"):
        try:
            build_parser().parse_args([stage, "--help"])
        except SystemExit as e:
            assert e.code == 0
    assert "usage" in capsys.readouterr().out


def test_presets_command():
    assert main(["presets"]) == 0


def test_check_and_dry_run(make_dataset):
    data = str(make_dataset())
    assert main(["check", "--data", data]) == 0
    assert main(["train", "--data", data, "--preset", "yolo26n", "--augment", "physics", "--dry-run"]) == 0


def test_dry_run_refuses_broken_dataset(make_dataset):
    data = str(make_dataset(label="9 0.5 0.5 0.1 0.1\n"))
    assert main(["train", "--data", data, "--dry-run"]) == 1


def test_unknown_profile_is_a_clean_error(make_dataset):
    assert main(["train", "--data", str(make_dataset()), "--augment", "nope", "--dry-run"]) == 2


def test_report_on_saved_run():
    assert main(["report", "saved_runs/Torpedo"]) == 0


def test_benchmark_table_deltas():
    results = [
        {"model": "base.pt", "mAP50": 0.80, "params_M": 2.4, "fps": 100.0},
        {"model": "pruned.pt", "mAP50": 0.78, "params_M": 1.2, "fps": 150.0},
    ]
    rows = {r[0]: r for r in comparison_rows(results)}
    assert rows["Params (M)"][2].startswith("1.20 (-50.0%")
    assert "!" in rows["mAP50"][2]  # accuracy drop is flagged
    assert "| Metric | base.pt | pruned.pt |" in to_markdown(results)
