from auv_pipeline.dataset import inspect_dataset, label_path


def test_clean_dataset_passes(make_dataset):
    report = inspect_dataset(make_dataset())
    assert report.ok, report.errors
    assert report.has_test
    assert report.splits["train"].images == 4
    assert report.splits["train"].instances[0] == 4
    assert any("torpedo" in w for w in report.warnings)  # class 1 has no instances


def test_bad_class_id_is_an_error(make_dataset):
    report = inspect_dataset(make_dataset(label="5 0.5 0.5 0.2 0.2\n"))
    assert not report.ok
    assert any("malformed" in e for e in report.errors)


def test_missing_labels_warn_and_empty_labels_count_as_background(make_dataset):
    report = inspect_dataset(make_dataset(label=None))
    assert report.ok
    assert report.splits["train"].missing_labels == 4

    report = inspect_dataset(make_dataset(label=""))
    assert report.splits["train"].background == 4


def test_missing_yaml(tmp_path):
    report = inspect_dataset(tmp_path / "nope.yaml")
    assert not report.ok


def test_label_path_swaps_last_images_dir(tmp_path):
    img = tmp_path / "images" / "x" / "images" / "train" / "a.jpg"
    assert label_path(img) == tmp_path / "images" / "x" / "labels" / "train" / "a.txt"
