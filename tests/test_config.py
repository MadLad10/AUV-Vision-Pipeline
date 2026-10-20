import pytest

from auv_pipeline.config import deep_merge, list_presets, load_with_extends, parse_set, resolve_preset
from auv_pipeline.train import build_config


def test_every_preset_resolves():
    presets = list_presets()
    assert {"yolo26n", "yolo26m", "yolo11n"} <= set(presets)
    for name in presets:
        cfg = load_with_extends(resolve_preset(name))
        assert cfg["model"].endswith(".pt")
        assert cfg["train"]["epochs"] > 0  # inherited from _base.yaml


def test_deep_merge_replaces_lists_and_merges_dicts():
    out = deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2]}, {"a": {"y": 3}, "l": [9]})
    assert out == {"a": {"x": 1, "y": 3}, "l": [9]}


def test_parse_set_types_values():
    assert parse_set(["lr0=0.002", "cos_lr=true", "optimizer=SGD"]) == {
        "lr0": 0.002, "cos_lr": True, "optimizer": "SGD",
    }
    with pytest.raises(ValueError):
        parse_set(["nonsense"])


def test_unknown_preset_lists_available():
    with pytest.raises(FileNotFoundError, match="yolo26n"):
        resolve_preset("does_not_exist")


def test_merge_order(tmp_path):
    hyp = tmp_path / "best_hyperparameters.yaml"
    hyp.write_text("lr0: 0.0042\nhsv_s: 0.33\nmosaic: 0.7\n")
    cfg = build_config("yolo26n", hyp=str(hyp), overrides={"epochs": 7, "batch": None}, sets={"mosaic": 0.1})
    t = cfg["train"]
    assert t["lr0"] == 0.0042  # tuned value beats the preset
    assert t["hsv_s"] == 0.33  # ...and the augment profile
    assert t["epochs"] == 7  # CLI beats everything below it
    assert t["batch"] == 16  # None overrides are ignored
    assert t["mosaic"] == 0.1  # --set wins last


def test_augment_profile_sets_hsv():
    cfg = build_config("yolo26n", augment="physics")
    assert cfg["augment"] == "physics"
    assert cfg["train"]["hsv_s"] == 0.5
    assert build_config("yolo11n")["train"]["hsv_s"] == 0.07
