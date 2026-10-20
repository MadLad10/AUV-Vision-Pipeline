import numpy as np
import pytest

A = pytest.importorskip("albumentations")

from auv_pipeline.augment import build_transforms, describe, load_profiles  # noqa: E402
from auv_pipeline.augment.underwater import CUSTOM_TRANSFORMS, WaterAttenuation  # noqa: E402


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", sorted(load_profiles()))
def test_every_profile_builds(name):
    profile = load_profiles()[name] or {}
    transforms = build_transforms(profile)
    assert len(describe(profile)) == (len(transforms) if transforms else 0)


@pytest.mark.parametrize("cls", CUSTOM_TRANSFORMS.values(), ids=CUSTOM_TRANSFORMS)
def test_custom_transform_keeps_shape_and_dtype(cls, image):
    out = cls(p=1.0)(image=image)["image"]
    assert out.shape == image.shape and out.dtype == np.uint8
    assert not np.array_equal(out, image)


def test_custom_transforms_are_reproducible(image):
    def run():
        return A.Compose([cls(p=1.0) for cls in CUSTOM_TRANSFORMS.values()], seed=7)(image=image)["image"]

    assert np.array_equal(run(), run())


def test_attenuation_removes_red_first():
    gray = np.full((8, 8, 3), 150, np.uint8)  # BGR
    out = WaterAttenuation(depth=5.0, water="clear", p=1.0)(image=gray)["image"].astype(int)
    b, g, r = out[..., 0].mean(), out[..., 1].mean(), out[..., 2].mean()
    assert r < g and r < b


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="Unknown transform"):
        build_transforms({"transforms": ["NotATransform"]})
    with pytest.raises(ValueError, match="water"):
        WaterAttenuation(water="lava")


def test_ultralytics_accepts_profile_transforms():
    pytest.importorskip("ultralytics")
    from ultralytics.data.augment import Albumentations

    alb = Albumentations(p=1.0, transforms=build_transforms(load_profiles()["physics"]))
    assert not alb.contains_spatial
    labels = {"img": np.zeros((64, 64, 3), np.uint8), "cls": np.zeros((0, 1))}
    assert alb(labels)["img"].shape == (64, 64, 3)
