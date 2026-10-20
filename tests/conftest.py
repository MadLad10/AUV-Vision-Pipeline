from pathlib import Path

import numpy as np
import pytest


def _write_split(root: Path, split: str, n: int, label: str | None = "0 0.5 0.5 0.2 0.2\n") -> None:
    cv2 = pytest.importorskip("cv2")
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), np.full((32, 32, 3), 127, np.uint8))
        if label is not None:
            (root / "labels" / split / f"{i}.txt").write_text(label)


@pytest.fixture
def make_dataset(tmp_path):
    """Build a tiny YOLO dataset; returns the data.yaml path."""

    def _make(train=4, val=2, test=2, names=("gate", "torpedo"), label="0 0.5 0.5 0.2 0.2\n"):
        _write_split(tmp_path, "train", train, label)
        _write_split(tmp_path, "val", val, label)
        if test:
            _write_split(tmp_path, "test", test, label)
        lines = [f"path: {tmp_path}", "train: images/train", "val: images/val"]
        lines += ["test: images/test"] if test else []
        lines += ["names:"] + [f"  - {n}" for n in names]
        (tmp_path / "data.yaml").write_text("\n".join(lines) + "\n")
        return tmp_path / "data.yaml"

    return _make
