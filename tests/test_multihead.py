import yaml
import torch
import shutil
from types import SimpleNamespace
import numpy as np
import cv2
import pytest

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ops, loss
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.nn.modules import head

from tests import TMP


def test_check_det_dataset_multihead():
    """Asserts that `nc` is a list of class counts per task."""
    yaml_path = TMP / "multihead.yaml"
    data = {
        "path": str(TMP),
        "train": "images/train",
        "val": "images/val",
        "names": [["a0", "b0"], ["a1", "b1", "c1"]],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    (TMP / "images/train").mkdir(parents=True, exist_ok=True)
    (TMP / "images/val").mkdir(parents=True, exist_ok=True)
    parsed = check_det_dataset(yaml_path, autodownload=False)
    assert parsed["nc"] == [2, 3]

def test_check_det_dataset_singlehead():
    """Asserts that `nc` is a list with a single element."""
    yaml_path = TMP / "singlehead.yaml"
    data = {
        "path": str(TMP),
        "train": "images/train",
        "val": "images/val",
        "names": ["a", "b", "c"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    (TMP / "images/train").mkdir(parents=True, exist_ok=True)
    (TMP / "images/val").mkdir(parents=True, exist_ok=True)
    parsed = check_det_dataset(yaml_path, autodownload=False)
    assert parsed["nc"] == [3]


def test_yaml_model_load_multihead():
    """Model YAML now directly uses `nc` as a list of integers."""
    yaml_path = TMP / "model_multi.yaml"
    data = {
        "nc": [2, 1],
        "names": [["a", "b"], ["c"]],
        "backbone": [],
        "head": []
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    parsed = yaml_model_load(yaml_path)
    assert parsed["nc"] == [2, 1]


def test_yaml_model_load_singlehead():
    """Model YAML `nc` is an int, and the loader preserves it as an int."""
    yaml_path = TMP / "model_single.yaml"
    data = {
        "nc": 2,
        "names": ["a", "b"],
        "backbone": [],
        "head": []
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    parsed = yaml_model_load(yaml_path)
    assert parsed["nc"] == 2


def test_non_max_suppression_single_and_multihead():
    """Validating the actual output shape for multi-head NMS and single-head."""
    pred_single = torch.rand(1, 4 + 2, 10)
    out_single = ops.non_max_suppression(pred_single, nc=[2])
    if len(out_single[0]) > 0:
        assert out_single[0].shape[1] == 6

    pred_multi = torch.rand(1, 4 + 2 + 2, 10)
    out_multi = ops.non_max_suppression(pred_multi, nc=[2, 2])
    if len(out_multi[0]) > 0:
        assert out_multi[0].shape[1] == 8


def test_detect_init_and_forward_multihead():
    """Directly testing the `pre_forward` method for channel calculation."""
    ch = [8, 8, 8]
    nc_list = [2, 1]
    m = head.Detect(nc=nc_list, ch=ch)
    x = [torch.randn(1, c, 4, 4) for c in ch]
    training_features = m.pre_forward(x)
    # Expected channels = (4 * reg_max for bbox) + (nc_task1) + (nc_task2)
    expected_no = m.reg_max * 4 + sum(nc_list)
    assert training_features[0].shape[1] == expected_no


def test_detect_init_and_forward_singlehead():
    """Directly testing the `pre_forward` method for channel calculation."""
    ch = [8, 8, 8]
    nc_list = [3]
    m = head.Detect(nc=nc_list, ch=ch)
    x = [torch.randn(1, c, 4, 4) for c in ch]
    training_features = m.pre_forward(x)
    expected_no = m.reg_max * 4 + sum(nc_list)
    assert training_features[0].shape[1] == expected_no


class DummyModel(torch.nn.Module):
    def __init__(self, nc=None):
        super().__init__()
        ch = [8, 8, 8]
        self.model = torch.nn.ModuleList([head.Detect(nc=nc, ch=ch)])
        self.model[-1].stride = torch.tensor([8, 16, 32])
        self.args = SimpleNamespace(box=1.0, cls=1.0, dfl=1.0)
        self.nc = nc


def test_v8_detection_loss_multihead():
    model = DummyModel(nc=[2, 1])
    crit = loss.v8DetectionLoss(model)
    x = [torch.randn(1, 8, 4, 4) for _ in range(3)]
    preds = model.model[-1].pre_forward(x)
    batch = {"batch_idx": torch.tensor([0]), "cls": torch.tensor([[0, 0]]), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
    total, items = crit(preds, batch)
    assert total >= 0


def test_v8_detection_loss_singlehead():
    model = DummyModel(nc=[2])
    crit = loss.v8DetectionLoss(model)
    x = [torch.randn(1, 8, 4, 4) for _ in range(3)]
    preds = model.model[-1].pre_forward(x)
    batch = {"batch_idx": torch.tensor([0]), "cls": torch.tensor([[1]]), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
    total, items = crit(preds, batch)
    assert total >= 0


def create_sample_dataset(num_images=5, image_size=(64, 64), multihead=True):
    root = TMP / ("multihead_data" if multihead else "singlehead_data")
    shutil.rmtree(root, ignore_errors=True)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for i in range(num_images):
            img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
            for _ in range(2):
                w, h = rng.uniform(0.2, 0.4, size=2)
                xc, yc = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
                if multihead:
                    c0, c1 = rng.integers(0, 2), rng.integers(0, 3)
                    label = [c0, c1, xc, yc, w, h]
                else:
                    c = rng.integers(0, 2)
                    label = [c, xc, yc, w, h]
                with open(root / "labels" / split / f"{i}.txt", "a", encoding="utf-8") as f:
                    f.write(" ".join(map(str, label)) + "\n")
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)

    yaml_path = root / "dataset.yaml"
    names = [["a0", "b0"], ["a1", "b1", "c1"]] if multihead else ["a0", "b0"]
    yaml.safe_dump(
        {"path": str(root), "train": "images/train", "val": "images/val", "names": names},
        open(yaml_path, "w", encoding="utf-8"),
    )
    return root, yaml_path, names