import yaml
import torch
import shutil
from types import SimpleNamespace
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops, loss
from ultralytics.utils.metrics import DetMetrics, ConfusionMatrix
from tests import TMP
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.nn.modules import head


def validate_prediction_format(results, num_heads):
    b = results[0].boxes
    assert b.data.shape[1] == 4 + num_heads * 2
    assert b.conf.min() >= 0 and b.conf.max() <= 1
    if num_heads > 1:
        assert b.cls2 is not None and b.conf2 is not None


def test_check_det_dataset_multihead():
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
    assert parsed["nc_per_task"] == [2, 3]
    assert parsed["nc"] == 5
    assert len(parsed["names_per_task"]) == 2


def test_check_det_dataset_singlehead():
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
    assert "nc_per_task" not in parsed
    assert parsed["nc"] == 3


def test_yaml_model_load_multihead():
    yaml_path = TMP / "model_multi.yaml"
    data = {
        "names": [["a", "b"], ["c"]],
        "backbone": [],
        "head": []
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    parsed = yaml_model_load(yaml_path)
    assert parsed["num_classes_per_head"] == [2, 1]
    assert parsed["nc"] == 3


def test_yaml_model_load_singlehead():
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
    assert "num_classes_per_head" not in parsed
    assert parsed["nc"] == 2


def test_non_max_suppression_single_and_multihead():
    pred_single = torch.tensor(
        [
            [
                [10.0, 20.0],
                [5.0, 10.0],
                [5.0, 10.0],
                [10.0, 20.0],
                [0.9, 0.4],
                [0.1, 0.8],
            ]
        ]
    )
    out_single = ops.non_max_suppression(pred_single, nc=2)
    assert out_single[0].shape[1] == 6

    pred_multi = torch.tensor(
        [
            [
                [10.0, 20.0],
                [5.0, 10.0],
                [5.0, 10.0],
                [10.0, 20.0],
                [0.9, 0.4],
                [0.1, 0.8],
                [0.2, 0.1],
                [0.8, 0.3],
                [0.3, 0.6],
                [0.4, 0.1],
            ]
        ]
    )
    out_multi = ops.non_max_suppression(pred_multi, num_classes_per_head=[2, 2])
    assert out_multi[0].shape[1] == 8


def test_non_max_suppression_primary_only():
    # three boxes: one class0 and two class1 overlapping
    pred = torch.tensor(
        [
            [10.0, 20.0, 20.0],
            [10.0, 20.0, 20.0],
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [0.9, 0.8, 0.7],
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.9],
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
        ]
    ).unsqueeze(0)
    out = ops.non_max_suppression(pred, num_classes_per_head=[2, 2])
    assert out[0].shape[0] == 3

    out_full = ops.non_max_suppression(pred, num_classes_per_head=[2, 2], full_class_nms=True)
    assert out_full[0].shape[0] == 2


def test_process_batch_edge_cases():
    validator = DetectionValidator(args=SimpleNamespace())
    validator.device = torch.device("cpu")
    empty_pred = torch.zeros((0, 8))
    gt_box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_cls = torch.tensor([0])
    out = validator._process_batch(empty_pred, gt_box, gt_cls, task=0)
    assert out.shape == (0, validator.niou)

    det = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0, 0.8, 0.0]])
    out = validator._process_batch(det, torch.zeros((0, 4)), torch.zeros(0, dtype=torch.long), task=0)
    assert out.shape == (1, validator.niou)


def test_metrics_get_stats_multihead():
    args = SimpleNamespace(
        conf=0.25,
        iou=0.45,
        single_cls=False,
        agnostic_nms=False,
        plots=False,
        save_json=False,
        save_txt=False,
        half=False,
    )
    validator = DetectionValidator(args=args)
    validator.device = torch.device("cpu")
    validator.is_multihead = True
    validator.tasks = [{"nc": 1, "names": {0: "a"}}, {"nc": 1, "names": {0: "b"}}]
    validator.metrics = [DetMetrics(names=t["names"]) for t in validator.tasks]
    validator.confusion_matrix = [ConfusionMatrix(nc=t["nc"], conf=args.conf) for t in validator.tasks]
    validator.stats = [dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[]) for _ in validator.tasks]
    det = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0, 0.8, 0.0]])
    gt_box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_cls = torch.tensor([0])
    for t in range(2):
        correct = validator._process_batch(det, gt_box, gt_cls, task=t)
        validator.stats[t]["tp"].append(correct)
        validator.stats[t]["conf"].append(det[:, 4 + 2 * t])
        validator.stats[t]["pred_cls"].append(det[:, 5 + 2 * t])
        validator.stats[t]["target_cls"].append(gt_cls)
        validator.stats[t]["target_img"].append(gt_cls)
    results = validator.get_stats()
    assert all(k.startswith("task0_") or k.startswith("task1_") for k in results)


def test_detect_init_and_forward_multihead():
    ch = [8, 8, 8]
    m = head.Detect(ch=ch, num_classes_per_head=[2, 1])
    m.eval()
    x = [torch.randn(1, c, 4, 4) for c in ch]
    out, _ = m(x)
    assert out.shape[1] == 4 + sum(m.num_classes_per_head) + len(m.num_classes_per_head)


def test_detect_init_and_forward_singlehead():
    ch = [8, 8, 8]
    m = head.Detect(nc=3, ch=ch)
    m.eval()
    x = [torch.randn(1, c, 4, 4) for c in ch]
    out, _ = m(x)
    assert out.shape[1] == 4 + m.nc


class DummyModel(torch.nn.Module):
    def __init__(self, num_classes_per_head=None, nc=None):
        super().__init__()
        ch = [8, 8, 8]
        if num_classes_per_head is not None:
            self.model = torch.nn.ModuleList([head.Detect(ch=ch, num_classes_per_head=num_classes_per_head)])
        else:
            self.model = torch.nn.ModuleList([head.Detect(nc=nc, ch=ch)])
        self.model[-1].stride = torch.tensor([8, 16, 32])
        self.args = SimpleNamespace(box=1.0, cls=1.0, dfl=1.0)


def test_v8_detection_loss_multihead():
    model = DummyModel(num_classes_per_head=[2, 1])
    crit = loss.v8DetectionLoss(model)
    x = [torch.randn(1, 8, 4, 4) for _ in range(3)]
    preds = model.model[-1](x)
    batch = {"batch_idx": torch.tensor([0]), "cls": torch.tensor([0]), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
    total, items = crit(preds, batch)
    assert total >= 0


def test_v8_detection_loss_singlehead():
    model = DummyModel(nc=2)
    crit = loss.v8DetectionLoss(model)
    x = [torch.randn(1, 8, 4, 4) for _ in range(3)]
    preds = model.model[-1](x)
    batch = {"batch_idx": torch.tensor([0]), "cls": torch.tensor([1]), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
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
            labels = []
            for _ in range(2):
                w, h = rng.uniform(0.2, 0.4, size=2)
                xc, yc = rng.uniform(w / 2, 1 - w / 2), rng.uniform(h / 2, 1 - h / 2)
                x1, y1 = int((xc - w / 2) * image_size[0]), int((yc - h / 2) * image_size[1])
                x2, y2 = int((xc + w / 2) * image_size[0]), int((yc + h / 2) * image_size[1])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), -1)
                if multihead:
                    c0 = rng.integers(0, 2)
                    c1 = rng.integers(0, 3)
                    labels.append([c0, c1, xc, yc, w, h])
                else:
                    c = rng.integers(0, 2)
                    labels.append([c, xc, yc, w, h])
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
            with open(root / "labels" / split / f"{i}.txt", "w", encoding="utf-8") as f:
                if multihead:
                    for c0, c1, xc, yc, w, h in labels:
                        line = f"{int(c0)} {int(c1)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                        f.write(line)
                else:
                    for c, xc, yc, w, h in labels:
                        line = f"{int(c)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
                        f.write(line)
    yaml_path = root / "dataset.yaml"
    names = [["a0", "b0"], ["a1", "b1", "c1"]] if multihead else ["a0", "b0"]
    yaml.safe_dump(
        {
            "path": str(root),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        },
        open(yaml_path, "w", encoding="utf-8"),
    )
    return root, yaml_path, names


import pytest


@pytest.mark.skip(reason="slow")
def test_multihead_training_pipeline():
    dataset_root, yaml_file, names = create_sample_dataset(num_images=20)
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    overrides = dict(model="yolov8n.yaml", data=str(yaml_file), epochs=1, imgsz=64, batch=1,
                     cache="ram", mosaic=0.0, copy_paste=0.0, mixup=0.0, verbose=False, val=False, save=False)
    trainer = DetectionTrainer(overrides=overrides)
    trainer.save_model = lambda *a, **k: None
    trainer.read_results_csv = lambda *a, **k: {}
    trainer.final_eval = lambda *a, **k: None
    trainer.validate = lambda *a, **k: ({}, 0)
    trainer.train()
    model = YOLO("yolov8n.yaml")
    model.model = trainer.model
    _ = model.predict(source=str(dataset_root / "images/val/0.jpg"), imgsz=64, verbose=False)
    model = YOLO("yolov8n.yaml")
    model.model = trainer.model
    res = model.predict(source=str(dataset_root / "images/val/0.jpg"), imgsz=64, verbose=False)
    validate_prediction_format(res, num_heads=len(names))


@pytest.mark.skip(reason="slow")
def test_singlehead_training_pipeline():
    dataset_root, yaml_file, _ = create_sample_dataset(num_images=20, multihead=False)
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    overrides = dict(model="yolov8n.yaml", data=str(yaml_file), epochs=1, imgsz=64, batch=1,
                     cache="ram", mosaic=0.0, copy_paste=0.0, mixup=0.0, verbose=False, val=False, save=False)
    trainer = DetectionTrainer(overrides=overrides)
    trainer.save_model = lambda *a, **k: None
    trainer.read_results_csv = lambda *a, **k: {}
    trainer.final_eval = lambda *a, **k: None
    trainer.validate = lambda *a, **k: ({}, 0)
    trainer.train()


