import yaml
import torch
from types import SimpleNamespace
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import DetMetrics, ConfusionMatrix
from tests import TMP


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
