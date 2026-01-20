# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import LOGGER, ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolov8n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, nc: list[int] = [80]):
        """Post-processes predictions and returns a list of Results objects."""

        nhwc = getattr(self.model, "nhwc", False)
        img_hw = tuple(int(i) for i in (img.shape[1:3] if nhwc else img.shape[2:4]))

        if self.nms: # nms inside the graph
            if self.engine:
                preds = ops.process_nms_trt_results(preds, self.output_names)
            elif self.onnx:
                preds = ops.process_nms_onnx_results(preds)
        else:
            if self.rknn:
                preds = ops.process_rknn_dfl_results(
                    input_data=preds,
                    imgsz=img_hw,
                    conf_thres=self.args.conf,
                )

            agnostic = self.args.agnostic_nms or self.is_multitask
            preds = ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=agnostic,
                nc=self.nc,
                max_det=self.args.max_det,
                classes=self.args.classes,
            )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img_hw, pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results