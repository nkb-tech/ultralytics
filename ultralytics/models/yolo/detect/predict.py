# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr, nms, ops
from ultralytics.utils.metrics import box_intersection
from ultralytics.models.yolo.detect.sahi_predict import slice_image, SAHIPredictAggregator


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DetectionPredictor with optional SAHI support."""
        super().__init__(cfg, overrides, _callbacks)

        self.sahi_enabled = getattr(self.args, "sahi", False)
        self.sahi_aggregator = None
        if self.sahi_enabled:
            crop_size = getattr(self.args, "crop_size", 640)
            overlap_ratio = getattr(self.args, "overlap_ratio", 0.2)
            self.sahi_aggregator = SAHIPredictAggregator(crop_size, overlap_ratio)
            LOGGER.info(f"SAHI inference enabled: crop_size={crop_size}, overlap_ratio={overlap_ratio}, {self.args.nms_strategy} NMS.")

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        nhwc = getattr(self.model, "nhwc", False)
        img_hw = tuple(int(i) for i in (img.shape[1:3] if nhwc else img.shape[2:4]))

        if getattr(self.model, "nms", False):
            output_names = sorted(self.model.output_names) if hasattr(self.model, "output_names") else None
            if getattr(self.model, "engine", False):
                preds = ops.process_nms_trt_results(preds, output_names)
            elif getattr(self.model, "onnx", False):
                preds = ops.process_nms_onnx_results(preds)
            elif getattr(self.model, "hef", False):
                preds = ops.process_nms_hef_results(preds, img_hw=img_hw)
        else:
            preds = self._decode_backend_preds(preds, img_hw)
            preds = nms.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms or self.is_multitask,
                nc=self.nc,
                max_det=self.args.max_det,
                classes=self.args.classes,
                end2end=getattr(self.model, "end2end", False),
                rotated=self.args.task == "obb",
            )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            if orig_img.dtype == np.uint16:
                orig_img = (orig_img >> 8).astype(np.uint8)
            if orig_img.ndim == 2:
                orig_img = orig_img[..., None]
            if orig_img.shape[2] == 1:
                orig_img = np.repeat(orig_img, 3, axis=2)
            pred[:, :4] = ops.scale_boxes(img_hw, pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference, routing to SAHI when enabled."""
        if self.sahi_enabled:
            return self._sahi_stream_inference(source, model, *args, **kwargs)
        return super().stream_inference(source, model, *args, **kwargs)

    def _sahi_stream_inference(self, source=None, model=None, *args, **kwargs):
        """Full-image + crop inference with merge deduplication."""
        if self.args.verbose:
            LOGGER.info("")

        with self._lock:
            self._setup_sahi(source)

            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                self.sahi_aggregator.reset()

                preds_list = [self._sahi_single_image(im0, idx, *args, **kwargs)
                              for idx, im0 in enumerate(im0s)]

                self._build_results(preds_list, im0s, paths, s)
                self.run_callbacks("on_predict_batch_end")
                yield from self.results

            self._finalize()
            self.run_callbacks("on_predict_end")

    def _setup_sahi(self, source):
        """One-time stream setup: source, save dirs, warmup, profilers."""
        self.setup_source(source if source is not None else self.args.source)
        if self.args.save or self.args.save_txt:
            (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True
        self.seen, self.windows, self.batch = 0, [], None
        self.profilers = tuple(ops.Profile(device=self.device) for _ in range(3))
        self.run_callbacks("on_predict_start")

    def _sahi_single_image(self, im0, img_idx, *args, **kwargs):
        """Run full-image + crop inference on one image and merge results."""
        orig_shape = im0.shape[:2]
        img_key = str(self.seen + img_idx)

        with self.profilers[0]:
            crops = slice_image(im0, self.sahi_aggregator.crop_size, self.sahi_aggregator.overlap_ratio)

        full_preds = self._infer_single(im0, *args, **kwargs)

        for crop_im, crop_coords in crops:
            crop_preds = self._infer_single(crop_im, *args, **kwargs)
            if crop_preds is not None:
                self.sahi_aggregator.add_crop_predictions(img_key, crop_preds, crop_coords)

        return self._merge_predictions(img_key, orig_shape, full_preds)

    def _infer_single(self, image, *args, **kwargs):
        """Preprocess → inference → NMS → scale_boxes for one image/crop. Returns None if no detections."""
        im = self.preprocess([image])

        with self.profilers[1]:
            preds = self.inference(im, *args, **kwargs)
            if self.args.embed:
                return None

        with self.profilers[2]:
            nhwc = getattr(self.model, "nhwc", False)
            img_hw = tuple(int(i) for i in (im.shape[1:3] if nhwc else im.shape[2:4]))

            if getattr(self.model, "nms", False):
                output_names = sorted(self.model.output_names) if hasattr(self.model, "output_names") else None
                if getattr(self.model, "engine", False):
                    result = ops.process_nms_trt_results(preds, output_names)[0]
                elif getattr(self.model, "onnx", False):
                    result = ops.process_nms_onnx_results(preds)[0]
                elif getattr(self.model, "hef", False):
                    result = ops.process_nms_hef_results(preds, img_hw=img_hw)[0]
                else:
                    return None
            else:
                preds = self._decode_backend_preds(preds, img_hw)
                result = nms.non_max_suppression(
                    preds, self.args.conf, self.args.iou,
                    agnostic=self.args.agnostic_nms or self.is_multitask,
                    max_det=self.args.max_det, classes=self.args.classes, nc=self.nc,
                )[0]

            if len(result) == 0:
                return None

            result[:, :4] = ops.scale_boxes(img_hw, result[:, :4], image.shape[:2])
            return result

    def _merge_predictions(self, img_key, orig_shape, full_preds):
        """Merge full-image and crop predictions.

        Full-image detections are kept as-is (correct coordinates for large objects).
        Crop detections are only added if they aren't already covered by a full-image
        detection (IOS < 0.5), then deduplicated among themselves via NMS.
        """
        with self.profilers[2]:
            crop_preds = self.sahi_aggregator.aggregate_predictions(img_key, orig_shape)

            if full_preds is None and len(crop_preds) == 0:
                return crop_preds
            if full_preds is None:
                return self._nms(crop_preds)
            if len(crop_preds) == 0:
                return full_preds

            # Filter out crop detections that are mostly inside a full-image detection
            inter = box_intersection(crop_preds[:, :4], full_preds[:, :4])
            crop_areas = ((crop_preds[:, 2] - crop_preds[:, 0]) *
                          (crop_preds[:, 3] - crop_preds[:, 1])).clamp(min=1e-6)
            max_ios = (inter / crop_areas.unsqueeze(1)).max(dim=1)[0]
            novel = crop_preds[max_ios < 0.5]

            if len(novel) == 0:
                return full_preds

            return torch.cat([full_preds, self._nms(novel)], dim=0)

    def _nms(self, preds):
        """Run NMS on post-processed predictions."""
        nms_strategy = getattr(self.args, "nms_strategy", "usual")
        return nms.non_max_suppression(
            preds.unsqueeze(0), self.args.conf, self.args.iou,
            agnostic=self.args.agnostic_nms or self.is_multitask,
            max_det=self.args.max_det, classes=self.args.classes,
            nc=self.nc, nms_strategy=nms_strategy,
        )[0]

    def _build_results(self, preds_list, im0s, paths, s):
        """Build Results objects and write output for a batch."""
        im0s_list = ops.convert_torch2numpy_batch(im0s) if not isinstance(im0s, list) else im0s
        self._orig_imgs = im0s_list

        self.results = [Results(img, path=p, names=self.model.names, boxes=pred)
                        for pred, img, p in zip(preds_list, im0s_list, paths)]
        self.run_callbacks("on_predict_postprocess_end")

        n = len(self.results)
        for i in range(n):
            self.seen += 1
            self.results[i].speed = {k: self.profilers[j].dt * 1e3 / n
                                     for j, k in enumerate(("preprocess", "inference", "postprocess"))}
            if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                orig_img = self._orig_imgs[i]
                bit_depth = getattr(self.args, "image_bit_depth", 8)
                norm = 65_535.0 if bit_depth == 16 else 255.0
                orig_t = torch.from_numpy(orig_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / norm
                s[i] += self.write_results(i, Path(paths[i]), orig_t, s)

        if self.args.verbose:
            LOGGER.info("\n".join(s))

    def _finalize(self):
        """Release video writers and print speed summary."""
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in self.profilers)
            LOGGER.info("Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at "
                        "shape %s" % (*t, (min(self.args.batch, self.seen), 3, *self.imgsz)))
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _decode_backend_preds(self, preds, img_hw):
        """Decode RKNN/HEF backend-specific prediction formats."""
        end2end = getattr(self.model, "end2end", False)
        if getattr(self.model, "rknn", False):
            if not end2end:
                preds = ops.process_rknn_dfl_results(input_data=preds, imgsz=img_hw, conf_thres=self.args.conf)
            else:
                preds = ops.process_rknn_end2end_results(input_data=preds, imgsz=img_hw, conf_thres=self.args.conf, nc=self.nc)
        elif getattr(self.model, "hef", False):
            if not end2end:
                preds = ops.process_hef_dfl_results(input_data=preds, imgsz=img_hw, conf_thres=self.args.conf)
            else:
                preds = ops.process_hef_end2end_results(input_data=preds, imgsz=img_hw, conf_thres=self.args.conf, nc=self.nc)
        return preds
