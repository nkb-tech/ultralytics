# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops, LOGGER, colorstr

from ultralytics.models.yolo.detect.sahi_predict import slice_image, SAHIPredictAggregator

POSTPROCESS_REGISTRY = {
    "nms": ops.non_max_suppression,
    "nmm": ops.non_max_merging,
    "greedy_nmm": ops.non_max_merging,
}
FINAL_POSTPROCESS_MAX_WH = 7680


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DetectionPredictor with SAHI support."""
        super().__init__(cfg, overrides, _callbacks)

        self._set_postprocess_preferences()
        # SAHI support
        self.sahi_enabled = getattr(self.args, 'sahi', False)
        self.sahi_aggregator = None
        # Cache detection head metadata
        self._head_cache = None
        self._agnostic_cache = None
        
        if self.sahi_enabled:
            crop_size = getattr(self.args, 'crop_size', 640)
            overlap_ratio = getattr(self.args, 'overlap_ratio', 0.2)
            self.sahi_aggregator = SAHIPredictAggregator(crop_size=crop_size, overlap_ratio=overlap_ratio)
            LOGGER.info(f"SAHI inference enabled: crop_size={crop_size}, overlap_ratio={overlap_ratio}")

    def _results_names(self):
        """
        Normalize model names so Results.verbose can index them safely.
        Always return a list whose first element is an indexable mapping/list of names.
        """
        names = self.model.names
        head, _ = self._get_detection_head_meta()
        # Resolve number of classes from detection head
        try:
            nc = int(head.nc[0] if isinstance(head.nc, (list, tuple)) else head.nc)
        except Exception:
            nc = None

        def ensure_len(name_list):
            """Pad/trim to nc if known to avoid index errors."""
            if nc is None:
                return name_list
            if len(name_list) < nc:
                name_list = name_list + [f"class_{i}" for i in range(len(name_list), nc)]
            elif len(name_list) > nc:
                name_list = name_list[:nc]
            return name_list

        # Mapping -> convert to indexable list
        if isinstance(names, dict):
            try:
                keys = [int(k) for k in names.keys()]
                max_k = max(keys) if keys else -1
                mapped = [''] * (max_k + 1)
                for k, v in names.items():
                    mapped[int(k)] = v
                return [ensure_len(mapped)]
            except Exception:
                return [ensure_len(list(names.values()))]

        if isinstance(names, (list, tuple)):
            if names and isinstance(names[0], (list, tuple, dict)):
                # Already nested (e.g., multi-task)
                return list(names)
            return [ensure_len(list(names))]

        # Unknown type: best effort wrap
        return [ensure_len([str(names)])]

    def _set_postprocess_preferences(self):
        """Resolve which duplicate-removal routine should be used for this predictor."""
        key = getattr(self.args, "postprocess", "nms")
        if key not in POSTPROCESS_REGISTRY:
            LOGGER.warning(
                f"Unsupported postprocess='{key}', falling back to 'nms'. "
                f"Available options: {', '.join(POSTPROCESS_REGISTRY)}"
            )
            key = "nms"
        self.postprocess_type = key
        self.postprocess_fn = POSTPROCESS_REGISTRY[key]

    def _get_detection_head_meta(self):
        """Return detection head along with the current agnostic flag (cached)."""
        if self._head_cache is None:
            self._head_cache = self.model.model.model[-1]
            is_multitask = isinstance(self._head_cache.nc, (list, tuple)) and len(self._head_cache.nc) > 1
            self._agnostic_cache = self.args.agnostic_nms or is_multitask
        return self._head_cache, self._agnostic_cache

    def _apply_detection_postprocess(self, preds, agnostic, nc):
        """Apply the selected duplicate-removal routine to raw model predictions."""
        return self.postprocess_fn(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=agnostic,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=nc,
        )

    def _empty_prediction_tensor(self):
        """Utility helper that returns an empty predictions tensor on the active device."""
        device = self.device if self.device is not None else torch.device("cpu")
        return torch.empty((0, 6), dtype=torch.float32, device=device)

    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Streams real-time inference with SAHI support."""
        if self.sahi_enabled:
            return self._sahi_stream_inference(source, model, *args, **kwargs)
        else:
            return super().stream_inference(source, model, *args, **kwargs)

    def _sahi_stream_inference(self, source=None, model=None, *args, **kwargs):
        """SAHI-enabled stream inference."""

        if self.args.verbose:
            LOGGER.info("")

        with self._lock:
            self._prepare_sahi_stream(source)

            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                self.sahi_aggregator.reset()

                aggregated_predictions = self._process_batch_sahi(paths, im0s, *args, **kwargs)

                # Формируем Results
                self._create_sahi_results(aggregated_predictions, im0s, paths)

                # Визуализация и сохранение
                self._finalize_sahi_results(s, paths)

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

            self._print_summary()
            self.run_callbacks("on_predict_end")

    def _prepare_sahi_stream(self, source):
        """Initial stream setup and warmup."""
        self.setup_source(source if source is not None else self.args.source)

        # Prepare save dir
        if self.args.save or self.args.save_txt:
            (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

        # Warmup
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs,
                                     3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch = 0, [], None

        # Profilers
        self.profilers = (
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
        )

        self.run_callbacks("on_predict_start")

    def _process_batch_sahi(self, paths, im0s, *args, **kwargs):
        aggregated_predictions = []
        for img_idx, (im0, img_path) in enumerate(zip(im0s, paths)):
            pred = self._process_single_image_sahi(im0, img_idx, *args, **kwargs)
            aggregated_predictions.append(pred)
        return aggregated_predictions

    def _process_single_image_sahi(self, im0, img_idx, *args, **kwargs):
        orig_shape = im0.shape[:2]

        with self.profilers[0]:
            crops = slice_image(im0, self.sahi_aggregator.crop_size,
                                self.sahi_aggregator.overlap_ratio)
            img_key = str(self.seen + img_idx)

        for crop_im, crop_coords in crops:
            self._process_single_crop(img_key, crop_im, crop_coords, *args, **kwargs)

        return self._finalize_sahi_predictions(img_key, im0, orig_shape, *args, **kwargs)

    def _process_single_crop(self, img_key, crop_im, crop_coords, *args, **kwargs):
        crop_im_batch = [crop_im]
        im = self.preprocess(crop_im_batch)

        # Inference
        with self.profilers[1]:
            preds = self.inference(im, *args, **kwargs)
            if self.args.embed:
                return

        # Postprocess
        with self.profilers[2]:
            crop_preds = self._postprocess_crop_predictions(preds, crop_im.shape[:2], im.shape[2:])
            if crop_preds.shape[0] == 0:
                return
            self.sahi_aggregator.add_crop_predictions(img_key, crop_preds, crop_coords)

    def _postprocess_crop_predictions(self, preds, crop_hw, model_hw):
        """
        Apply standard NMS to crop predictions and map them back to the crop reference frame.

        Args:
            preds (torch.Tensor): Raw detector output for the crop.
            crop_hw (Tuple[int, int]): Crop height/width.
            model_hw (Tuple[int, int]): Model input height/width.

        Returns:
            torch.Tensor: Filtered predictions in crop coordinates.
        """
        head, agnostic = self._get_detection_head_meta()
        # Tiles always use classic NMS regardless of the final deduplication mode.
        crop_preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=agnostic,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=head.nc,
        )[0]
        
        # Scale boxes from model to crop coordinates
        if crop_preds.shape[0] > 0:
            crop_preds[:, :4] = ops.scale_boxes(model_hw, crop_preds[:, :4], crop_hw)
        return crop_preds

    def _finalize_sahi_predictions(self, img_key, im0, orig_shape, *args, **kwargs):
        with self.profilers[2]:
            tile_preds = self.sahi_aggregator.aggregate_predictions(
                img_key, orig_shape, conf_threshold=self.args.conf
            )
            full_image_preds = self._run_full_image_prediction(im0, *args, **kwargs)
            combined = self._combine_predictions(tile_preds, full_image_preds)

            if combined.shape[0] == 0:
                return combined

            _, agnostic = self._get_detection_head_meta()
            return self._deduplicate_final_boxes(combined, agnostic)

    def _run_full_image_prediction(self, im0, *args, **kwargs):
        """Run a standard forward pass on the full-resolution image."""
        im = self.preprocess([im0])
        with self.profilers[1]:
            preds = self.inference(im, *args, **kwargs)
            if self.args.embed:
                return self._empty_prediction_tensor()

        head, agnostic = self._get_detection_head_meta()
        processed = self._apply_detection_postprocess(preds, agnostic, head.nc)[0]

        if processed.shape[0] == 0:
            return processed

        model_h, model_w = im.shape[2:]
        processed[:, :4] = ops.scale_boxes((model_h, model_w), processed[:, :4], im0.shape[:2])
        return processed

    def _combine_predictions(self, tile_preds, full_image_preds):
        """Concatenate tile-level and full-image predictions onto a single tensor."""
        # Fast path: both empty#FIXME
        if (tile_preds is None or tile_preds.shape[0] == 0) and \
           (full_image_preds is None or full_image_preds.shape[0] == 0):
            return self._empty_prediction_tensor()
        
        # Fast path: only one has predictions
        if tile_preds is None or tile_preds.shape[0] == 0:
            return full_image_preds
        if full_image_preds is None or full_image_preds.shape[0] == 0:
            return tile_preds
        
        # Both have predictions - concatenate
        return torch.cat([tile_preds, full_image_preds], dim=0)

    def _deduplicate_final_boxes(self, preds, agnostic):
        """Apply the requested duplicate-removal routine on fused SAHI predictions."""
        if self.postprocess_type == "nms":
            return self._apply_final_nms(preds, agnostic)
        
        greedy = self.postprocess_type == "greedy_nmm"
        return self._apply_final_merge(preds, agnostic, greedy)

    def _apply_final_nms(self, preds, agnostic):
        """Final NMS step operating directly on [xyxy, conf, cls] tensors."""
        if preds.shape[0] == 0:
            return preds
        head, _ = self._get_detection_head_meta()
        detection_preds = preds.unsqueeze(0)
        merged = ops.non_max_suppression(
            detection_preds,
            self.args.conf,
            self.args.iou,
            agnostic=agnostic,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=head.nc,
            force_nms_for_e2e=True,  # Force NMS for SAHI final deduplication
        )[0]
        return merged

    def _apply_final_merge(self, preds, agnostic, greedy):
        """Use merge-style suppression on already decoded detections."""
        if preds.shape[0] == 0:
            return preds
        
        # Use specialized merge function for already-decoded detections
        merge_mode = "greedy" if greedy else "full"
        head, _ = self._get_detection_head_meta()
        detection_preds = preds.unsqueeze(0)
        merged = ops.non_max_merging(
            detection_preds,
            self.args.conf,
            self.args.iou,
            agnostic=agnostic,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=head.nc,
            merge_mode=merge_mode,
            force_nmm_for_e2e=True,  # Force NMM for SAHI final deduplication
        )[0]
        return merged

    def _create_sahi_results(self, aggregated_preds_list, im0s, paths):
        im0s_list = ops.convert_torch2numpy_batch(im0s) if not isinstance(im0s, list) else im0s
        self._sahi_orig_imgs = im0s_list
        names = self._results_names()

        self.results = []
        for pred, orig_img, img_path in zip(aggregated_preds_list, im0s_list, paths):
            self.results.append(Results(orig_img, path=img_path,
                                        names=names, boxes=pred))

        self.run_callbacks("on_predict_postprocess_end")

    def _finalize_sahi_results(self, s, paths):
        n = len(self.results)

        for i in range(n):
            self.seen += 1
            self.results[i].speed = {
                "preprocess": self.profilers[0].dt * 1e3 / n,
                "inference": self.profilers[1].dt * 1e3 / n,
                "postprocess": self.profilers[2].dt * 1e3 / n,
            }

            if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                orig_img = self._sahi_orig_imgs[i]
                orig_tensor = torch.from_numpy(orig_img).permute(2, 0, 1)\
                    .unsqueeze(0).float().to(self.device) / 255.0

                s[i] += self.write_results(i, Path(paths[i]), orig_tensor, s)

        if self.args.verbose:
            LOGGER.info("\n".join(s))


    def _print_summary(self):
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in self.profilers)
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at "
                f"shape {(min(self.args.batch, self.seen), 3, *self.imgsz)}" % t
            )

        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" \
                if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
            
    # For standar inference without Sahi
    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        if not self.nms:
            m = self.model.model.model[-1]  # detect head
            is_multitask = isinstance(m.nc, (list, tuple)) and len(m.nc) > 1
            agnostic = self.args.agnostic_nms or is_multitask
            preds = self._apply_detection_postprocess(preds, agnostic, m.nc)
        elif self.engine:
            preds = ops.process_nms_trt_results(preds, self.output_names)
        elif self.onnx:
            preds = ops.process_nms_onnx_results(preds)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self._results_names(), boxes=pred))
        return results
