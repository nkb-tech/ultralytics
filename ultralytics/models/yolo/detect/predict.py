# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr, nms, ops

from ultralytics.models.yolo.detect.sahi_predict import slice_image, SAHIPredictAggregator


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize DetectionPredictor with SAHI support."""
        super().__init__(cfg, overrides, _callbacks)

        # SAHI support
        self.sahi_enabled = getattr(self.args, 'sahi', False)
        self.sahi_aggregator = None
        if self.sahi_enabled:
            crop_size = getattr(self.args, 'crop_size', 640)
            overlap_ratio = getattr(self.args, 'overlap_ratio', 0.2)
            self.sahi_aggregator = SAHIPredictAggregator(crop_size=crop_size, overlap_ratio=overlap_ratio)
            LOGGER.info(f"SAHI inference enabled: crop_size={crop_size}, overlap_ratio={overlap_ratio}")

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

        return self._aggregate_final_predictions(img_key, orig_shape)

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
            if not self.nms:
                m = self.model.model.model[-1]
                is_multitask = isinstance(m.nc, (list, tuple)) and len(m.nc) > 1
                agnostic = self.args.agnostic_nms or is_multitask

                crop_preds = nms.non_max_suppression(
                    preds, self.args.conf, self.args.iou,
                    agnostic=agnostic,
                    max_det=self.args.max_det,
                    classes=self.args.classes,
                    nc=m.nc,
                )[0]

                if len(crop_preds) == 0:
                    return

                crop_h, crop_w = crop_im.shape[:2]
                model_h, model_w = im.shape[2:]

                crop_preds[:, :4] = ops.scale_boxes(
                    (model_h, model_w), crop_preds[:, :4], (crop_h, crop_w)
                )

                crop_preds_xywh = crop_preds.clone()
                crop_preds_xywh[:, :4] = ops.xyxy2xywh(crop_preds[:, :4])

                self.sahi_aggregator.add_crop_predictions(
                    img_key, crop_preds_xywh, crop_coords
                )

    def _aggregate_final_predictions(self, img_key, orig_shape):
        with self.profilers[2]:
            aggregated = self.sahi_aggregator.aggregate_predictions(
                img_key, orig_shape, conf_threshold=0.001
            )

            if len(aggregated) == 0:
                return aggregated

            aggregated_xyxy = ops.xywh2xyxy(aggregated[:, :4])
            aggregated_for_nms = torch.cat([aggregated_xyxy, aggregated[:, 4:]], dim=1)
            aggregated_for_nms = aggregated_for_nms.unsqueeze(0)

            m = self.model.model.model[-1]
            is_multitask = isinstance(m.nc, (list, tuple)) and len(m.nc) > 1
            agnostic = self.args.agnostic_nms or is_multitask

            final = nms.non_max_suppression(
                aggregated_for_nms,
                self.args.conf,
                self.args.iou,
                agnostic=agnostic,
                max_det=self.args.max_det,
                classes=self.args.classes,
                nc=m.nc,
            )[0]

            return final

    def _create_sahi_results(self, aggregated_preds_list, im0s, paths):
        im0s_list = ops.convert_torch2numpy_batch(im0s) if not isinstance(im0s, list) else im0s
        self._sahi_orig_imgs = im0s_list

        self.results = []
        for pred, orig_img, img_path in zip(aggregated_preds_list, im0s_list, paths):
            self.results.append(Results(orig_img, path=img_path,
                                        names=self.model.names, boxes=pred))

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
                # Normalize based on bit depth from config
                bit_depth = getattr(self.args, 'image_bit_depth', 8)
                if bit_depth == 8:
                    norm_divisor = 255.0
                elif bit_depth == 16:
                    norm_divisor = 65_535.0
                else:
                    LOGGER.error(f"BitDepth {bit_depth} unsupported.")
                norm_divisor = 65_535.0 if bit_depth == 16 else 255.0
                orig_tensor = torch.from_numpy(orig_img).permute(2, 0, 1) \
                    .unsqueeze(0).float().to(self.device) / norm_divisor

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
    def postprocess(self, preds, img, orig_imgs, nc: list[int] = [80]):
        """Post-processes predictions and returns a list of Results objects."""

        nhwc = getattr(self.model, "nhwc", False)
        end2end = getattr(self.model, "end2end", False)
        img_hw = tuple(int(i) for i in (img.shape[1:3] if nhwc else img.shape[2:4]))

        if self.nms: # nms inside the graph
            if self.engine:
                preds = ops.process_nms_trt_results(preds, self.output_names)
            elif self.onnx:
                preds = ops.process_nms_onnx_results(preds)
        else:
            if self.rknn:
                if not end2end:
                    preds = ops.process_rknn_dfl_results(
                        input_data=preds,
                        imgsz=img_hw,
                        conf_thres=self.args.conf,
                    )
                else:
                    preds = ops.process_rknn_end2end_results(
                        input_data=preds,
                        imgsz=img_hw,
                        conf_thres=self.args.conf,
                        nc=self.nc,
                    )

            agnostic = self.args.agnostic_nms or self.is_multitask
            preds = nms.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                agnostic=agnostic,
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
            # Handle 16-bit single-channel images (e.g., X-ray images)
            # Convert 16-bit to 8-bit and triple single channel to 3 channels
            if orig_img.dtype == np.uint16:
                orig_img = (orig_img >> 8).astype(np.uint8)
            if orig_img.ndim == 2:
                orig_img = orig_img[..., None]
            if orig_img.shape[2] == 1:
                orig_img = np.repeat(orig_img, 3, axis=2)
            pred[:, :4] = ops.scale_boxes(img_hw, pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results