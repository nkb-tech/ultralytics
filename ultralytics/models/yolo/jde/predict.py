from ultralytics.engine.results import Results
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import DEFAULT_CFG, ops


class JDEPredictor(BasePredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "jde"

    def postprocess(self, preds, img, orig_imgs):
        # preds can be (y, preds_dict)
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=[len(self.model.names)],
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            # pred cant be empty
            if pred is not None and pred.shape[0]:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            r = Results(
                orig_img,
                path=img_path,
                names=self.model.names,
                boxes=pred[:, :6] if pred is not None else pred,
            )
            EMBED_DIM = 128
            r.embeds = pred[:, -EMBED_DIM:]
            results.append(r)

        return results