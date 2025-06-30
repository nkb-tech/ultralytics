from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ultralytics.utils import LOGGER, colorstr

from .augment import Compose, Format, LetterBox, crop_transforms, crop_val_transforms, v8_transforms
from .dataset import YOLODataset


class SAHIDataset(YOLODataset):  # only for bboxes, TODO: keypoints and masks
    """
    A dataset class that supports Slicing Aided Hyper Inference (SAHI) strategies for training and validation.

    This class extends the standard YOLODataset to allow slicing of images into smaller crops using configurable cutting
    strategies such as grid-based slicing or random cropping. It is useful for training models whose performance during
    inference with SAHI should be improved by exposing them to similar crop-based training samples.

    Attributes:
        cut_strategy (str): Strategy used to crop images. Options are:
            - 'grid': Slice image into a grid of fixed-size crops with optional overlap (similar to SAHI).
            - 'random_crop': Random crops from the image. The number of crops per image is determined by sampling_rate.
        crop_size (int): Size (width and height) of each crop in pixels.
        overlap_ratio (float): Overlap ratio between adjacent slices when using 'grid' strategy (0 to 1).
        sampling_rate (float): For 'random_crop' strategy, determines how many random crops to generate relative to the
                               number of grid slices an image would produce. E.g., 0.3 means 30% of the number of grid slices.
        use_slicing (bool): Whether slicing is being used (True if cut_strategy is 'grid').
        slice_indices (List[Tuple[int, int]]): Precomputed list of (image_index, slice_index) pairs for all slices.
        avg_slices (float): Average number of slices per image.
        min_slices (int): Minimum number of slices across all images.
        max_slices (int): Maximum number of slices across all images.

    Methods:
        _precompute_slices: Precomputes slice indices for all images using parallel processing.
        __len__: Returns the total number of slices if slicing is used, otherwise returns the number of original images.
        get_image_and_label: Retrieves a single processed image and its corresponding label.
        _get_grid_slice: Generates a single slice from an image using grid slicing.
        _filter_and_transform_annotations: Filters and transforms annotations to match the current slice.
        build_transforms: Builds data augmentation and preprocessing transformations based on the slicing strategy.
    """

    def __init__(
        self,
        img_path: str,
        cut_strategy: str = "grid",
        crop_size: int = 640,
        overlap_ratio: float = 0,
        sampling_rate: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initializes the SAHIDataset instance.

        Args:
            img_path (str): Path/paths to the directory containing images.
            cut_strategy (str): Strategy for slicing images ('grid' or 'random_crop'). Default is 'grid'.
            crop_size (int): Size of each crop (square). Default is 640.
            overlap_ratio (float): Fraction of overlap between adjacent slices (only for 'grid'). Default is 0.
            sampling_rate (float): When using 'random_crop', this defines the ratio of random crops per image compared to
                                   how many slices that image would generate under the 'grid' strategy. Default is 1.0.
            *args, **kwargs: Additional arguments passed to the parent YOLODataset class.
        """
        self.cut_strategy = cut_strategy
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.sampling_rate = sampling_rate
        self.use_slicing = cut_strategy == "grid"
        super().__init__(img_path=img_path, *args, **kwargs)

        # logging
        task = "train" if self.augment else "val"
        prefix = colorstr(f"SAHIDataset for {task}")
        info = [f":\n  cut_strategy: {self.cut_strategy}", f"\n  crop_size: {self.crop_size}"]
        # overlap_ratio only for grid
        if self.cut_strategy == "grid":
            info.append(f"\n  overlap_ratio: {self.overlap_ratio}")

        info.append(f"\n  total_images: {self.ni}")
        LOGGER.info(prefix + "".join(info))

        self.slice_indices = self._precompute_slices()  # Список (img_idx, slice_idx)
        crop_info = (
            f"  Total crops: {len(self.slice_indices)}\n"
            f"  Avg crops/image: {self.avg_slices:.1f}\n"
            f"  Min/Max crops/image: {self.min_slices}/{self.max_slices}"
        )
        LOGGER.info(crop_info)

    def _precompute_slices(self) -> List[Tuple[int, int]]:
        """
        Precompute (img_idx, slice_idx) pairs and collect statistics.
        If cut_strategy is 'random_crop', returns fewer slices per image based on sampling_rate.
        """

        def process_single_image(
            idx: int, crop_size: int, overlap_ratio: float, load_image_func: callable
        ) -> Tuple[int, int]:
            im, (h0, w0), _ = load_image_func(idx)

            step_x = int(crop_size * (1 - overlap_ratio))
            step_y = int(crop_size * (1 - overlap_ratio))

            cols = max(1, (w0 - crop_size) // step_x + 1)
            rows = max(1, (h0 - crop_size) // step_y + 1)

            total_slices = cols * rows
            return idx, total_slices

        with ThreadPoolExecutor() as executor:
            process_fn = partial(
                process_single_image,
                crop_size=self.crop_size,
                overlap_ratio=self.overlap_ratio,
                load_image_func=self.load_image,
            )
            results = list(executor.map(process_fn, range(self.ni)))

        slice_indices: List[Tuple[int, int]] = []
        slices_per_image: List[int] = []

        for idx, total_slices in results:
            if self.cut_strategy == "random_crop":
                sampled_slices = max(1, round(total_slices * self.sampling_rate))
                slice_indices.extend([(idx, s) for s in range(sampled_slices)])
                slices_per_image.append(sampled_slices)
            else:  # grid
                slice_indices.extend([(idx, s) for s in range(total_slices)])
                slices_per_image.append(total_slices)

        # Store statistics
        self.avg_slices = sum(slices_per_image) / len(slices_per_image)
        self.min_slices = min(slices_per_image)
        self.max_slices = max(slices_per_image)

        return slice_indices

    def __len__(self):
        return len(self.slice_indices)

    def get_image_and_label(self, index: int) -> Dict[str, Any]:
        if self.use_slicing:
            return self._get_grid_slice(index)
        else:
            img_idx, _ = self.slice_indices[index]
            label = deepcopy(self.labels[img_idx])
            label.pop("shape", None)
            label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(img_idx)
            label["ratio_pad"] = (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            )
            return self.update_labels_info(label)

    def _get_grid_slice(self, index: int) -> Dict[str, Any]:
        """
        Generate a single slice based on coordinates.

        Args:
            index (int): Index of the slice in `self.slice_indices`.

        Returns:
            Dict[str, Any]: Dictionary containing sliced image and filtered labels.
        """
        img_idx, slice_idx = self.slice_indices[index]
        im, (h0, w0), _ = self.load_image(img_idx)
        labels = deepcopy(self.labels[img_idx])

        step_x = int(self.crop_size * (1 - self.overlap_ratio))
        step_y = int(self.crop_size * (1 - self.overlap_ratio))
        cols = max(1, (w0 - self.crop_size) // step_x + 1)

        row = slice_idx // cols
        col = slice_idx % cols

        start_x = col * step_x
        start_y = row * step_y

        end_x = min(start_x + self.crop_size, w0)
        end_y = min(start_y + self.crop_size, h0)
        slice_im = im[start_y:end_y, start_x:end_x]
        slice_bbox = [start_x, start_y, end_x, end_y]

        slice_labels = self._filter_and_transform_annotations(labels, slice_bbox, h0, w0)

        labels.update(
            {
                "img": slice_im,
                "ori_shape": (h0, w0),
                "resized_shape": slice_im.shape[:2],
                "ratio_pad": (1.0, 1.0),
                "cls": slice_labels["cls"],
            }
        )

        if slice_labels["bboxes"].size == 0:
            labels["bboxes"] = np.empty((0, 4), dtype=np.float32)
        else:
            bboxes = np.array(slice_labels["bboxes"], dtype=np.float32)
            if bboxes.ndim != 2 or bboxes.shape[1] != 4:
                raise ValueError(f"Expected shape (N, 4), got {bboxes.shape}")
            labels["bboxes"] = bboxes

        if "segments" in labels:
            labels["segments"] = slice_labels.get("segments", [])
        if "keypoints" in labels:
            labels["keypoints"] = slice_labels.get("keypoints", None)

        return self.update_labels_info(labels)

    def _filter_and_transform_annotations(
        self, labels: Dict[str, Any], slice_bbox: List[int], h0: int, w0: int
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Filter and transform annotations to match current slice coordinates.

        Args:
            labels (Dict[str, Any]): Original labels dictionary.
            slice_bbox (List[int]): Bounding box of the slice in pixel coordinates [x_min, y_min, x_max, y_max].
            h0 (int): Original image height.
            w0 (int): Original image width.

        Returns:
            Dict[str, Union[np.ndarray, List]]: Transformed labels within the slice.
        """
        x_min, y_min, x_max, y_max = slice_bbox
        slice_labels = {"cls": [], "bboxes": []}

        for i in range(len(labels["bboxes"])):
            cls = labels["cls"][i]
            bbox = labels["bboxes"][i]  # (cx, cy, w, h) normalized

            cx, cy, w, h = bbox
            x1 = (cx - w / 2) * w0
            y1 = (cy - h / 2) * h0
            x2 = (cx + w / 2) * w0
            y2 = (cy + h / 2) * h0

            inter_x1 = max(x1, x_min)
            inter_y1 = max(y1, y_min)
            inter_x2 = min(x2, x_max)
            inter_y2 = min(y2, y_max)

            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                continue  # No intersection

            new_x1 = max(x1 - x_min, 0)
            new_y1 = max(y1 - y_min, 0)
            new_x2 = min(x2 - x_min, self.crop_size)
            new_y2 = min(y2 - y_min, self.crop_size)

            cx_new = (new_x1 + new_x2) / 2 / self.crop_size
            cy_new = (new_y1 + new_y2) / 2 / self.crop_size
            w_new = (new_x2 - new_x1) / self.crop_size
            h_new = (new_y2 - new_y1) / self.crop_size

            slice_labels["cls"].append(cls)
            slice_labels["bboxes"].append([cx_new, cy_new, w_new, h_new])

        slice_labels["cls"] = np.array(slice_labels["cls"], dtype=np.float32)
        slice_labels["bboxes"] = np.array(slice_labels["bboxes"], dtype=np.float32)
        return slice_labels

    def build_transforms(self, hyp: Optional[Dict[str, Any]] = None) -> Compose:
        if self.cut_strategy == "grid":
            if self.augment:
                transforms = v8_transforms(
                    dataset=self,
                    imgsz=self.crop_size,
                    hyp=hyp,
                    stretch=False,
                )
            else:
                transforms = Compose([LetterBox(new_shape=(self.crop_size, self.crop_size), scaleup=False)])
        elif self.cut_strategy == "random_crop":
            if self.augment:
                transforms = crop_transforms(
                    dataset=self,
                    imgsz=self.crop_size,
                    hyp=hyp,
                )
            else:
                transforms = crop_val_transforms(
                    dataset=self,
                    imgsz=self.crop_size,
                    hyp=hyp,
                )
        else:
            raise ValueError(f"Unknown cut strategy: {self.cut_strategy}")

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio if hyp else 0.5,
                mask_overlap=hyp.overlap_mask if hyp else False,
                bgr=hyp.bgr if hyp and self.augment else 0.0,
            )
        )

        return transforms
