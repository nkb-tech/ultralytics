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
        overlap_ratio: float = 0,  # now minimal overlap
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
            overlap_ratio (float): Minimal desired fraction of overlap between adjacent slices (used for 'grid', also calculate num of crops for 'random_crop'). Default is 0.
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
            info.append(f"\n  Minimal overlap_ratio: {self.overlap_ratio}")

        info.append(f"\n  total_images: {self.ni}")
        LOGGER.info(prefix + "".join(info))

        # slice_indices store (img_idx, slice_idx) for 'random_crop' or
        # (img_idx, slice_idx, (start_x, start_y, end_x, end_y)) for 'grid'
        self.slice_indices = self._precompute_slices()  # Список (img_idx, slice_idx)
        crop_info = (
            f"  Total crops: {len(self.slice_indices)}\n"
            f"  Avg crops/image: {self.avg_slices:.1f}\n"
            f"  Min/Max crops/image: {self.min_slices}/{self.max_slices}"
        )
        LOGGER.info(crop_info)

    def _precompute_slices(self) -> List[Tuple[int, Any]]:
        """
        Precompute (img_idx, slice_idx) pairs and collect statistics.
        For 'grid', it also computes and stores the coordinates of each slice.
        If cut_strategy is 'random_crop', returns fewer slices per image based on sampling_rate.
        """

        def _calculate_grid_params_for_image(
            idx: int, crop_size: int, min_overlap_ratio: float, load_image_func: callable
        ) -> Tuple[int, List[Tuple[int, int, int, int]]]:
            """
            Calculates optimal grid parameters and slice coordinates for a single image.
            Ensures full image coverage by adjusting overlap for the last crop if necessary.
            Returns image index and a list of (start_x, start_y, end_x, end_y) coordinates for each slice.
            """
            im, (h0, w0), _ = load_image_func(idx)

            # Determine effective crop size, respecting image dimensions
            # If image dimension is smaller than crop_size, the effective crop is the image dimension itself
            effective_crop_w = min(crop_size, w0)
            effective_crop_h = min(crop_size, h0)

            # Calculate step and number of slices for X-axis (width)
            if w0 <= effective_crop_w:
                cols = 1
                step_x = 0
            else:
                desired_step_x = int(effective_crop_w * (1 - min_overlap_ratio))
                if desired_step_x <= 0:  # Ensure step is at least 1 if overlap is too high
                    desired_step_x = 1

                # how many full steps
                num_steps_to_fit = (w0 - effective_crop_w) // desired_step_x
                remaining_width = (w0 - effective_crop_w) % desired_step_x

                if remaining_width == 0:
                    cols = num_steps_to_fit + 1
                    step_x = desired_step_x
                else:
                    cols = num_steps_to_fit + 2  # +1 for the first crop, +1 for the last partial coverage
                    step_x = (w0 - effective_crop_w) // (cols - 1)

            # Calculate step and number of slices for Y-axis (height)
            if h0 <= effective_crop_h:
                rows = 1
                step_y = 0
            else:
                desired_step_y = int(effective_crop_h * (1 - min_overlap_ratio))
                if desired_step_y <= 0:
                    desired_step_y = 1

                num_steps_to_fit = (h0 - effective_crop_h) // desired_step_y
                remaining_height = (h0 - effective_crop_h) % desired_step_y

                if remaining_height == 0:
                    rows = num_steps_to_fit + 1
                    step_y = desired_step_y
                else:
                    rows = num_steps_to_fit + 2
                    step_y = (h0 - effective_crop_h) // (rows - 1)

            # Generate coordinates for all slices
            slice_coords = []
            for r in range(rows):
                for c in range(cols):
                    start_x = c * step_x
                    start_y = r * step_y

                    end_x = start_x + effective_crop_w
                    end_y = start_y + effective_crop_h

                    if end_x > w0:
                        start_x = w0 - effective_crop_w
                        end_x = w0
                    if end_y > h0:
                        start_y = h0 - effective_crop_h
                        end_y = h0

                    start_x = max(0, start_x)
                    start_y = max(0, start_y)

                    slice_coords.append((start_x, start_y, end_x, end_y))
            return idx, slice_coords

        with ThreadPoolExecutor() as executor:
            if self.cut_strategy == "grid":
                process_fn = partial(
                    _calculate_grid_params_for_image,
                    crop_size=self.crop_size,
                    min_overlap_ratio=self.overlap_ratio,
                    load_image_func=self.load_image,
                )
                results = list(executor.map(process_fn, range(self.ni)))
            else:  # random_crop

                def _get_total_grid_slices_count(idx, crop_size, overlap_ratio, load_image_func):
                    _, slice_coords_list = _calculate_grid_params_for_image(
                        idx, crop_size, overlap_ratio, load_image_func
                    )
                    return idx, len(slice_coords_list)

                process_fn = partial(
                    _get_total_grid_slices_count,
                    crop_size=self.crop_size,
                    overlap_ratio=self.overlap_ratio,
                    load_image_func=self.load_image,
                )
                results = list(executor.map(process_fn, range(self.ni)))

        slice_indices: List[Tuple[int, Any]] = []
        slices_per_image: List[int] = []

        for idx, calculated_data in results:
            if self.cut_strategy == "random_crop":
                total_grid_slices = calculated_data  # calculated_data is the slice count
                sampled_slices = max(1, round(total_grid_slices * self.sampling_rate))
                slice_indices.extend([(idx, s) for s in range(sampled_slices)])
                slices_per_image.append(sampled_slices)
            else:  # grid
                image_slice_coords = calculated_data  # calculated_data is the list of coords
                for s_idx, coords in enumerate(image_slice_coords):
                    slice_indices.append((idx, s_idx, coords))
                slices_per_image.append(len(image_slice_coords))

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
        Generate a single slice based on precomputed coordinates.

        Args:
            index (int): Index of the slice in `self.slice_indices`.

        Returns:
            Dict[str, Any]: Dictionary containing sliced image and filtered labels.
        """
        img_idx, slice_idx, slice_bbox_coords = self.slice_indices[index]
        start_x, start_y, end_x, end_y = slice_bbox_coords
        im, (h0, w0), _ = self.load_image(img_idx)
        labels = deepcopy(self.labels[img_idx])

        slice_im = im[start_y:end_y, start_x:end_x]
        slice_bbox = [start_x, start_y, end_x, end_y]

        slice_labels = self._filter_and_transform_annotations(labels, slice_bbox, h0, w0)
        n_attrs = labels["cls"].shape[1] if len(labels["cls"]) > 0 else 1

        labels.update(
            {
                "img": slice_im,
                "ori_shape": (h0, w0),
                "resized_shape": slice_im.shape[:2],
                "ratio_pad": (1.0, 1.0),
                "cls": slice_labels["cls"],
                "original_img_idx": img_idx,  # Index of original image
                "slice_idx": slice_idx,        # Index of slice within image
                "slice_coords": slice_bbox_coords,  # Coordinates of this crop in original image
                "original_im_file": self.im_files[img_idx],  # Original image path
            
            }
        )

        if slice_labels["bboxes"].size == 0:
            labels["bboxes"] = np.empty((0, 4), dtype=np.float32)
            if labels["cls"].size == 0:
                labels["cls"] = np.empty((0, n_attrs), dtype=np.float32)
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
        x_crop_size, y_crop_size = x_max - x_min, y_max - y_min
        slice_labels = {"cls": [], "bboxes": []}
        n_attrs = labels["cls"].shape[1] if len(labels["cls"]) > 0 else 1        
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
            new_x2 = min(x2 - x_min, x_crop_size)
            new_y2 = min(y2 - y_min, y_crop_size)

            cx_new = (new_x1 + new_x2) / 2 / x_crop_size
            cy_new = (new_y1 + new_y2) / 2 / y_crop_size
            w_new = (new_x2 - new_x1) / x_crop_size
            h_new = (new_y2 - new_y1) / y_crop_size

            slice_labels["cls"].append(cls)
            slice_labels["bboxes"].append([cx_new, cy_new, w_new, h_new])

        if len(slice_labels["cls"]) > 0:
            slice_labels["cls"] = np.array(slice_labels["cls"], dtype=np.float32)
        else:
            slice_labels["cls"] = np.zeros((0, n_attrs), dtype=np.float32)
        
        slice_labels["bboxes"] = np.array(slice_labels["bboxes"], dtype=np.float32)
        if len(slice_labels["bboxes"]) == 0:
            slice_labels["bboxes"] = np.zeros((0, 4), dtype=np.float32)
        
        return slice_labels

    def build_transforms(self, hyp: Optional[Dict[str, Any]] = None) -> Compose:
        if isinstance(hyp.scale_range, str):
            hyp.scale_range = hyp.scale_range.strip("()")
            x, y = hyp.scale_range.split(",")
            hyp.scale_range = (float(x), float(y))
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
                n_cls_tasks=1 if self.single_cls else len(self.nc),
            )
        )

        return transforms
